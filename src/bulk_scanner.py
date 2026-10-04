import os
import cv2
import sqlite3
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import BrickNetDual
from db import RebrickableOfflineDB

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "Bricknet_dual.pth")
DEFAULT_DETECTOR_PATH = os.path.join(ROOT_DIR, "models", "Brick_detector.pt")
DEFAULT_INVENTORY_DB = os.path.join(ROOT_DIR, "inventory.db")
DEFAULT_TRAINING_DIR = os.path.join(ROOT_DIR, "training_data")


def letterbox_crop_image(image: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Pads cropped contours into square canvases preserving true aspect ratios.

    @parameters:
        @param image: Image.Image - Cropped piece image.
        @param target_size: int - Output dimension in pixels.
    @returns:
        Image.Image - Square padded image.
    """
    width, height = image.size
    ratio = min((target_size - 16) / width, (target_size - 16) / height)
    new_w = max(1, int(width * ratio))
    new_h = max(1, int(height * ratio))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    canvas.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return canvas


class BulkBrickScanner:
    """
    Bulk Brick piece localizer and dual-head classifier.
    Combines YOLOv8 object detection with Marker-Controlled Watershed fallback
    and color-assisted bounding box disentanglement for adjacent/touching parts.
    """

    def __init__(
            self,
            model_path: str = DEFAULT_MODEL_PATH,
            detector_path: str = DEFAULT_DETECTOR_PATH,
            inventory_db_path: str = DEFAULT_INVENTORY_DB
    ):
        """
        Initializes the model architecture, YOLO localizer, and database interfaces.

        @parameters:
            @param model_path: str - Path to the trained BrickNetDual classifier weights.
            @param detector_path: str - Path to trained YOLOv8 piece localizer weights.
            @param inventory_db_path: str - Path to inventory SQLite file.
        @returns:
            None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reb_db = RebrickableOfflineDB()
        self.reb_db.sync_database()

        self.inventory_db_path = inventory_db_path
        self._init_inventory_table()

        self.model = None
        self.part_classes = []
        self.color_classes = []

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.part_classes = checkpoint["part_classes"]
            self.color_classes = checkpoint["color_classes"]

            self.model = BrickNetDual(
                num_parts=len(self.part_classes),
                num_colors=len(self.color_classes)
            ).to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()

        self.yolo_detector = None
        possible_detector_paths = [
            detector_path,
            os.path.join(ROOT_DIR, "models", "Brick_detector.pt"),
            os.path.join(ROOT_DIR, "models", "run_yolo_Brick", "weights", "best.pt")
        ]

        for p in possible_detector_paths:
            if os.path.isfile(p):
                try:
                    from ultralytics import YOLO
                    self.yolo_detector = YOLO(p)
                    print(f"[DETECTOR] Loaded YOLO localizer from '{p}'.")
                    break
                except Exception as err:
                    print(f"[WARN] Failed loading detector from '{p}': {err}")

        self.norm_geo = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _init_inventory_table(self) -> None:
        """
        Creates the inventory logging table if not already present.

        @parameters:
            None
        @returns:
            None
        """
        with sqlite3.connect(self.inventory_db_path) as conn:
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS inventory_scans
                         (
                             id               INTEGER PRIMARY KEY AUTOINCREMENT,
                             part_num         TEXT,
                             part_name        TEXT,
                             color_name       TEXT,
                             color_id         INTEGER,
                             element_id       TEXT,
                             confidence       REAL,
                             detection_source TEXT,
                             scanned_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                         );
                         """)

    def _disentangle_touching_box(
            self,
            crop_bgr: np.ndarray,
            base_x: int,
            base_y: int,
            orig_w: int,
            orig_h: int
    ) -> list[tuple[int, int, int, int]]:
        """
        Inspects candidate bounding boxes. If a box contains multiple distinct plastic colors
        or separate distance transform peaks, splits it into individual piece boundaries.

        @parameters:
            @param crop_bgr: np.ndarray - Cropped bounding box image.
            @param base_x: int - Offset X in original image.
            @param base_y: int - Offset Y in original image.
            @param orig_w: int - Total image width for boundary clamping.
            @param orig_h: int - Total image height for boundary clamping.
        @returns:
            list[tuple[int, int, int, int]] - Disentangled bounding boxes.
        """
        h_crop, w_crop = crop_bgr.shape[:2]
        crop_area = h_crop * w_crop

        if crop_area < 450:
            return [(base_x, base_y, w_crop, h_crop)]

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Plastic color thresholds in HSV space
        masks = {
            "blue": (h_ch >= 95) & (h_ch <= 135) & (s_ch >= 45) & (v_ch >= 35),
            "red": ((h_ch <= 12) | (h_ch >= 168)) & (s_ch >= 50) & (v_ch >= 40),
            "black": (v_ch < 55),
            "yellow": (h_ch >= 18) & (h_ch <= 38) & (s_ch >= 60) & (v_ch >= 70),
            "green": (h_ch >= 38) & (h_ch <= 85) & (s_ch >= 50) & (v_ch >= 40)
        }

        active_colors = []
        min_piece_pixels = int(crop_area * 0.10)

        for c_name, mask_bool in masks.items():
            count = int(np.count_nonzero(mask_bool))
            if count > max(180, min_piece_pixels):
                active_colors.append((c_name, mask_bool.astype(np.uint8) * 255))

        # Multi-color separation for adjacent pieces of differing colors
        if len(active_colors) >= 2:
            split_boxes = []
            for _, mask_u8 in active_colors:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 220:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        if bw > 14 and bh > 14:
                            x_clamped = max(0, min(orig_w - bw, base_x + bx))
                            y_clamped = max(0, min(orig_h - bh, base_y + by))
                            split_boxes.append((x_clamped, y_clamped, bw, bh))

            if len(split_boxes) >= 2:
                return split_boxes

        # Distance transform separation for adjacent pieces of identical color
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, fg = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if np.count_nonzero(fg) > 300:
            dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
            _, peaks = cv2.threshold(dist, 0.55 * dist.max(), 255, 0)
            peaks_u8 = np.uint8(peaks)
            num_labels, markers = cv2.connectedComponents(peaks_u8)

            if num_labels > 2:
                markers = markers + 1
                markers[fg == 0] = 0
                markers = cv2.watershed(crop_bgr, markers)

                sub_boxes = []
                for m_idx in range(2, num_labels + 1):
                    piece_m = np.uint8(markers == m_idx)
                    cnts, _ = cv2.findContours(piece_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        if cv2.contourArea(cnt) > 220:
                            bx, by, bw, bh = cv2.boundingRect(cnt)
                            x_clamped = max(0, min(orig_w - bw, base_x + bx))
                            y_clamped = max(0, min(orig_h - bh, base_y + by))
                            sub_boxes.append((x_clamped, y_clamped, bw, bh))

                if len(sub_boxes) >= 2:
                    return sub_boxes

        return [(base_x, base_y, w_crop, h_crop)]

    def _watershed_separate_touching_boxes(
            self,
            img: np.ndarray,
            min_area: int = 400
    ) -> list[tuple[int, int, int, int]]:
        """
        Fallback segmentation using Marker-Controlled Watershed with strict size bounding.

        @parameters:
            @param img: np.ndarray - Source BGR image.
            @param min_area: int - Minimum contour area threshold in pixels.
        @returns:
            list[tuple[int, int, int, int]] - Extracted bounding boxes as (x, y, w, h).
        """
        h_img, w_img = img.shape[:2]
        total_area = w_img * h_img
        max_allowed_area = int(total_area * 0.35)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        if dist_transform.max() == 0:
            return []

        _, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(img, markers)
        boxes = []
        unique_markers = np.unique(markers)

        for m_id in unique_markers:
            if m_id <= 1:
                continue

            piece_mask = np.uint8(markers == m_id)
            contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_allowed_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                if w >= int(w_img * 0.75) or h >= int(h_img * 0.75):
                    continue

                boxes.append((x, y, w, h))

        return boxes

    def extract_piece_bounding_boxes(
            self,
            img: np.ndarray,
            conf_thresh: float = 0.20,
            iou_thresh: float = 0.65
    ) -> list[tuple[int, int, int, int]]:
        """
        Segments physical Brick pieces with area bounds and color disentanglement.

        @parameters:
            @param img: np.ndarray - Source BGR image.
            @param conf_thresh: float - YOLO confidence threshold.
            @param iou_thresh: float - Elevated NMS threshold preventing suppression of crossed pieces.
        @returns:
            list[tuple[int, int, int, int]] - Extracted individual piece bounding boxes.
        """
        h_img, w_img = img.shape[:2]
        total_area = w_img * h_img
        max_piece_area = int(total_area * 0.35)
        raw_boxes = []

        if self.yolo_detector is not None:
            results = self.yolo_detector.predict(img, conf=conf_thresh, iou=iou_thresh, verbose=False)
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box[:4]
                    x = max(0, int(x1))
                    y = max(0, int(y1))
                    w = min(w_img - x, int(x2 - x1))
                    h = min(h_img - y, int(y2 - y1))

                    if w > 16 and h > 16 and (w * h) <= max_piece_area:
                        if w < int(w_img * 0.80) and h < int(h_img * 0.80):
                            raw_boxes.append((x, y, w, h))

        if not raw_boxes:
            raw_boxes = self._watershed_separate_touching_boxes(img)

        refined_boxes = []
        for x, y, w, h in raw_boxes:
            if (w * h) > max_piece_area or w >= int(w_img * 0.80) or h >= int(h_img * 0.80):
                continue

            crop = img[y:y + h, x:x + w]
            if crop.size == 0:
                continue

            sub_boxes = self._disentangle_touching_box(crop, x, y, w_img, h_img)
            refined_boxes.extend(sub_boxes)

        final_boxes = []
        refined_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

        for bx, by, bw, bh in refined_boxes:
            box_area = bw * bh
            if box_area > max_piece_area or bw >= int(w_img * 0.80) or bh >= int(h_img * 0.80):
                continue

            is_fragment = False
            for ax, ay, aw, ah in final_boxes:
                ix1, iy1 = max(bx, ax), max(by, ay)
                ix2, iy2 = min(bx + bw, ax + aw), min(by + bh, ay + ah)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter_area = iw * ih
                if inter_area / float(box_area) > 0.70:
                    is_fragment = True
                    break

            if not is_fragment:
                final_boxes.append((bx, by, bw, bh))

        final_boxes.sort(key=lambda b: (b[1] // 80, b[0]))
        return final_boxes

    def harvest_tray_crops_for_training(
            self,
            image_path: str,
            part_num: str | None = None,
            color_name: str | None = None,
            output_dir: str = DEFAULT_TRAINING_DIR
    ) -> int:
        """
        Extracts segmented physical pieces from a bulk tray photograph into training directories.

        @parameters:
            @param image_path: str - File path of bulk tray photo.
            @param part_num: str | None - Known part identifier or None for auto-prediction.
            @param color_name: str | None - Known color string or None for auto-prediction.
            @param output_dir: str - Target training directory.
        @returns:
            int - Total saved crops.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        h_img, w_img = img.shape[:2]
        boxes = self.extract_piece_bounding_boxes(img)
        saved_samples = 0

        for idx, (x, y, w, h) in enumerate(boxes):
            pad = 8
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            letterboxed = letterbox_crop_image(pil_crop, target_size=224)

            active_part = part_num
            active_color = color_name

            if not active_part or not active_color:
                if self.model:
                    gray_img = letterboxed.convert("L").convert("RGB")
                    x_geo = self.norm_geo(transforms.functional.to_tensor(gray_img)).unsqueeze(0).to(self.device)

                    hsv_img = letterboxed.convert("HSV")
                    x_color = transforms.functional.normalize(
                        transforms.functional.to_tensor(hsv_img),
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    ).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        part_logits, color_logits = self.model(x_geo, x_color)
                        idx_p = torch.argmax(part_logits, dim=1).item()
                        idx_c = torch.argmax(color_logits, dim=1).item()

                    if not active_part:
                        active_part = self.part_classes[idx_p]
                    if not active_color:
                        active_color = self.color_classes[idx_c]
                else:
                    active_part = active_part or "unknown_part"
                    active_color = active_color or "unknown_color"

            clean_color = "".join(c for c in active_color if c.isalnum() or c in ("-", "_")).strip()
            dest_dir = os.path.join(output_dir, f"{active_part}_{clean_color}")
            os.makedirs(dest_dir, exist_ok=True)

            dest_file = f"harvested_{idx}_{active_part}_{clean_color}.jpg"
            letterboxed.save(os.path.join(dest_dir, dest_file), "JPEG", quality=95)
            saved_samples += 1

        print(f"[HARVEST] Extracted {saved_samples} crops to '{output_dir}'.")
        return saved_samples

    def process_tray_image(
            self,
            image_path: str,
            conf_threshold: float = 0.40
    ) -> tuple[str, list[dict]]:
        """
        Extracts parts, executes decoupled classification, draws bounding boxes,
        and logs results to SQLite.

        @parameters:
            @param image_path: str - Path to the bulk tray photo.
            @param conf_threshold: float - Confidence boundary below which detections are rejected.
        @returns:
            tuple[str, list[dict]] - (annotated_image_path, detections_list).
        """
        if self.model is None:
            raise RuntimeError("Model checkpoint not found. Train the model first.")

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        h_img, w_img = img.shape[:2]
        boxes = self.extract_piece_bounding_boxes(img)

        ref_dim = max(h_img, w_img)
        box_thickness = max(2, int(ref_dim / 450))
        font_scale = max(0.45, ref_dim / 1500.0)
        text_thickness = max(1, int(ref_dim / 850))

        logged_records = []
        detections = []

        for (x, y, w, h) in boxes:
            pad = 6
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            letterboxed = letterbox_crop_image(pil_crop, target_size=224)

            gray_img = letterboxed.convert("L").convert("RGB")
            x_geo = self.norm_geo(transforms.functional.to_tensor(gray_img)).unsqueeze(0).to(self.device)

            hsv_img = letterboxed.convert("HSV")
            x_color = transforms.functional.normalize(
                transforms.functional.to_tensor(hsv_img),
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                part_logits, color_logits = self.model(x_geo, x_color)
                p_prob = torch.softmax(part_logits, dim=1)[0]
                c_prob = torch.softmax(color_logits, dim=1)[0]

                conf_p, idx_p = torch.max(p_prob, dim=0)
                conf_c, idx_c = torch.max(c_prob, dim=0)

            if conf_p.item() < conf_threshold:
                continue

            detected_part = self.part_classes[idx_p.item()]
            detected_color = self.color_classes[idx_c.item()]
            avg_conf = (conf_p.item() + conf_c.item()) / 2.0

            part_info = self.reb_db.get_part_info(detected_part)
            part_name = part_info[1] if part_info else "Unknown Part"

            color_rec = self.reb_db.get_color_by_name(detected_color)
            color_id = color_rec[0] if color_rec else None

            element_id = "-"
            if color_id is not None:
                elem_tuple = self.reb_db.resolve_element_id(detected_part, color_id)
                if elem_tuple:
                    element_id = str(elem_tuple[0])

            logged_records.append((
                detected_part, part_name, detected_color, color_id, element_id, float(avg_conf), "Local_CNN"
            ))

            detections.append({
                "part_num": detected_part,
                "part_name": part_name,
                "color_name": detected_color,
                "confidence": avg_conf,
                "element_id": element_id,
                "source": "Local_CNN",
                "bbox": (x, y, w, h)
            })

            box_color = (0, 225, 70)
            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, box_thickness)

            label_text = f"#{detected_part} ({detected_color})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)

            tag_y1 = max(0, y - th - 8)
            tag_y2 = y
            cv2.rectangle(img, (x, tag_y1), (min(w_img, x + tw + 8), tag_y2), box_color, -1)
            cv2.putText(
                img,
                label_text,
                (x + 4, max(th + 2, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                text_thickness,
                cv2.LINE_AA
            )

        with sqlite3.connect(self.inventory_db_path) as conn:
            conn.executemany("""
                             INSERT INTO inventory_scans (part_num, part_name, color_name, color_id, element_id,
                                                          confidence, detection_source)
                             VALUES (?, ?, ?, ?, ?, ?, ?);
                             """, logged_records)

        out_path = os.path.join(ROOT_DIR, "annotated_scan.jpg")
        cv2.imwrite(out_path, img)
        return out_path, detections


if __name__ == "__main__":
    scanner = BulkBrickScanner()
    scanner.process_tray_image(os.path.join(ROOT_DIR, "tray_sample.jpg"))
