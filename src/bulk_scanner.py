import os
import cv2
import sqlite3
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model import LegoNetDual
from db import RebrickableOfflineDB

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "legonet_dual.pth")
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


class BulkLegoScanner:
    """
    100% Local LEGO bulk piece segmenter and classifier using LegoNetDual.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, inventory_db_path: str = DEFAULT_INVENTORY_DB):
        """
        Initializes the model architecture, offline database, and normalization parameters.

        @parameters:
            @param model_path: str - Path to the trained checkpoint.
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

            self.model = LegoNetDual(
                num_parts=len(self.part_classes),
                num_colors=len(self.color_classes)
            ).to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()

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

    def extract_piece_bounding_boxes(
            self,
            img: np.ndarray,
            min_pixel_span: int = 35
    ) -> list[tuple[int, int, int, int]]:
        """
        Segments physical LEGO pieces by resizing to a standardized 1280px working canvas,
        merging pin slots/holes, and filtering by physical geometry and Non-Maximum Suppression.

        @parameters:
            @param img: np.ndarray - Source image in BGR format.
            @param min_pixel_span: int - Minimum width/height span in normalized pixels.
        @returns:
            list[tuple[int, int, int, int]] - Bounding boxes mapped to original image dimensions (x, y, w, h).
        """
        orig_h, orig_w = img.shape[:2]

        # 1. Normalize working resolution to 1280px max dimension
        target_max_dim = 1280.0
        scale = target_max_dim / max(orig_h, orig_w)
        norm_w = int(orig_w * scale)
        norm_h = int(orig_h * scale)
        norm_img = cv2.resize(img, (norm_w, norm_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(norm_img, cv2.COLOR_BGR2HSV)

        # 2. Extract multi-channel foreground cues
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        # Adaptive threshold to isolate local contrast
        thresh_adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
        )

        # High-saturation mask to pull saturated plastic (Blue, Red, Yellow)
        sat = hsv[:, :, 1]
        _, thresh_sat = cv2.threshold(sat, 45, 255, cv2.THRESH_BINARY)

        combined_mask = cv2.bitwise_or(thresh_adapt, thresh_sat)

        # 3. Fuse internal slots and pin holes into solid piece bodies
        kernel_fuse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_fuse, iterations=2)
        dilated = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)

        # Flood-fill holes inside pieces
        contours_raw, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(dilated)
        for cnt in contours_raw:
            cv2.drawContours(filled_mask, [cnt], -1, 255, -1)

        # 4. Find final candidate contours
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_boxes = []
        conf_scores = []
        total_norm_area = norm_w * norm_h

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Physical size limits on 1280px scale:
            # Drop tiny specks (< 900 px²) and massive background blankets (> 35% of frame)
            if area < 900 or area > (total_norm_area * 0.35):
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # LEGO pieces must meet minimum physical dimensions
            if max(w, h) < min_pixel_span or min(w, h) < 18:
                continue

            aspect = float(max(w, h)) / max(1, min(w, h))
            if aspect > 8.0:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue

            solidity = float(area) / hull_area
            if solidity < 0.50:
                continue

            candidate_boxes.append([x, y, w, h])
            conf_scores.append(float(solidity))

        if not candidate_boxes:
            return []

        # 5. Non-Maximum Suppression to eliminate double-boxing
        indices = cv2.dnn.NMSBoxes(
            bboxes=candidate_boxes,
            scores=conf_scores,
            score_threshold=0.50,
            nms_threshold=0.25
        )

        final_boxes = []
        inv_scale = 1.0 / scale
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = candidate_boxes[idx]

                # Map coordinates back to full resolution
                orig_x = int(bx * inv_scale)
                orig_y = int(by * inv_scale)
                orig_w_box = int(bw * inv_scale)
                orig_h_box = int(bh * inv_scale)

                orig_x = max(0, min(orig_w - 1, orig_x))
                orig_y = max(0, min(orig_h - 1, orig_y))
                orig_w_box = min(orig_w - orig_x, orig_w_box)
                orig_h_box = min(orig_h - orig_y, orig_h_box)

                final_boxes.append((orig_x, orig_y, orig_w_box, orig_h_box))

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
            pad = 12
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            active_part = part_num
            active_color = color_name

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            letterboxed = letterbox_crop_image(pil_crop, target_size=224)

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
            conf_threshold: float = 0.50
    ) -> tuple[str, list[dict]]:
        """
        Extracts parts, performs 100% local model classification, logs confident matches,
        and returns annotated scan path and detection details.

        @parameters:
            @param image_path: str - Path to the bulk tray photo.
            @param conf_threshold: float - Minimum local confidence required to accept a detection.
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
            pad = 12
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

            # Rejection Gate: Discard low-confidence background / noise crops
            if conf_p.item() < conf_threshold:
                continue

            detected_part = self.part_classes[idx_p.item()]
            detected_color = self.color_classes[idx_c.item()]
            avg_conf = (conf_p.item() + conf_c.item()) / 2.0
            source = "Local_CNN"

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
                detected_part, part_name, detected_color, color_id, element_id, float(avg_conf), source
            ))

            detections.append({
                "part_num": detected_part,
                "part_name": part_name,
                "color_name": detected_color,
                "confidence": avg_conf,
                "element_id": element_id,
                "source": source,
                "bbox": (x, y, w, h)
            })

            box_color = (0, 225, 70)
            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, box_thickness)

            label_text = f"#{detected_part} ({detected_color})"
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )

            if y - th - 10 < 0:
                tag_y1 = y
                tag_y2 = y + th + 10
                text_y = y + th + 4
            else:
                tag_y1 = y - th - 10
                tag_y2 = y
                text_y = y - 4

            tag_x1 = x
            tag_x2 = min(w_img, x + tw + 8)
            text_x = x + 4

            cv2.rectangle(img, (tag_x1, tag_y1), (tag_x2, tag_y2), box_color, -1)
            cv2.putText(
                img,
                label_text,
                (text_x, text_y),
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
    scanner = BulkLegoScanner()
    scanner.process_tray_image(os.path.join(ROOT_DIR, "tray_sample.jpg"))
