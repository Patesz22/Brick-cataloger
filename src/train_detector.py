import os
import glob
import math
import random
import shutil
import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from torchvision.io import read_image, write_jpeg
from ultralytics import YOLO

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DATA_DIR = os.path.join(ROOT_DIR, "training_data")
YOLO_DATA_DIR = os.path.join(ROOT_DIR, "detector_dataset")
MODEL_OUT_DIR = os.path.join(ROOT_DIR, "models")


class GPUSceneSynthesizer:
    """
    Accelerated synthetic training scene generator utilizing CUDA tensors
    for procedural background generation, rotation, scaling, and alpha blending.
    """

    def __init__(self, device: torch.device | None = None, img_size: int = 640):
        """
        Initializes the synthesizer, loads piece paths, and pre-allocates GPU buffers.

        @parameters:
            @param device: torch.device | None - CUDA or CPU device target.
            @param img_size: int - Output canvas square dimension.
        @returns:
            None
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self.sample_files = glob.glob(os.path.join(TRAIN_DATA_DIR, "*", "*.jpg"))

        if not self.sample_files:
            raise FileNotFoundError(f"No rendered images found in '{TRAIN_DATA_DIR}'. Generate training data first.")

        print(f"[SYNTHESIZER] Initialized on {self.device}. Catalog pool: {len(self.sample_files)} renders.")

    def generate_procedural_background_gpu(self) -> torch.Tensor:
        """
        Synthesizes patterned desk mats, camo blobs, or wood grain directly in VRAM.

        @parameters:
            None
        @returns:
            torch.Tensor - Background tensor (3, H, W) in range [0, 255] as float32.
        """
        bg_style = random.choice(["camo", "wood", "stripes", "solid_light"])
        h, w = self.img_size, self.img_size

        if bg_style == "camo":
            # Generate coarse random latent blobs and upsample with bicubic interpolation
            low_res = torch.randint(20, 200, (1, 3, 16, 16), device=self.device, dtype=torch.float32)
            smooth = TF.resize(low_res, [h, w], interpolation=TF.InterpolationMode.BICUBIC)
            # Add Gaussian smoothing on GPU
            bg = TF.gaussian_blur(smooth, kernel_size=[31, 31], sigma=[8.0, 8.0])

        elif bg_style == "stripes":
            base_color = torch.tensor([random.randint(180, 240), random.randint(180, 240), random.randint(180, 240)],
                                      device=self.device, dtype=torch.float32).view(3, 1, 1)
            stripe_color = torch.tensor([random.randint(30, 80), random.randint(30, 80), random.randint(60, 120)],
                                        device=self.device, dtype=torch.float32).view(3, 1, 1)

            x_grid = torch.linspace(0, 10 * math.pi, w, device=self.device).view(1, 1, w)
            y_grid = torch.linspace(0, 10 * math.pi, h, device=self.device).view(1, h, 1)
            wave = torch.sin(x_grid + y_grid)
            mask = (wave > 0.2).float()
            bg = base_color * (1.0 - mask) + stripe_color * mask

        elif bg_style == "wood":
            base = torch.tensor([random.randint(160, 200), random.randint(110, 150), random.randint(70, 100)],
                                device=self.device, dtype=torch.float32).view(3, 1, 1)
            noise = torch.randn((1, h, w), device=self.device) * 14.0
            bg = torch.clamp(base + noise, 0.0, 255.0)

        else:
            base = torch.tensor([random.randint(220, 250), random.randint(220, 250), random.randint(220, 250)],
                                device=self.device, dtype=torch.float32).view(3, 1, 1)
            noise = torch.randn((3, h, w), device=self.device) * 4.0
            bg = torch.clamp(base + noise, 0.0, 255.0)

        return bg.squeeze(0) if bg.dim() == 4 else bg

    def _load_and_isolate_piece_gpu(self, file_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a rendered piece to GPU and masks out the white background into an alpha channel.

        @parameters:
            @param file_path: str - Path to rendered JPG.
        @returns:
            tuple[torch.Tensor, torch.Tensor] - (rgb_tensor, alpha_mask) on device.
        """
        img_cpu = read_image(file_path)  # (3, H, W) uint8
        img_gpu = img_cpu.to(self.device, dtype=torch.float32)

        # Build alpha mask: pixels below threshold are plastic foreground
        white_thresh = 228.0
        is_bg = (img_gpu[0] > white_thresh) & (img_gpu[1] > white_thresh) & (img_gpu[2] > white_thresh)
        alpha = (~is_bg).float().unsqueeze(0)  # (1, H, W)

        return img_gpu, alpha

    def synthesize_single_scene(self) -> tuple[torch.Tensor, list[str]]:
        """
        Synthesizes a full multi-piece clustered scene on GPU and computes YOLO annotations.

        @parameters:
            None
        @returns:
            tuple[torch.Tensor, list[str]] - (Composited (3, H, W) uint8 tensor, YOLO labels list).
        """
        bg = self.generate_procedural_background_gpu()
        h_canvas, w_canvas = self.img_size, self.img_size
        labels = []

        # Cluster between 3 and 8 pieces per scene
        num_pieces = random.randint(3, 8)
        base_x = random.randint(100, w_canvas - 200)
        base_y = random.randint(100, h_canvas - 200)

        for p_idx in range(num_pieces):
            sample_p = random.choice(self.sample_files)
            rgb, alpha = self._load_and_isolate_piece_gpu(sample_p)

            # Combine to (4, H, W) for joint geometric transformation
            rgba = torch.cat([rgb, alpha], dim=0)

            # 1. Random GPU scaling
            scale_factor = random.uniform(0.45, 0.85)
            new_h = max(24, int(rgba.shape[1] * scale_factor))
            new_w = max(24, int(rgba.shape[2] * scale_factor))
            rgba = TF.resize(rgba, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)

            # 2. Random GPU rotation (expand canvas to avoid clipping corners)
            angle = random.uniform(0.0, 360.0)
            rgba = TF.rotate(rgba, angle, interpolation=TF.InterpolationMode.BILINEAR, expand=True)

            piece_rgb = rgba[:3]
            piece_alpha = rgba[3:4]
            _, ph, pw = piece_rgb.shape

            if ph >= h_canvas or pw >= w_canvas:
                continue

            # 3. Position clustering: Place adjacent/touching parts near base coordinate
            if p_idx == 0:
                pos_x = base_x
                pos_y = base_y
            else:
                pos_x = base_x + random.randint(-pw + 10, pw - 10)
                pos_y = base_y + random.randint(-ph + 10, ph - 10)

            pos_x = max(10, min(w_canvas - pw - 10, pos_x))
            pos_y = max(10, min(h_canvas - ph - 10, pos_y))

            # 4. Vectorized alpha blending in VRAM: C_out = C_fg * alpha + C_bg * (1 - alpha)
            target_slice = bg[:, pos_y:pos_y + ph, pos_x:pos_x + pw]
            blended = piece_rgb * piece_alpha + target_slice * (1.0 - piece_alpha)
            bg[:, pos_y:pos_y + ph, pos_x:pos_x + pw] = blended

            # 5. Extract exact non-zero bounding box from the alpha channel
            nonzero_indices = torch.nonzero(piece_alpha.squeeze(0) > 0.1)
            if nonzero_indices.numel() == 0:
                continue

            y_min = nonzero_indices[:, 0].min().item()
            y_max = nonzero_indices[:, 0].max().item()
            x_min = nonzero_indices[:, 1].min().item()
            x_max = nonzero_indices[:, 1].max().item()

            box_w = max(12, x_max - x_min)
            box_h = max(12, y_max - y_min)
            box_cx = pos_x + x_min + (box_w / 2.0)
            box_cy = pos_y + y_min + (box_h / 2.0)

            # Convert to YOLO normalized format: [class xc yc w h]
            xc_norm = box_cx / float(w_canvas)
            yc_norm = box_cy / float(h_canvas)
            w_norm = box_w / float(w_canvas)
            h_norm = box_h / float(h_canvas)

            labels.append(f"0 {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        final_img = torch.clamp(bg, 0.0, 255.0).to(torch.uint8).cpu()
        return final_img, labels

    def build_dataset(self, num_train: int = 350, num_val: int = 70) -> str:
        """
        Generates full synthetic datasets on GPU and saves them to disk.

        @parameters:
            @param num_train: int - Number of training scenes.
            @param num_val: int - Number of validation scenes.
        @returns:
            str - Path to the dataset configuration YAML.
        """
        for split in ["train", "val"]:
            os.makedirs(os.path.join(YOLO_DATA_DIR, "images", split), exist_ok=True)
            os.makedirs(os.path.join(YOLO_DATA_DIR, "labels", split), exist_ok=True)

        splits = [("train", num_train), ("val", num_val)]

        for split_name, count in splits:
            print(f"[GPU PIPELINE] Generating {count} synthetic scenes for split '{split_name}'...")
            for idx in range(count):
                img_tensor, labels = self.synthesize_single_scene()

                img_path = os.path.join(YOLO_DATA_DIR, "images", split_name, f"scene_{idx:05d}.jpg")
                lbl_path = os.path.join(YOLO_DATA_DIR, "labels", split_name, f"scene_{idx:05d}.txt")

                write_jpeg(img_tensor, img_path, quality=90)
                with open(lbl_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(labels))

                if (idx + 1) % 50 == 0 or (idx + 1) == count:
                    print(f" -> Processed {idx + 1}/{count} frames on CUDA...")

        yaml_path = os.path.join(YOLO_DATA_DIR, "lego_detector.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {os.path.abspath(YOLO_DATA_DIR)}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n")
            f.write("names:\n  0: lego_piece\n")

        print(f"[READY] Synthetic dataset written to '{YOLO_DATA_DIR}'.")
        return yaml_path


def train_yolo_localizer(yaml_path: str, epochs: int = 25) -> str:
    """
    Trains the class-agnostic YOLOv8 piece localizer on GPU.

    @parameters:
        @param yaml_path: str - Path to dataset YAML configuration.
        @param epochs: int - Training epochs.
    @returns:
        str - Path to best checkpoint weights.
    """
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    out_weights = os.path.join(MODEL_OUT_DIR, "lego_detector.pt")

    print(f"\n[TRAIN] Launching YOLOv8n detector training on GPU for {epochs} epochs...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        patience=8,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project=MODEL_OUT_DIR,
        name="run_yolo_lego"
    )

    best_pt = os.path.join(MODEL_OUT_DIR, "run_yolo_lego", "weights", "best.pt")
    if os.path.isfile(best_pt):
        shutil.copy(best_pt, out_weights)
        print(f"[DONE] Best weights successfully saved to '{out_weights}'.")
        return out_weights

    return best_pt


if __name__ == "__main__":
    synthesizer = GPUSceneSynthesizer()
    dataset_yaml = synthesizer.build_dataset(num_train=350, num_val=70)
    train_yolo_localizer(dataset_yaml, epochs=25)
