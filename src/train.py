import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance

from model import LegoNetDual, FocalLoss

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TRAINING_DIR = os.path.join(ROOT_DIR, "training_data")
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "legonet_dual.pth")


def letterbox_image(image: Image.Image, target_size: int = 224) -> Image.Image:
    """
    Pads an image into a square canvas maintaining aspect ratio without stretching.

    @parameters:
        @param image: Image.Image - Input PIL image.
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


def apply_domain_randomization(image: Image.Image) -> Image.Image:
    """
    Applies synthetic background replacement, overhead glare flares, shadows, and sensor blur.

    @parameters:
        @param image: Image.Image - Input PIL image on a light background.
    @returns:
        Image.Image - Domain-randomized PIL image.
    """
    arr = np.array(image, dtype=np.float32)
    h, w = arr.shape[:2]

    bg_mask = (arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235)

    base_r = random.uniform(220, 245)
    base_g = random.uniform(220, 245)
    base_b = random.uniform(215, 240)

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    grad = (x_coords / w) * random.uniform(-15, 15) + (y_coords / h) * random.uniform(-15, 15)

    synthetic_bg = np.stack([
        np.clip(base_r + grad, 0, 255),
        np.clip(base_g + grad, 0, 255),
        np.clip(base_b + grad, 0, 255)
    ], axis=-1)

    arr[bg_mask] = synthetic_bg[bg_mask]

    if random.random() < 0.45:
        flare_x = random.randint(int(w * 0.25), int(w * 0.75))
        flare_y = random.randint(int(h * 0.25), int(h * 0.75))
        radius = random.randint(20, 50)
        dist_sq = (x_coords - flare_x) ** 2 + (y_coords - flare_y) ** 2
        glare_intensity = np.exp(-dist_sq / (2.0 * (radius ** 2))) * random.uniform(40, 90)
        arr = np.clip(arr + np.expand_dims(glare_intensity, -1), 0, 255)

    if random.random() < 0.45:
        sh_x = random.randint(0, w)
        sh_y = random.randint(0, h)
        radius = random.randint(35, 75)
        dist_sq = (x_coords - sh_x) ** 2 + (y_coords - sh_y) ** 2
        shadow_mult = 1.0 - (np.exp(-dist_sq / (2.0 * (radius ** 2))) * random.uniform(0.15, 0.40))
        arr = np.clip(arr * np.expand_dims(shadow_mult, -1), 0, 255)

    out_img = Image.fromarray(arr.astype(np.uint8))

    if random.random() < 0.35:
        out_img = out_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    if random.random() < 0.35:
        enhancer = ImageEnhance.Sharpness(out_img)
        out_img = enhancer.enhance(random.uniform(1.3, 2.2))

    return out_img


class RobustLegoDataset(Dataset):
    """
    Loads labeled samples, applies domain randomization, and decouples inputs into
    color-invariant geometry and chromatic feature streams.
    """

    def __init__(self, data_root: str):
        """
        Indexes subdirectories, builds label spaces, and prepares augmentations.

        @parameters:
            @param data_root: str - Directory containing training subfolders.
        @returns:
            None
        """
        self.data_root = data_root
        self.samples = []

        self.part_classes = sorted(list({d.split('_')[0] for d in os.listdir(data_root) if '_' in d}))
        self.color_classes = sorted(list({d.split('_')[1] for d in os.listdir(data_root) if '_' in d}))

        self.part_to_idx = {p: i for i, p in enumerate(self.part_classes)}
        self.color_to_idx = {c: i for i, c in enumerate(self.color_classes)}

        for folder in os.listdir(data_root):
            if '_' not in folder:
                continue
            p_name, c_name = folder.split('_', 1)
            folder_path = os.path.join(data_root, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(folder_path, fname),
                        self.part_to_idx[p_name],
                        self.color_to_idx[c_name]
                    ))

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.cutout = transforms.RandomErasing(p=0.45, scale=(0.05, 0.15), ratio=(0.5, 2.0), value="random")

    def __len__(self) -> int:
        """
        Returns the number of samples.

        @parameters:
            None
        @returns:
            int - Total sample count.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Produces decoupled tensors for geometry and color streams along with labels.

        @parameters:
            @param idx: int - Sample index.
        @returns:
            tuple[torch.Tensor, torch.Tensor, int, int] - (x_geo, x_color, part_idx, color_idx).
        """
        path, part_label, color_label = self.samples[idx]
        raw_img = Image.open(path).convert("RGB")

        letterboxed = letterbox_image(raw_img, target_size=224)
        randomized = apply_domain_randomization(letterboxed)

        gray = randomized.convert("L").convert("RGB")
        geo_tensor = transforms.functional.to_tensor(gray)
        geo_tensor = self.norm(geo_tensor)
        geo_tensor = self.cutout(geo_tensor)

        noise = torch.randn_like(geo_tensor) * random.uniform(0.01, 0.04)
        geo_tensor = torch.clamp(geo_tensor + noise, -3.0, 3.0)

        hsv_img = randomized.convert("HSV")
        color_tensor = transforms.functional.to_tensor(hsv_img)
        color_tensor = transforms.functional.normalize(color_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        return geo_tensor, color_tensor, part_label, color_label


def execute_training(
        data_dir: str = DEFAULT_TRAINING_DIR,
        output_model_path: str = DEFAULT_MODEL_PATH,
        epochs: int = 15,
        probe_epochs: int = 3,
        batch_size: int = 32
) -> None:
    """
    Executes a two-phase training run using Focal Loss and ArcFace.

    @parameters:
        @param data_dir: str - Root directory containing labeled folders.
        @param output_model_path: str - Target destination path for trained weights.
        @param epochs: int - Total training epochs.
        @param probe_epochs: int - Number of warm-up epochs with frozen backbone.
        @param batch_size: int - Batch size.
    @returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Training hardware: {device}")

    dataset = RobustLegoDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    print(
        f"[DATASET] Loaded {len(dataset)} items across {len(dataset.part_classes)} parts and {len(dataset.color_classes)} colors.")

    model = LegoNetDual(num_parts=len(dataset.part_classes), num_colors=len(dataset.color_classes)).to(device)

    focal_criterion = FocalLoss(gamma=2.0)
    color_criterion = nn.CrossEntropyLoss()

    print(f"\n[PHASE 1] Linear Probing ({probe_epochs} epochs) - Training classification heads...")
    model.freeze_backbone()

    optimizer = optim.AdamW([
        {"params": model.geo_embed.parameters(), "lr": 1e-3},
        {"params": model.part_head.parameters(), "lr": 1e-3},
        {"params": model.color_net.parameters(), "lr": 1e-3}
    ], weight_decay=1e-4)

    for epoch in range(probe_epochs):
        model.train()
        total_loss = 0.0

        for x_geo, x_color, p_labels, c_labels in loader:
            x_geo = x_geo.to(device)
            x_color = x_color.to(device)
            p_labels = p_labels.to(device)
            c_labels = c_labels.to(device)

            optimizer.zero_grad()
            part_logits, color_logits = model(x_geo, x_color, p_labels)

            loss_p = focal_criterion(part_logits, p_labels)
            loss_c = color_criterion(color_logits, c_labels)
            loss = loss_p + (0.75 * loss_c)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Probe Epoch [{epoch + 1:02d}/{probe_epochs:02d}] - Loss: {total_loss / len(loader):.4f}")

    print(f"\n[PHASE 2] End-to-End Fine-Tuning ({epochs - probe_epochs} epochs) - Unfreezing EfficientNet-B0...")
    model.unfreeze_backbone()

    optimizer = optim.AdamW([
        {"params": model.geo_backbone.parameters(), "lr": 1e-4},
        {"params": model.geo_embed.parameters(), "lr": 5e-4},
        {"params": model.part_head.parameters(), "lr": 5e-4},
        {"params": model.color_net.parameters(), "lr": 5e-4}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - probe_epochs, eta_min=1e-6)

    for epoch in range(probe_epochs, epochs):
        model.train()
        total_loss = 0.0

        for x_geo, x_color, p_labels, c_labels in loader:
            x_geo = x_geo.to(device)
            x_color = x_color.to(device)
            p_labels = p_labels.to(device)
            c_labels = c_labels.to(device)

            optimizer.zero_grad()
            part_logits, color_logits = model(x_geo, x_color, p_labels)

            loss_p = focal_criterion(part_logits, p_labels)
            loss_c = color_criterion(color_logits, c_labels)
            loss = loss_p + (0.75 * loss_c)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Fine-Tune Epoch [{epoch + 1:02d}/{epochs:02d}] - Loss: {total_loss / len(loader):.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "part_classes": dataset.part_classes,
        "color_classes": dataset.color_classes
    }, output_model_path)
    print(f"\n[DONE] Model weights saved to '{output_model_path}'.")


if __name__ == "__main__":
    if os.path.exists(DEFAULT_TRAINING_DIR):
        execute_training()
