import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BrickDecoupledDataset
from model import BrickNetDual

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TRAINING_DIR = os.path.join(ROOT_DIR, "training_data")
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "Bricknet_dual.pth")


def execute_training(
        data_dir: str = DEFAULT_TRAINING_DIR,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4
) -> str:
    """
    Executes training of the decoupled dual-head architecture using cosine learning rate decay.

    @parameters:
        @param data_dir: str - Filepath to root training samples directory.
        @param epochs: int - Total iteration cycles across the dataset.
        @param batch_size: int - Batch dimension for parallel mini-batch SGD.
        @param lr: float - Base AdamW learning rate parameter.
    @returns:
        str - Path to serialized state dictionary weights.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Loading decoupled dataset from '{data_dir}' on {device}...")

    dataset = BrickDecoupledDataset(data_root=data_dir, target_size=224, is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    num_parts = len(dataset.part_classes)
    num_colors = len(dataset.color_classes)
    print(f"[CLASSES] Geometry: {num_parts} part shapes | Color: {num_colors} plastic dyes.")

    model = BrickNetDual(num_parts=num_parts, num_colors=num_colors).to(device)

    criterion_part = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_color = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct_p, correct_c, total = 0.0, 0, 0, 0

        for x_geo, x_color, p_labels, c_labels in loader:
            x_geo = x_geo.to(device, non_blocking=True)
            x_color = x_color.to(device, non_blocking=True)
            p_labels = p_labels.to(device, non_blocking=True)
            c_labels = c_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            part_logits, color_logits = model(x_geo, x_color, p_labels)

            loss_p = criterion_part(part_logits, p_labels)
            loss_c = criterion_color(color_logits, c_labels)
            loss = loss_p + (0.75 * loss_c)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(p_labels)
            correct_p += (part_logits.argmax(dim=1) == p_labels).sum().item()
            correct_c += (color_logits.argmax(dim=1) == c_labels).sum().item()
            total += len(p_labels)

        scheduler.step()
        acc_p = (correct_p / total) * 100
        acc_c = (correct_c / total) * 100
        avg_loss = total_loss / total
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Part Acc: {acc_p:.1f}% | Color Acc: {acc_c:.1f}%")

    checkpoint = {
        "state_dict": model.state_dict(),
        "part_classes": dataset.part_classes,
        "color_classes": dataset.color_classes
    }
    torch.save(checkpoint, DEFAULT_MODEL_PATH)
    print(f"[SAVED] Checkpoint exported to '{DEFAULT_MODEL_PATH}'.")
    return DEFAULT_MODEL_PATH


if __name__ == "__main__":
    execute_training()
