import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader

from config import Secrets
from src.dataset import DopplerDataset, discover_images_with_mask
from src.model import DynamicUNet
from src.utils import format_time, seed_worker


# TODO: Add start_over:bool = True
def train(
    model: DynamicUNet, epochs: int, lr: float = 0.001, batch_size: int = 5, checkpoints_dir: Path | None = None
) -> None:
    if checkpoints_dir:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    device = model.device
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=60),  # type: ignore
            v2.RandomAffine(degrees=20, scale=(0.8, 1.2)),  # type: ignore
        ]
    )

    images, masks = discover_images_with_mask(Secrets.LABELED_DATA_DIR)
    dataset = DopplerDataset(images, masks, transform=transform)

    seed = torch.initial_seed()
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    model.train()
    min_loss = float("inf")
    start_time = time.perf_counter()
    for epoch in range(epochs):
        total_loss = 0
        total_dice = 0

        for batch_imgs, batch_masks in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_masks = batch_masks.to(device)

            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = criterion(logits, batch_masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice_coefficient(logits, batch_masks)

        epoch_loss = total_loss / len(train_loader)
        epoch_dice = total_dice / len(train_loader)
        total_time = time.perf_counter() - start_time

        if epoch_loss < min_loss:
            min_loss = epoch_loss
            print(
                f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f}"
                f" - Dice: {epoch_dice:.4f} - Total Time: {format_time(total_time)}"
            )

            if checkpoints_dir:
                loss_str = f"{epoch_loss:.4f}".replace(".", "p")

                model_depth_folder = checkpoints_dir / f"unet_depth{model.depth}"
                checkpoint_folder = model_depth_folder / f"loss{loss_str}_epoch{epoch + 1}_seed{seed}"

                old_folder = next(model_depth_folder.glob(f"*_seed{seed}"), None)
                if old_folder is None:
                    checkpoint_folder.mkdir(parents=True, exist_ok=True)
                else:
                    old_folder.rename(checkpoint_folder)

                filepath = checkpoint_folder / "weights.pt"
                torch.save(model.state_dict(), filepath)


def dice_coefficient(
    y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-6, apply_sigmoid: bool = True
) -> float:
    if apply_sigmoid:
        y_pred = torch.sigmoid(y_pred)

    y_pred = (y_pred > 0.5).float()

    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    intersection = (y_pred * y_true).sum()
    dice = (2.0 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

    return dice.item()
