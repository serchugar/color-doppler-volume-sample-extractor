import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as t


# https://docs.pytorch.org/docs/stable/notes/randomness.html


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def format_time(secs: float, sub_min_precision: int = 2) -> str:
    hours, rem = divmod(secs, 3600)
    mins, secs = divmod(rem, 60)

    if hours > 0:
        return f"{int(hours)}h {int(mins)}m {secs:.2f}s"
    elif mins > 0:
        return f"{int(mins)}m {secs:.2f}s"
    else:
        return f"{secs:.{sub_min_precision}f}s"


def visualize(img: torch.Tensor) -> None:
    t.ToPILImage()(img.cpu()).show()


def visualize_predictions(
    imgs: list[torch.Tensor] | list[Path],
    masks: list[torch.Tensor] | list[Path],
    metadata: bool = False,
    persist: bool = True,
) -> None:
    import tempfile

    import fiftyone as fo

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset = fo.Dataset("Predicted Masks", overwrite=True)

        samples = []
        for i, (img, mask) in enumerate(zip(imgs, masks, strict=True)):
            if isinstance(img, torch.Tensor):
                img_path = Path(temp_dir) / f"img_{i + 1}.png"
                img = t.ToPILImage()(img.cpu())
                img.save(img_path)
                img = img_path

            sample = fo.Sample(filepath=str(img))

            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy().astype(np.uint8)
                sample["Prediction"] = fo.Segmentation(mask=mask, label="Predicted Mask")
            else:
                sample["Prediction"] = fo.Segmentation(mask_path=str(mask), label="Predicted Mask")

            samples.append(sample)

        print("Adding samples to dataset...")
        dataset.add_samples(samples)

        active_fields = fo.DatasetAppConfig.default_active_fields(dataset)
        active_fields.paths.extend(["Prediction"])
        dataset.app_config.active_fields = active_fields

        sidebar_groups = fo.DatasetAppConfig.default_sidebar_groups(dataset)
        for g in sidebar_groups:
            if g.name != "labels":
                g.expanded = False

        dataset.app_config.sidebar_groups = sidebar_groups
        dataset.save()

        if metadata:
            dataset.compute_metadata()

        session = fo.launch_app(dataset)
        session.wait() if persist else session.wait(-1)
