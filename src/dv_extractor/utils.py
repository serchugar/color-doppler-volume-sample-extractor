import random

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
