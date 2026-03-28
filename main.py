import random
from pathlib import Path

from config import Consts
from src.model import DynamicUNet
from src.train import train
from src.utils import seed_all


def main() -> None:

    seed = random.getrandbits(32)
    seed_all(seed)
    print(f"Seed: {seed}")

    model = DynamicUNet(in_channels=1, out_channels=1, depth=4, init_features=32)
    model.to(Consts.DEVICE)
    print(f"Model device: {model.device}\n")

    train(model, epochs=100, lr=0.001, batch_size=5, checkpoints_dir=Path("weights"))


if __name__ == "__main__":
    main()
