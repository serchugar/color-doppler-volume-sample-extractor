import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

import torch
from dotenv import load_dotenv


_root: Final = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

load_dotenv()


def _get_env_var(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(
            f"{name} environment variable is not set\nMake sure that the .env file exists in your project root"
        )
    return value


@dataclass
class Consts:
    ROOT: ClassVar[Path] = _root
    DEVICE: ClassVar[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Secrets:
    DATA_DIR: ClassVar[Path] = Path(_get_env_var("DATA_DIR"))
    LABELED_DATA_DIR: ClassVar[Path] = Path(_get_env_var("LABELED_DATA_DIR"))
    UNLABELED_DATA_DIR: ClassVar[Path] = Path(_get_env_var("UNLABELED_DATA_DIR"))
