# This allows to do "from config import Consts, Secrets"
from config.settings import Consts, Secrets


# __all__ defines what gets imported when doing "from config import *"
__all__ = ["Consts", "Secrets"]
