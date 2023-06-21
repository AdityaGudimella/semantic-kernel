import typing as t
import warnings
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

try:
    import torch as torch
except ImportError:
    warnings.warn(
        "torch is not installed."
        + " See: https://pytorch.org/get-started/locally/ for installation instructions"
    )
    torch = MagicMock()

    if TYPE_CHECKING:
        import torch as torch
