import pydantic as pdt

from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.optional_packages import ensure_installed
from semantic_kernel.optional_packages.torch import torch
from semantic_kernel.pydantic_ import SKBaseModel


class HFBaseModel(SKBaseModel):
    """Base model used by all HuggingFace pydantic models."""

    model_id: str = pdt.Field(
        description=(
            "Hugging Face model card string, see: https://huggingface.co/models"
            + " for a list of available models."
            + " Model will be downloaded if not already cached."
        )
    )
    device: int = pdt.Field(
        default=-1,
        description="Device to run the model on, -1 for CPU, 0+ for GPU.",
    )
    _logger: SKLogger = pdt.PrivateAttr(default_factory=NullLogger)

    @pdt.validator("device")
    def _validate_device(cls, v: int) -> int:
        if v < -1:
            raise ValueError("Device must be -1 for CPU or 0+ for GPU.")
        return v

    @property
    def torch_device(self) -> torch.device:
        ensure_installed(
            package="torch",
            error_message="Please install torch to use HuggingFace utilities.",
        )
        return f"cuda:{self.device}" if self.device >= 0 else "cpu"
