import warnings
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from semantic_kernel.pydantic_ import PydanticField

try:
    import transformers as transformers
    from transformers.pipelines.base import Pipeline as _Pipeline

    class Pipeline(_Pipeline, PydanticField):
        """Enable `Pipeline` to be used as a pydantic field."""

except ImportError:
    warnings.warn(
        "transformers is not installed."
        + " See: https://huggingface.co/transformers/installation.html for installation instructions"  # noqa: E501
    )
    transformers = MagicMock()
    Pipeline = MagicMock()

    if TYPE_CHECKING:
        import transformers as transformers
        from transformers.pipelines.base import Pipeline as _Pipeline

        class Pipeline(_Pipeline, PydanticField):
            """Enable `Pipeline` to be used as a pydantic field."""
