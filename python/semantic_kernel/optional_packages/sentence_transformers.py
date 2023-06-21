import warnings
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

try:
    import sentence_transformers as sentence_transformers
    from sentence_transformers import SentenceTransformer as SentenceTransformer
except ImportError:
    warnings.warn(
        "sentence_transformers is not installed."
        + " See: https://www.sbert.net for installation instructions"  # noqa: E501
    )
    sentence_transformers = MagicMock()
    SentenceTransformer = MagicMock()

    if TYPE_CHECKING:
        import sentence_transformers as sentence_transformers
        from sentence_transformers import SentenceTransformer as SentenceTransformer
