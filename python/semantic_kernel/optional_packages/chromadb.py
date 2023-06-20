import warnings
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

try:
    import chromadb as chromadb
    from chromadb.api import API as API
    from chromadb.api.models.Collection import Collection as Collection
    from chromadb.config import Settings as Settings
except ImportError:
    warnings.warn(
        "chromadb is not installed. Please install it with `pip install chromadb`."
    )
    chromadb = MagicMock()
    config = MagicMock()
    API = MagicMock()
    Collection = MagicMock()

    if TYPE_CHECKING:
        import chromadb as chromadb
        from chromadb.api import API as API
        from chromadb.api.models.Collection import Collection as Collection
        from chromadb.config import Settings as Settings
