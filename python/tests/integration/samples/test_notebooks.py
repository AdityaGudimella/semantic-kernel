"""Ensure that the notebooks in the samples directory run without error."""
from pathlib import Path

import papermill as pm
import pytest

from semantic_kernel import REPO_ROOT

NOTEBOOKS_DIR = REPO_ROOT / "samples" / "notebooks" / "python"


@pytest.mark.parametrize(
    "notebook_path",
    [pytest.param(path, id=path.name) for path in NOTEBOOKS_DIR.glob("*.ipynb")],
)
def test_notebook(notebook_path: Path) -> None:
    """Test the notebooks in the samples directory."""
    pm.execute_notebook(
        notebook_path,
        None,
    )
