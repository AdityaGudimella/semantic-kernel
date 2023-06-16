"""Tests for the pyproject.toml file."""
import typing as t

import pytest
import toml

from semantic_kernel import PYTHON_REPO_ROOT, __version__


@pytest.fixture()
def pyproject_config() -> dict[str, t.Any]:
    """Load the pyproject.toml file."""
    return toml.load(PYTHON_REPO_ROOT / "pyproject.toml")


def test_semantic_kernel_version(pyproject_config: dict[str, t.Any]) -> None:
    """Ensure that the version of sk in the pyproject.toml file matches that in the
    __init__.py file.
    """
    poetry = pyproject_config["tool"]["poetry"]
    assert poetry["version"] == __version__
