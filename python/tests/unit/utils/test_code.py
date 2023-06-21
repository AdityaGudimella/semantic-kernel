"""Tests related to the `semantic-kernel/utils/code.py` module."""

import ast
import inspect
import textwrap
import typing as t
from pathlib import Path
from types import ModuleType

import pytest

import semantic_kernel
from semantic_kernel.utils import code


def all_class_names(
    module: ModuleType = semantic_kernel, seen: t.Optional[t.Set[object]] = None
) -> t.Iterator[str]:
    """Get all SK classes in the semantic_kernel package.

    Args:
        module (ModuleType, optional): The module to inspect. Defaults to semantic_kernel.
        seen (set[str] | None): A set of seen objects.

    Yields:
        t.Iterator[t.Iterator[t.Any]]: An iterator of all SK classes.
    """

    def class_or_module(obj: t.Any) -> bool:
        """Check if an object is a class or module."""
        return inspect.isclass(obj) or inspect.ismodule(obj)

    seen = seen or set()
    for _, obj in inspect.getmembers(module, predicate=class_or_module):
        if obj in seen:
            continue
        seen.add(obj)
        if inspect.isclass(obj):
            yield obj.__name__
        elif inspect.ismodule(obj):
            yield from all_class_names(obj, seen=seen)


@pytest.mark.parametrize(
    "module, path, exp_module",
    [
        pytest.param(
            "code", code.PYTHON_PACKAGE_ROOT / "utils" / "code.py", code, id="code"
        ),
    ],
)
def test_import_from_path(module: str, path: Path, exp_module: ModuleType) -> None:
    """Test that a module can be imported from a path."""
    imported_module = code.import_module(module, path)
    assert imported_module.__file__ == exp_module.__file__


def test_is_nested_class() -> None:
    """Test that a class is nested."""
    source = textwrap.dedent(
        """
        class A:
            class B:
                pass
        """
    )
    tree = ast.parse(source)
    node_a = tree.body[0]
    node_b = node_a.body[0]
    assert code.is_nested_class(parent=node_a, node=node_b)
    assert not code.is_nested_class(parent=node_b, node=node_a)


def test_is_not_nested_class() -> None:
    """Test that a class is not nested."""
    source = textwrap.dedent(
        """
        class A:
            pass
        class B:
            pass
        """
    )
    tree = ast.parse(source)
    node_a, node_b, *_ = tree.body
    assert not code.is_nested_class(node_a, node_b)


def test_parent_class() -> None:
    """If a class is defined within another class, the parent class should be returned."""
    source = textwrap.dedent(
        """
        class A:
            class B:
                pass
        """
    )
    tree = ast.parse(source)
    node_a = tree.body[0]
    node_b = node_a.body[0]
    assert code.parent_class(node_b, tree.body) == node_a


def test_no_parent_class() -> None:
    """Classes that aren't nested within another class should not have a parent class."""
    source = textwrap.dedent(
        """
        class A:
            pass
        class B:
            pass
        """
    )
    tree = ast.parse(source)
    _, node_b, *_ = tree.body
    with pytest.raises(ValueError):
        code.parent_class(node_b, tree.body)


def test_is_pydantic_model() -> None:
    """Test that a class is a Pydantic model."""
    source = textwrap.dedent(
        """
        from pydantic import BaseModel

        class A(BaseModel):
            pass
        """
    )
    tree = ast.parse(source)
    node_a = tree.body[1]
    assert code.is_pydantic_model(node_a)


@pytest.mark.skip(reason="Fix this test.")
def test_all_sk_classes() -> None:
    """Test that all SK classes are returned."""
    if not set(code.all_class_names(nested=False)).issubset(set(all_class_names())):
        for cls in code.all_class_names(nested=False):
            if cls not in all_class_names():
                pytest.fail(f"{cls} not part of `semantic_kernel` package.")
