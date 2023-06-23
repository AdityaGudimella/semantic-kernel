"""Utility functions that inspect and manipulate the semantic-kernel codebase."""

import ast
import contextlib
import typing as t
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

from semantic_kernel import PYTHON_PACKAGE_ROOT


def import_module(module: str, path: Path) -> ModuleType:
    """Imports module named `module` from `path`."""
    spec = spec_from_file_location(module, path)
    imported = module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not import {module} from {path}.")
    spec.loader.exec_module(imported)
    return imported


def import_class_from_path(class_: str, module: str, path: Path) -> t.Type[t.Any]:
    """Imports class named `class_` from `module` at `path`."""
    imported_module = import_module(module, path)
    imported_class = getattr(imported_module, class_)
    if not isinstance(imported_class, type):
        raise TypeError(f"{class_} is not a class.")
    return imported_class


def is_nested_class(node: ast.ClassDef, parent: ast.ClassDef) -> bool:
    """Returns true if `node` is a class defined within `parent`."""
    return any(isinstance(body, ast.ClassDef) and body == node for body in parent.body)


def parent_class(
    node: ast.ClassDef, classes: t.Iterable[ast.ClassDef]
) -> t.Optional[ast.ClassDef]:
    """Returns the parent class of `node` if it is defined within any one of `classes`."""
    for parent in classes:
        if is_nested_class(node, parent):
            return parent
    raise ValueError(f"{node} is not defined within any of {classes}.")


def is_pydantic_model(node: ast.ClassDef) -> bool:
    """Returns true if `node` is a Pydantic model."""
    _PYDANTIC_CLASSES = ("BaseModel", "BaseConfig", "BaseSettings", "GenericModel")

    def is_pydantic_class(node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id in _PYDANTIC_CLASSES:
            return True
        return isinstance(node, ast.Attribute) and node.attr in _PYDANTIC_CLASSES

    return any(is_pydantic_class(base) for base in node.bases)


def _all_classes(
    directory: Path = PYTHON_PACKAGE_ROOT, nested: t.Optional[bool] = None
) -> t.Iterator[t.Tuple[str, Path]]:
    """Get all classes defined within the `directory`.

    NOTE:
        If no directory is specified, this function returns all classes defined in the
        `semantic_kernel` package.

    Args:
        directory (Path, optional): The directory to inspect.
            Defaults to PYTHON_PACKAGE_ROOT, which is the `semantic_kernel` package.
        nested (bool | None, optional): If True, only return nested classes.
            If False, only return top-level classes. If None, return all classes.

    Yields:
        t.Iterator[t.Iterator[t.Any]]: An iterator of all classes.
    """
    for path in directory.glob("**/*.py"):
        with open(path, "r") as fp:
            tree = ast.parse(fp.read())
        top_level_classes = [
            node for node in tree.body if isinstance(node, ast.ClassDef)
        ]
        for node in filter(lambda x: isinstance(x, ast.ClassDef), ast.walk(tree)):
            parent_class_of_node = None
            with contextlib.suppress(ValueError):
                parent_class_of_node = parent_class(node, top_level_classes)
            if nested is True and parent_class_of_node is None:
                continue
            if nested is False and parent_class_of_node is not None:
                continue
            yield node.name, path


def all_class_names(
    directory: Path = PYTHON_PACKAGE_ROOT, nested: t.Optional[bool] = None
) -> t.Iterator[str]:
    """Get all classes defined within the `directory`.

    NOTE:
        If no directory is specified, this function returns all classes defined in the
        `semantic_kernel` package.

    Args:
        directory (Path, optional): The directory to inspect.
            Defaults to PYTHON_PACKAGE_ROOT, which is the `semantic_kernel` package.
        nested (bool | None, optional): If True, only return nested classes.
            If False, only return top-level classes. If None, return all classes.

    Yields:
        t.Iterator[t.Iterator[t.Any]]: An iterator of all classes.
    """
    for name, _ in _all_classes(directory=directory, nested=nested):
        yield name


def all_classes(
    directory: Path = PYTHON_PACKAGE_ROOT, nested: t.Optional[bool] = None
) -> t.Iterator[t.Type[t.Any]]:
    """Get all classes defined within the `directory`.

    NOTE:
        If no directory is specified, this function returns all classes defined in the
        `semantic_kernel` package.

    Args:
        directory (Path, optional): The directory to inspect.
            Defaults to PYTHON_PACKAGE_ROOT, which is the `semantic_kernel` package.
        nested (bool | None, optional): If True, only return nested classes.
            If False, only return top-level classes. If None, return all classes.

    Yields:
        t.Iterator[t.Iterator[t.Any]]: An iterator of all classes.
    """
    for name, path in _all_classes(directory=directory, nested=nested):
        try:  # TODO(ADI): Remove this try-except block.
            yield import_class_from_path(class_=name, module=path.stem, path=path)
        except Exception:
            print(f"Failed to import {name} from {path}.")
