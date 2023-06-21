"""Import packages that are not installed by default with semantic-kernel."""
import importlib
import typing as t


def ensure_installed(
    package: str, module: t.Optional[str] = None, error_message: t.Optional[str] = None
) -> None:
    """Ensure that a package is installed, and raise an error if it is not.

    Arguments:
        package {str} -- Package name to check for.
        module {t.Optional[str]} -- Module relative to package.
            If None, package name is used.
            (default: {None})
        error_message {t.Optional[str]} -- Error message to raise if package is not
            installed.
            (default: {None})

    Keyword Arguments:
        error_message {t.Optional[str]} -- Error message to raise if package is not
            installed.
            (default: {None})

    Raises:
        ImportError: If package is not installed.
    """
    try:
        if module:
            importlib.import_module(module, package=package)
        else:
            importlib.import_module(package)
    except ImportError as e:
        error_message = error_message or (
            f"{package} is not installed. Please install it."
        )
        raise ImportError(error_message) from e
