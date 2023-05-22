# Copyright (c) Microsoft. All rights reserved.

from logging import Logger
from typing import Any, Union, Tuple


class NullLogger(Logger):
    """
    A logger that does nothing.
    """

    def __init__(self) -> None:
        self.name = "NullLogger"

    def debug(self, _: str) -> None:
        pass

    def info(self, _: str) -> None:
        pass

    def warning(self, _: str) -> None:
        pass

    def error(self, _: str) -> None:
        pass

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return NullLogger, ()
