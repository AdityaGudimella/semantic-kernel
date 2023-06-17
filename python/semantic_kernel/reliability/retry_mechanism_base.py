# Copyright (c) Microsoft. All rights reserved.

import logging
from abc import abstractmethod
from typing import Awaitable, Callable, TypeVar

from semantic_kernel.pydantic_ import PydanticField

T = TypeVar("T")


class RetryMechanismBase(PydanticField):
    @abstractmethod
    async def execute_with_retry_async(
        self, action: Callable[[], Awaitable[T]], log: logging.Logger
    ) -> Awaitable[T]:
        """Executes the given action with retry logic.

        Arguments:
            action {Callable[[], Awaitable[T]]} -- The action to retry on exception.
            log {logging.Logger} -- The logger to use.

        Returns:
            Awaitable[T] -- An awaitable that will return the result of the action.
        """
        pass
