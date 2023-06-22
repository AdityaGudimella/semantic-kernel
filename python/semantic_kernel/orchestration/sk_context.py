# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Generic, Literal, Optional, Tuple, Union

import pydantic as pdt

from semantic_kernel.kernel_exception import KernelException
from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.memory.semantic_text_memory_base import SemanticTextMemoryBase
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.pydantic_ import SKGenericModel
from semantic_kernel.skill_definition.read_only_skill_collection import (
    ReadOnlySkillCollection,
    SkillCollectionT,
)


class SKContext(SKGenericModel, Generic[SkillCollectionT]):
    """Semantic Kernel context."""

    variables: ContextVariables = pdt.Field(description="The context variables.")
    memory: SemanticTextMemoryBase = pdt.Field(description="The semantic text memory.")
    skill_collection: ReadOnlySkillCollection[SkillCollectionT] = pdt.Field(
        description="The skill collection."
    )
    _error_occurred: bool = pdt.Field(
        default=False,
        description="Whether an error occurred while executing functions in the pipeline.",
    )
    _last_exception: Optional[Exception] = pdt.Field(
        default=None,
        description="When an error occurs, this is the most recent exception.",
    )
    _last_error_description: str = pdt.Field(
        default="",
        description="The last error description.",
    )
    _logger: SKLogger = pdt.PrivateAttr(
        default_factory=NullLogger,
    )

    def fail(self, error_description: str, exception: Optional[Exception] = None):
        """
        Call this method to signal that an error occurred.
        In the usual scenarios, this is also how execution is stopped
        e.g., to inform the user or take necessary steps.

        Arguments:
            error_description {str} -- The error description.

        Keyword Arguments:
            exception {Exception} -- The exception (default: {None}).
        """
        self._error_occurred = True
        self._last_error_description = error_description
        self._last_exception = exception

    @property
    def result(self) -> str:
        """
        Print the processed input, aka the current data
        after any processing that has occurred.

        Returns:
            str -- Processed input, aka result.
        """
        return str(self._variables)

    # TODO(ADI): Why use `__setitem__`? Do we expect to use this class as a dictionary?
    def __setitem__(self, key: str, value: Any) -> None:
        """Sets a context variable.

        Arguments:
            key {str} -- The variable name.
            value {Any} -- The variable value.
        """
        self._variables[key] = value

    def __getitem__(self, key: str) -> Any:
        """
        Gets a context variable.

        Arguments:
            key {str} -- The variable name.

        Returns:
            Any -- The variable value.
        """
        return self._variables[key]

    def func(self, skill_name: str, function_name: str):
        """Access registered functions by skill + name.

        IMPORTANT:
            Not case sensitive.

        IMPORTANT:
            The function might be native or semantic, it's up to the caller handling it.

        Arguments:
            skill_name {str} -- The skill name.
            function_name {str} -- The function name.

        Returns:
            SKFunctionBase -- The function.
        """
        if self._skill_collection is None:
            raise ValueError("The skill collection hasn't been set")
        assert self._skill_collection is not None  # for type checker

        if self._skill_collection.has_native_function(skill_name, function_name):
            return self._skill_collection.get_native_function(skill_name, function_name)

        return self._skill_collection.get_semantic_function(skill_name, function_name)

    def __str__(self) -> str:
        if self._error_occurred:
            return f"Error: {self._last_error_description}"

        return self.result

    def throw_if_skill_collection_not_set(self) -> None:
        """
        Throws an exception if the skill collection hasn't been set.
        """
        if self._skill_collection is None:
            raise KernelException(
                KernelException.ErrorCodes.SkillCollectionNotSet,
                "Skill collection not found in the context",
            )

    def is_function_registered(
        self, skill_name: str, function_name: str
    ) -> Union[Tuple[Literal[True], Any], Tuple[Literal[False], None]]:
        """
        Checks whether a function is registered in this context.

        Arguments:
            skill_name {str} -- The skill name.
            function_name {str} -- The function name.

        Returns:
            Tuple[bool, SKFunctionBase] -- A tuple with a boolean indicating
            whether the function is registered and the function itself (or None).
        """
        self.throw_if_skill_collection_not_set()
        assert self._skill_collection is not None  # for type checker

        if self._skill_collection.has_native_function(skill_name, function_name):
            the_func = self._skill_collection.get_native_function(
                skill_name, function_name
            )
            return True, the_func

        if self._skill_collection.has_native_function(None, function_name):
            the_func = self._skill_collection.get_native_function(None, function_name)
            return True, the_func

        if self._skill_collection.has_semantic_function(skill_name, function_name):
            the_func = self._skill_collection.get_semantic_function(
                skill_name, function_name
            )
            return True, the_func

        return False, None
