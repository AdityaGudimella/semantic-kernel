# Copyright (c) Microsoft. All rights reserved.

import typing as t
from abc import abstractmethod
from typing import TYPE_CHECKING

import typing_extensions as te

from semantic_kernel.pydantic_ import PydanticField
from semantic_kernel.skill_definition.read_only_skill_collection import (
    ReadOnlySkillCollection,
)

if TYPE_CHECKING:
    from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
    from semantic_kernel.skill_definition.functions_view import FunctionsView


class SkillCollectionBase(PydanticField):
    @property
    @abstractmethod
    def read_only_skill_collection(self: te.Self) -> ReadOnlySkillCollection[te.Self]:
        pass

    @abstractmethod
    def add_semantic_function(
        self, semantic_function: "SKFunctionBase"
    ) -> "SkillCollectionBase":
        pass

    @abstractmethod
    def add_native_function(
        self, native_function: "SKFunctionBase"
    ) -> "SkillCollectionBase":
        pass

    @abstractmethod
    def has_function(self, skill_name: t.Optional[str], function_name: str) -> bool:
        pass

    @abstractmethod
    def has_semantic_function(
        self, skill_name: t.Optional[str], function_name: str
    ) -> bool:
        pass

    @abstractmethod
    def has_native_function(
        self, skill_name: t.Optional[str], function_name: str
    ) -> bool:
        pass

    @abstractmethod
    def get_semantic_function(
        self, skill_name: t.Optional[str], function_name: str
    ) -> "SKFunctionBase":
        pass

    @abstractmethod
    def get_native_function(
        self, skill_name: t.Optional[str], function_name: str
    ) -> "SKFunctionBase":
        pass

    @abstractmethod
    def get_functions_view(
        self, include_semantic: bool = True, include_native: bool = True
    ) -> "FunctionsView":
        pass

    @abstractmethod
    def get_function(
        self, skill_name: t.Optional[str], function_name: str
    ) -> "SKFunctionBase":
        pass
