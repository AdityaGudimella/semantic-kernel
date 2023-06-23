# Copyright (c) Microsoft. All rights reserved.

from abc import abstractmethod
from typing import TYPE_CHECKING, List

from semantic_kernel.pydantic_ import PydanticField

if TYPE_CHECKING:
    from semantic_kernel.orchestration.sk_context import SKContext
    from semantic_kernel.skill_definition.parameter_view import ParameterView


class PromptTemplateBase(PydanticField):
    @abstractmethod
    def get_parameters(self) -> List["ParameterView"]:
        pass

    @abstractmethod
    async def render_async(self, context: "SKContext") -> str:
        pass
