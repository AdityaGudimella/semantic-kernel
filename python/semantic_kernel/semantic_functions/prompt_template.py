# Copyright (c) Microsoft. All rights reserved.

from typing import TYPE_CHECKING, List

import pydantic as pdt

from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.semantic_functions.prompt_template_base import PromptTemplateBase
from semantic_kernel.semantic_functions.prompt_template_config import (
    PromptTemplateConfig,
)
from semantic_kernel.skill_definition.parameter_view import ParameterView
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.blocks.var_block import VarBlock
from semantic_kernel.template_engine.protocols.prompt_templating_engine import (
    PromptTemplatingEngine,
)

if TYPE_CHECKING:
    from semantic_kernel.orchestration.sk_context import SKContext


class PromptTemplate(SKBaseModel, PromptTemplateBase):
    template: str = pdt.Field(
        description="Prompt template to render",
    )
    template_engine: PromptTemplatingEngine = pdt.Field(
        description="Template engine to use for rendering",
    )
    prompt_config: PromptTemplateConfig = pdt.Field(
        description="Prompt configuration",
    )
    logger: SKLogger = pdt.Field(
        default_factory=NullLogger,
        description="Logger to use for logging",
    )

    def get_parameters(self) -> List[ParameterView]:
        seen = set()

        result = []
        for param in self.prompt_config.input.parameters:
            if param is None:
                continue

            result.append(
                ParameterView(param.name, param.description, param.default_value)
            )

            seen.add(param.name)

        blocks = self.template_engine.extract_blocks(self.template)
        for block in blocks:
            if block.type != BlockTypes.VARIABLE:
                continue
            if block is None:
                continue

            var_block: VarBlock = block  # type: ignore
            if var_block.name in seen:
                continue

            result.append(ParameterView(var_block.name, "", ""))

            seen.add(var_block.name)

        return result

    async def render_async(self, context: "SKContext") -> str:
        return await self.template_engine.render_async(self.template, context)
