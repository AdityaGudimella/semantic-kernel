# Copyright (c) Microsoft. All rights reserved.

import typing as t

import pydantic as pdt

from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.template_engine.blocks.block_types import BlockTypes


class Block(SKBaseModel):
    content: pdt.constr(
        strip_whitespace=True
    ) = ""  # pyright: ignore[reportGeneralTypeIssues]
    logger: SKLogger = pdt.Field(default_factory=NullLogger)
    type: BlockTypes = BlockTypes.UNDEFINED

    def is_valid(self) -> t.Tuple[bool, str]:
        return True, ""

    def render(self, variables: t.Optional[ContextVariables]) -> str:
        raise NotImplementedError("Subclasses must implement this method.")
