# Copyright (c) Microsoft. All rights reserved.

from typing import Tuple

import pydantic as pdt

from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.template_engine.blocks.block_types import BlockTypes


class Block(pdt.BaseModel):
    content: str = ""
    logger: SKLogger = pdt.Field(default_factory=NullLogger)
    type: BlockTypes = BlockTypes.UNDEFINED

    class Config:
        allow_mutation = False

    def is_valid(self) -> Tuple[bool, str]:
        raise NotImplementedError("Subclasses must implement this method.")
