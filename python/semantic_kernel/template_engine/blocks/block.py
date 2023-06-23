# Copyright (c) Microsoft. All rights reserved.

from typing import Tuple

import pydantic as pdt

from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.template_engine.blocks.block_types import BlockTypes


class Block(SKBaseModel):
    content: pdt.constr(strip_whitespace=True) = ""
    logger: SKLogger = pdt.Field(default_factory=NullLogger)
    type: BlockTypes = BlockTypes.UNDEFINED

    def is_valid(self) -> Tuple[bool, str]:
        raise NotImplementedError("Subclasses must implement this method.")
