# Copyright (c) Microsoft. All rights reserved.

from typing import Optional, Tuple

from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.template_engine.blocks.block import Block
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.protocols.text_renderer import TextRenderer


class TextBlock(Block, TextRenderer):
    content: str = ""
    type: BlockTypes = BlockTypes.TEXT

    @classmethod
    def from_text(
        cls,
        text: str = "",
        start_index: Optional[int] = None,
        stop_index: Optional[int] = None,
        logger: Optional[SKLogger] = None,
    ):
        logger = logger or NullLogger()
        if start_index is not None and stop_index is not None:
            if start_index >= stop_index:
                raise ValueError(
                    f"start_index ({start_index}) must be less than "
                    f"stop_index ({stop_index})"
                )

            if start_index < 0:
                raise ValueError(f"start_index ({start_index}) must be greater than 0")

            text = text[start_index:stop_index]
        elif start_index is not None:
            text = text[start_index:]
        elif stop_index is not None:
            text = text[:stop_index]
        print("--------- ", text, type(text))
        return cls(content=text, logger=logger)

    def is_valid(self) -> Tuple[bool, str]:
        return True, ""

    def render(self, _: Optional[ContextVariables] = None) -> str:
        return self.content
