# Copyright (c) Microsoft. All rights reserved.

from re import match as re_match
from typing import Optional, Tuple

import pydantic as pdt

from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.template_engine.blocks.block import Block
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.protocols.text_renderer import TextRenderer


class FunctionIdBlock(Block, TextRenderer):
    content: pdt.constr(strip_whitespace=True, min_length=1)

    _skill_name: str = pdt.PrivateAttr(default=None)
    _function_name: str = pdt.PrivateAttr(default=None)

    @property
    def skill_name(self) -> str:
        if self._skill_name is None:
            (
                self._skill_name,
                self._function_name,
            ) = self._extract_skill_and_function_names(self.content)
        return self._skill_name

    @property
    def function_name(self) -> str:
        if self._function_name is None:
            (
                self._skill_name,
                self._function_name,
            ) = self._extract_skill_and_function_names(self.content)
        return self._function_name

    @property
    def type(self) -> BlockTypes:
        return BlockTypes.FUNCTION_ID

    def is_valid(self) -> Tuple[bool, str]:
        if self.content is None or len(self.content) == 0:
            error_msg = "The function identifier is empty"
            return False, error_msg

        if not re_match(r"^[a-zA-Z0-9_.]*$", self.content):
            # NOTE: this is not quite the same as
            # utils.validation.validate_function_name
            error_msg = (
                f"The function identifier '{self.content}' contains invalid "
                "characters. Only alphanumeric chars, underscore and a single "
                "dot are allowed."
            )
            return False, error_msg

        if self._has_more_than_one_dot(self.content):
            error_msg = (
                "The function identifier can contain max one '.' "
                "char separating skill name from function name"
            )
            return False, error_msg

        return True, ""

    def render(self, _: Optional[ContextVariables] = None) -> str:
        return self.content

    def _has_more_than_one_dot(self, value: Optional[str]) -> bool:
        if value is None or len(value) < 2:
            return False

        count = 0
        for char in value:
            if char == ".":
                count += 1
                if count > 1:
                    return True

        return False

    @staticmethod
    def _extract_skill_and_function_names(content: str) -> Tuple[str, str]:
        parts = content.split(".")
        if len(parts) > 2:
            raise ValueError(
                f"Function name: {content} can not contain more than one dot separating"
                + " the skill name from the function name"
            )
        return parts[0] if len(parts) == 2 else ""
