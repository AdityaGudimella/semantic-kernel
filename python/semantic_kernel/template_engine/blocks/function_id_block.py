# Copyright (c) Microsoft. All rights reserved.

import re
from typing import Optional, Tuple

import pydantic as pdt

from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.template_engine.blocks.block import Block
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.protocols.text_renderer import TextRenderer


class FunctionIdBlock(Block, TextRenderer):
    VALID_CONTENT_REGEX = r"^[a-zA-Z0-9_.]*$"
    type: BlockTypes = BlockTypes.FUNCTION_ID
    content: pdt.constr(
        strip_whitespace=True, min_length=1, regex=VALID_CONTENT_REGEX
    )  # pyright: ignore[reportGeneralTypeIssues]

    _skill_name: str = pdt.PrivateAttr(default=None)
    _function_name: str = pdt.PrivateAttr(default=None)

    @pdt.validator("content")
    def _validate_content(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) > 2:
            raise ValueError("The function identifier can contain max one '.' ")
        if len(parts) == 2:
            if not parts[0]:
                raise ValueError(
                    "If no skill name is provided, the function identifier must not"
                    + " start with a '.'"
                )
            if not parts[1]:
                raise ValueError("The function identifier is empty")
        return v

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

    def is_valid(self) -> Tuple[bool, str]:
        if self.content is None or len(self.content) == 0:
            error_msg = "The function identifier is empty"
            return False, error_msg

        if not re.match(self.VALID_CONTENT_REGEX, self.content):
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
        return (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
