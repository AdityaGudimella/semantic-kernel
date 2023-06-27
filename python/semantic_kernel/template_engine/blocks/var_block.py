# Copyright (c) Microsoft. All rights reserved.

import re
import warnings
from typing import Any, Optional, Tuple

import pydantic as pdt

from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.template_engine.blocks.block import Block
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.blocks.symbols import Symbols
from semantic_kernel.template_engine.protocols.text_renderer import TextRenderer


class VarBlock(Block, TextRenderer):
    content: pdt.constr(  # pyright: ignore[reportGeneralTypeIssues]
        strip_whitespace=True,
        regex=r"^\$[a-zA-Z0-9_]*$",
    ) = ""
    type: BlockTypes = BlockTypes.VARIABLE
    _name: str = pdt.PrivateAttr(default=None)

    @pdt.validator("content")
    def _validate_content(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise TypeError("content must be a string")
        if len(v) < 2:
            error_msg = (
                f"A variable must start with the symbol {Symbols.VAR_PREFIX}"
                + " and have a name"
            )
        elif v[0] != Symbols.VAR_PREFIX:
            error_msg = f"A variable must start with the symbol {Symbols.VAR_PREFIX}"
        else:
            return v.strip()
        raise ValueError(error_msg)

    @property
    def name(self) -> str:
        if self._name is None:
            self._name = self.content[1:]
        return self._name

    def is_valid(self) -> Tuple[bool, str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not self._validate_content(self.content):
                return False, ""

        if not re.match(r"^[a-zA-Z0-9_]*$", self.name):
            error_msg = (
                f"The variable name '{self.name}' contains invalid characters. "
                "Only alphanumeric chars and underscore are allowed."
            )
            self.logger.error(error_msg)
            return False, error_msg

        return True, ""

    def render(self, variables: Optional[ContextVariables] = None) -> str:
        if variables is None:
            return ""

        if not self.name:
            error_msg = "Variable rendering failed, the variable name is empty"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            value = variables[self.name]
        except KeyError:
            warnings.warn(f"Variable `{Symbols.VAR_PREFIX}{self.name}` not found")
            return ""
        else:
            return value
