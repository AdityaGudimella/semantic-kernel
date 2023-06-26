# Copyright (c) Microsoft. All rights reserved.

import warnings
from typing import Any, Optional, Tuple

import pydantic as pdt

from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.template_engine.blocks.block import Block
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.blocks.symbols import Symbols
from semantic_kernel.template_engine.protocols.text_renderer import TextRenderer


class ValBlock(Block, TextRenderer):
    type: BlockTypes = BlockTypes.VALUE
    _first: str = pdt.PrivateAttr(default=None)
    _last: str = pdt.PrivateAttr(default=None)
    _value: str = pdt.PrivateAttr(default=None)

    @pdt.validator("content")
    def _validate_content(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise TypeError("content must be a string")
        if len(v) < 2:
            error_msg = "A value must have single quotes or double quotes on both sides"
        elif v[0] not in (Symbols.DBL_QUOTE, Symbols.SGL_QUOTE):
            error_msg = (
                "A value must be wrapped in either single quotes or double quotes"
            )
        elif v[0] != v[-1]:
            error_msg = (
                "Cannot mix single quotes and double quotes in a value definition"
            )
        else:
            return v.strip()
        error_msg += f": {v}"
        raise ValueError(error_msg)

    @property
    def first(self) -> str:
        if self._first is None:
            self._first = self.content[0]
        return self._first

    @property
    def last(self) -> str:
        if self._last is None:
            self._last = self.content[-1]
        return self._last

    @property
    def value(self) -> str:
        if self._value is None:
            self._value = self.content[1:-1]
        return self._value

    def is_valid(self) -> Tuple[bool, str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not self._validate_content(self.content):
                return False, ""

        return True, ""

    def render(self, _: Optional[ContextVariables] = None) -> str:
        return self.value
