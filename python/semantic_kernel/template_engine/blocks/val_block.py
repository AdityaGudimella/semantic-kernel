# Copyright (c) Microsoft. All rights reserved.

import contextlib
import warnings
from typing import Any, Optional, Tuple

import pydantic as pdt

from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.template_engine.blocks.block import Block
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.blocks.symbols import Symbols
from semantic_kernel.template_engine.protocols.text_renderer import TextRenderer


class ValBlock(Block, TextRenderer):
    _first: str = pdt.PrivateAttr(default="\0")
    _last: str = pdt.PrivateAttr(default="\0")
    _value: str = pdt.PrivateAttr(default="")

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
            return v
        error_msg += f": {v}"
        warnings.warn(error_msg)
        return v.strip()

    def __init__(self, **data: Any):
        super().__init__(**data)

        with contextlib.suppress(IndexError):
            self._first = self.content[0]
            self._last = self.content[-1]
            self._value = self.content[1:-1]

    @property
    def type(self) -> BlockTypes:
        return BlockTypes.VALUE

    def is_valid(self) -> Tuple[bool, str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not self._validate_content(self.content):
                return False, ""

        return True, ""

    def render(self, _: Optional[ContextVariables] = None) -> str:
        return self._value

    @staticmethod
    def has_val_prefix(text: Optional[str]) -> bool:
        return (
            text is not None
            and len(text) > 0
            and text[0] in (Symbols.DBL_QUOTE, Symbols.SGL_QUOTE)
        )
