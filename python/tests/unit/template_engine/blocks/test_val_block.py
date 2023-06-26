# Copyright (c) Microsoft. All rights reserved.

import pydantic as pdt
import pytest

from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.blocks.val_block import ValBlock


@pytest.fixture()
def content() -> str:
    return "'test value'"


@pytest.fixture()
def val_block(content: str) -> ValBlock:
    return ValBlock(content=content)


def test_type(val_block: ValBlock):
    assert val_block.type == BlockTypes.VALUE


def test_is_valid(val_block: ValBlock):
    is_valid, error_msg = val_block.is_valid()
    assert is_valid
    assert error_msg == ""


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            "test value",
            id="no quotes",
        ),
        pytest.param(
            "test value'",
            id="single quote at end",
        ),
        pytest.param(
            "'test value",
            id="single quote at start",
        ),
        pytest.param(
            "'test value\"",
            id="Inconsistent quotes",
        ),
        pytest.param(
            '"test value',
            id="double quote at start",
        ),
        pytest.param(
            "!test value!",
            id="wrong quotes",
        ),
    ],
)
def test_validity(content: str):
    with pytest.raises(pdt.ValidationError):
        ValBlock(content=content)


def test_render():
    val_block = ValBlock(content="'test value'")
    rendered_value = val_block.render(ContextVariables())
    assert rendered_value == "test value"
