# Copyright (c) Microsoft. All rights reserved.
import typing as t

import pytest

from semantic_kernel.logging_ import SKLogger
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.template_engine.blocks.text_block import TextBlock


def test_normal_init():
    text_block = TextBlock(content="test text")
    assert text_block.content == "test text"


@pytest.fixture()
def text_block(
    text: str,
    start_index: t.Optional[int],
    stop_index: t.Optional[int],
    test_logger: SKLogger,
) -> TextBlock:
    return TextBlock.from_text(
        text=text, start_index=start_index, stop_index=stop_index, logger=test_logger
    )


@pytest.mark.parametrize(
    "text, start_index, stop_index, expected_content",
    [
        pytest.param("test text", None, None, "test text", id="no indices"),
        pytest.param("test text", 0, None, "test text", id="start index only"),
        pytest.param("test text", None, 4, "test", id="stop index only"),
        pytest.param("test text", 0, 4, "test", id="start and stop indices"),
        pytest.param("", None, None, "", id="empty-text"),
        pytest.param(" ", None, None, " ", id="single-space"),
        pytest.param("  ", None, None, "  ", id="double-space"),
        pytest.param(" \n", None, None, " \n", id="space-newline"),
        pytest.param(" \t", None, None, " \t", id="space-tab"),
        pytest.param(" \r", None, None, " \r", id="space-carriage-return"),
    ],
)
def test_initialization_from_text(text_block: TextBlock, expected_content: str):
    assert text_block.content == expected_content


@pytest.mark.parametrize(
    "text, start_index, stop_index",
    [
        pytest.param("test text", 0, 0, id="start and stop indices equal"),
        pytest.param("test text", 1, 0, id="start index greater than stop index"),
        pytest.param("test text", -1, 0, id="negative start index"),
        pytest.param("test text", 0, -1, id="negative stop index"),
    ],
)
def test_bad_initialization_raises_error(text: str, start_index: int, stop_index: int):
    with pytest.raises(ValueError):
        TextBlock.from_text(text=text, start_index=start_index, stop_index=stop_index)


@pytest.mark.parametrize(
    "text, start_index, stop_index, exp",
    [
        pytest.param("test text", None, None, "test text", id="no indices"),
        pytest.param("", None, None, ""),
        pytest.param(" ", None, None, " "),
        pytest.param("  ", None, None, "  "),
        pytest.param(" \n", None, None, " \n"),
        pytest.param(" \t", None, None, " \t"),
        pytest.param(" \r", None, None, " \r"),
        pytest.param("test", None, None, "test"),
        pytest.param(" \nabc", None, None, " \nabc"),
        pytest.param("'x'", None, None, "'x'"),
        pytest.param('"x"', None, None, '"x"'),
        pytest.param("\"'x'\"", None, None, "\"'x'\""),
    ],
)
def test_render(text_block: TextBlock, context_vars: ContextVariables, exp: str):
    rendered_value = text_block.render(context_vars)
    assert rendered_value == exp
