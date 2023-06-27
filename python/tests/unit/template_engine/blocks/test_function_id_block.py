# Copyright (c) Microsoft. All rights reserved.

import pydantic as pdt
import pytest

from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.blocks.function_id_block import FunctionIdBlock


@pytest.mark.parametrize(
    "content, expected",
    [
        pytest.param("skill", "skill", id="Without function identifier"),
        pytest.param("skill.function", "skill.function", id="With function identifier"),
        pytest.param("   skill.function  ", "skill.function", id="With spaces"),
        pytest.param("0", "0", id="test_zero"),
        pytest.param("1", "1", id="test_one"),
        pytest.param("a", "a", id="test_letter_a"),
        pytest.param("_", "_", id="test_underscore"),
        pytest.param("01", "01", id="test_zero_one"),
        pytest.param("01a", "01a", id="test_zero_one_a"),
        pytest.param("a01", "a01", id="test_a_zero_one"),
        pytest.param("_0", "_0", id="test_underscore_zero"),
        pytest.param("a01_", "a01_", id="test_a_zero_one_underscore"),
        pytest.param("_a01", "_a01", id="test_underscore_a_zero_one"),
        pytest.param("a.b", "a.b", id="test_letter_a_dot_b"),
    ],
)
def test_initialization(content: str, expected: str):
    function_id_block = FunctionIdBlock(content=content)
    assert function_id_block.content == expected
    assert function_id_block.type == BlockTypes.FUNCTION_ID


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("", id="Empty"),
        pytest.param(" ", id="Space"),
        pytest.param("skill.nope.function", id="Invalid function identifier"),
        pytest.param("-", id="hyphen"),
        pytest.param(".", id="dot"),
        pytest.param("a b", id="space"),
        pytest.param("a\nb", id="newline"),
        pytest.param("a\tb", id="tab"),
        pytest.param("a\rb", id="carriage_return"),
        pytest.param("a,b", id="comma"),
        pytest.param("a-b", id="dash"),
        pytest.param("a+b", id="plus"),
        pytest.param("a~b", id="tilde"),
        pytest.param("a`b", id="backtick"),
        pytest.param("a!b", id="exclamation"),
        pytest.param("a@b", id="at"),
        pytest.param("a#b", id="hash"),
        pytest.param("a$b", id="dollar"),
        pytest.param("a%b", id="percent"),
        pytest.param("a^b", id="caret"),
        pytest.param("a*b", id="asterisk"),
        pytest.param("a(b", id="open_parenthesis"),
        pytest.param("a)b", id="close_parenthesis"),
        pytest.param("a|b", id="pipe"),
        pytest.param("a{b", id="open_brace"),
        pytest.param("a}b", id="close_brace"),
        pytest.param("a[b", id="open_bracket"),
        pytest.param("a]b", id="close_bracket"),
        pytest.param("a:b", id="colon"),
        pytest.param("a;b", id="semicolon"),
        pytest.param("a'b", id="single_quote"),
        pytest.param('a"b', id="double_quote"),
        pytest.param("a<b", id="less_than"),
        pytest.param("a>b", id="greater_than"),
        pytest.param("a/b", id="forward_slash"),
        pytest.param("a\\b", id="backslash"),
    ],
)
def test_invalid_content_raises_error(content: str):
    with pytest.raises(pdt.ValidationError):
        FunctionIdBlock(content=content)


def test_render(context_vars: ContextVariables):
    function_id_block = FunctionIdBlock(content="skill.function")
    rendered_value = function_id_block.render(context_vars)
    assert rendered_value == "skill.function"
