# Copyright (c) Microsoft. All rights reserved.
import pydantic as pdt
import pytest

from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.blocks.var_block import VarBlock


def test_type_property():
    var_block = VarBlock(content="$test_var")
    assert var_block.type == BlockTypes.VARIABLE


@pytest.fixture()
def content() -> str:
    return "$test_var"


@pytest.fixture()
def var_block(content: str) -> VarBlock:
    return VarBlock(content=content)


@pytest.mark.parametrize(
    "content, expected",
    [
        pytest.param("$test_var", "$test_var", id="Vanilla"),
        pytest.param("   $x  ", "$x", id="With spaces"),
        pytest.param("$x\n", "$x", id="With new line"),
    ],
)
def test_content_sanitization(var_block: VarBlock, expected: str):
    assert var_block.content == expected


@pytest.mark.parametrize(
    "content, expected",
    [
        pytest.param("$test_var", "test_var", id="Vanilla"),
        pytest.param("   $x  ", "x", id="With spaces"),
        pytest.param("$x\n", "x", id="With new line"),
        pytest.param("$0", "0"),
        pytest.param("$1", "1"),
        pytest.param("$a", "a"),
        pytest.param("$_", "_"),
        pytest.param("$01", "01"),
        pytest.param("$01a", "01a"),
        pytest.param("$a01", "a01"),
        pytest.param("$_0", "_0"),
        pytest.param("$a01_", "a01_"),
        pytest.param("$_a01", "_a01"),
    ],
)
def test_name(var_block: VarBlock, expected: str):
    assert var_block.name == expected


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("test_var", id="No prefix"),
        pytest.param("$test-var", id="Invalid characters"),
        pytest.param("$", id="Empty name"),
        pytest.param("."),
        pytest.param("-"),
        pytest.param("a b"),
        pytest.param("a\nb"),
        pytest.param("a\tb"),
        pytest.param("a\rb"),
        pytest.param("a.b"),
        pytest.param("a,b"),
        pytest.param("a-b"),
        pytest.param("a+b"),
        pytest.param("a~b"),
        pytest.param("a`b"),
        pytest.param("a!b"),
        pytest.param("a@b"),
        pytest.param("a#b"),
        pytest.param("a$b"),
        pytest.param("a%b"),
        pytest.param("a^b"),
        pytest.param("a*b"),
        pytest.param("a(b"),
        pytest.param("a)b"),
        pytest.param("a|b"),
        pytest.param("a{b"),
        pytest.param("a}b"),
        pytest.param("a[b"),
        pytest.param("a]b"),
        pytest.param("a:b"),
        pytest.param("a;b"),
        pytest.param("a'b"),
        pytest.param('a"b'),
        pytest.param("a<b"),
        pytest.param("a>b"),
        pytest.param("a/b"),
        pytest.param("a\\b"),
    ],
)
def test_invalid_construction_raises_error(content: str):
    with pytest.raises(pdt.ValidationError):
        VarBlock(content=content)


@pytest.mark.parametrize(
    "variables, expected",
    [
        pytest.param({"test_var": "test_value"}, "test_value", id="With one variable"),
        pytest.param(
            {"test_var": "exp_value", "bar": "baz"},
            "exp_value",
            id="With multiple variables",
        ),
        pytest.param({}, "", id="With no variables"),
        pytest.param({"foo": "bar"}, "", id="With wrong variables"),
    ],
)
def test_rendering(
    var_block: VarBlock,
    context_vars: ContextVariables,
    expected: str,
) -> None:
    assert var_block.render(context_vars) == expected
