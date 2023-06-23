import copy
import json

import pydantic as pdt

from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.pydantic_ import SKBaseModel


class TestPydanticCompatibility:
    def test_valid_pydantic_field(self):
        """I should be able to use `ContextVariables` as a pydantic field type."""

        class TestModel(pdt.BaseModel):
            context_vars: ContextVariables

        test_model = TestModel(context_vars=ContextVariables("Hello, world!"))
        assert test_model.context_vars["input"] == "Hello, world!"

    def test_json_serialization(self):
        """For pydantic serialization, `ContextVariables` should serialize to a json."""
        variables = ContextVariables(input="Hello, world!", foo="bar")
        json_str = json.dumps(variables, cls=ContextVariables.Encoder)
        assert isinstance(json_str, str)

    def test_pydantic_serialization(self):
        """When `ContextVariables` is a pydantic field type on a pydantic BaseModel
        subclass, the subclass must be serializable."""

        class TestModel(SKBaseModel):
            context_vars: ContextVariables

        test_model = TestModel(context_vars=ContextVariables("Hello, world!"))
        assert TestModel.parse_raw(test_model.json()) == test_model


def test_context_vars_contain_single_var_by_default():
    context_vars = ContextVariables()
    assert context_vars._variables is not None
    assert len(context_vars._variables) == 1
    assert context_vars._variables["input"] == ""


def test_context_vars_can_be_constructed_with_string():
    content = "Hello, world!"
    context_vars = ContextVariables(content)
    assert context_vars._variables is not None
    assert len(context_vars._variables) == 1
    assert context_vars._variables["input"] == content


def test_context_vars_can_be_constructed_with_dict():
    variables = {"test_string": "Hello, world!"}
    context_vars = ContextVariables(**variables)
    assert context_vars._variables is not None
    assert len(context_vars._variables) == 2
    assert context_vars._variables["input"] == ""
    assert context_vars._variables["test_string"] == variables["test_string"]


def test_context_vars_can_be_constructed_with_string_and_dict():
    content = "I love muffins"
    variables = {"test_string": "Hello, world!"}
    context_vars = ContextVariables(content=content, **variables)
    assert context_vars._variables is not None
    assert len(context_vars._variables) == 2
    assert context_vars._variables["input"] == content
    assert context_vars._variables["test_string"] == variables["test_string"]


def test_merged_context_vars_with_empty_input_results_in_empty_input():
    content = "I love muffins"
    variables = {"test_string": "Hello, world!"}
    context_vars1 = ContextVariables(content=content)
    context_vars2 = ContextVariables(**variables)
    context_vars_combined_1with2 = copy.copy(context_vars1)
    context_vars_combined_1with2.update(context_vars2)
    context_vars_combined_2with1 = copy.copy(context_vars2)
    context_vars_combined_1with2.update(context_vars1)

    assert len(context_vars_combined_1with2) == 2
    assert context_vars_combined_1with2["input"] == ""
    assert context_vars_combined_1with2["test_string"] == variables["test_string"]

    assert context_vars_combined_2with1._variables is not None
    assert len(context_vars_combined_2with1._variables) == 2
    assert context_vars_combined_2with1._variables["input"] == ""
    assert (
        context_vars_combined_2with1._variables["test_string"]
        == variables["test_string"]
    )


def test_merged_context_vars_with_same_input_results_in_unchanged_input():
    content = "I love muffins"
    variables = {"test_string": "Hello, world!"}
    context_vars1 = ContextVariables(content=content)
    context_vars2 = ContextVariables(content=content, **variables)
    context_vars_combined_1with2 = copy.copy(context_vars1)
    context_vars_combined_1with2.update(context_vars2)
    context_vars_combined_2with1 = copy.copy(context_vars2)
    context_vars_combined_2with1.update(context_vars1)

    assert len(context_vars_combined_1with2) == 2
    assert context_vars_combined_1with2._variables["input"] == content
    assert (
        context_vars_combined_1with2._variables["test_string"]
        == variables["test_string"]
    )

    assert context_vars_combined_2with1._variables is not None
    assert len(context_vars_combined_2with1._variables) == 2
    assert context_vars_combined_2with1._variables["input"] == content
    assert (
        context_vars_combined_2with1._variables["test_string"]
        == variables["test_string"]
    )


def test_merged_context_vars_with_different_input_results_in_input_overwrite1():
    content = "I love muffins"
    content2 = "I love cupcakes"
    variables = {"test_string": "Hello, world!"}
    context_vars1 = ContextVariables(content=content)
    context_vars2 = ContextVariables(content=content2, **variables)
    context_vars_combined_1with2 = copy.copy(context_vars1)
    context_vars_combined_1with2.update(context_vars2)

    assert context_vars_combined_1with2._variables is not None
    assert len(context_vars_combined_1with2._variables) == 2
    assert (
        context_vars_combined_1with2._variables["input"]
        == context_vars2._variables["input"]
    )
    assert (
        context_vars_combined_1with2._variables["test_string"]
        == context_vars2._variables["test_string"]
    )


def test_merged_context_vars_with_different_input_results_in_input_overwrite2():
    content = "I love muffins"
    content2 = "I love cupcakes"
    variables = {"test_string": "Hello, world!"}
    context_vars1 = ContextVariables(content=content)
    context_vars2 = ContextVariables(content=content2, **variables)
    context_vars_combined_2with1 = copy.copy(context_vars2)
    context_vars_combined_2with1.update(context_vars1)

    assert context_vars_combined_2with1._variables is not None
    assert len(context_vars_combined_2with1._variables) == 2
    assert context_vars_combined_2with1._variables["input"] == context_vars1["input"]
    assert (
        context_vars_combined_2with1._variables["test_string"]
        == context_vars2._variables["test_string"]
    )


def test_can_overwrite_context_variables1():
    content = "I love muffins"
    content2 = "I love cupcakes"
    variables = {"test_string": "Hello, world!"}
    context_vars1 = ContextVariables(content=content)
    context_vars2 = ContextVariables(content=content2, **variables)
    context_vars_overwrite_1with2 = copy.copy(context_vars1)
    context_vars_overwrite_1with2.update(context_vars2)

    assert context_vars_overwrite_1with2._variables is not None
    assert len(context_vars_overwrite_1with2._variables) == len(
        context_vars2._variables
    )
    assert (
        context_vars_overwrite_1with2._variables["input"]
        == context_vars2._variables["input"]
    )
    assert (
        context_vars_overwrite_1with2._variables["test_string"]
        == context_vars2["test_string"]
    )


def test_can_overwrite_context_variables2():
    content = "I love muffins"
    content2 = "I love cupcakes"
    variables = {"test_string": "Hello, world!"}
    context_vars1 = ContextVariables(content=content)
    context_vars2 = ContextVariables(content=content2, **variables)
    context_vars_overwrite_2with1 = copy.copy(context_vars2)
    context_vars_overwrite_2with1.update(context_vars1)

    assert context_vars_overwrite_2with1._variables is not None
    assert len(context_vars_overwrite_2with1._variables) == len(
        context_vars1._variables
    )
    assert (
        context_vars_overwrite_2with1._variables["input"]
        == context_vars1._variables["input"]
    )
