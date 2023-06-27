# Copyright (c) Microsoft. All rights reserved.
import typing as t

import pytest

from semantic_kernel.orchestration.context_variables import ContextVariables


@pytest.fixture()
def variables() -> t.Dict[str, str]:
    return {"test_var": "test value"}


@pytest.fixture()
def context_vars_content() -> str:
    return "input"


@pytest.fixture()
def context_vars(
    context_vars_content: str, variables: t.Dict[str, str]
) -> ContextVariables:
    return ContextVariables(content=context_vars_content, **variables)
