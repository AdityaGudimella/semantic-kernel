# Copyright (c) Microsoft. All rights reserved.

from typing import List

import pydantic as pdt

from semantic_kernel.constants import FUNCTION_NAME_REGEX
from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.skill_definition.parameter_view import ParameterView


class FunctionView(SKBaseModel):
    name: str = pdt.Field(..., regex=FUNCTION_NAME_REGEX)
    skill_name: str
    description: str
    is_semantic: bool
    is_asynchronous: bool
    parameters: List[ParameterView]
