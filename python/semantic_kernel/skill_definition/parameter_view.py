# Copyright (c) Microsoft. All rights reserved.
import pydantic as pdt

from semantic_kernel.constants import SKILL_NAME_REGEX
from semantic_kernel.pydantic_ import SKBaseModel


class ParameterView(SKBaseModel):
    name: str = pdt.Field(..., regex=SKILL_NAME_REGEX)
    description_: str = pdt.Field(..., alias="description")
    default_value: str

    @classmethod
    def from_native_method(cls, method, name: str) -> "ParameterView":
        return cls(
            name=name,
            description_=method.__sk_function_input_description__,
            default_value=method.__sk_function_input_default_value__,
        )
