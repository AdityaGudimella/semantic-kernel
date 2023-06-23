# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.semantic_functions.chat_prompt_template import ChatPromptTemplate
from semantic_kernel.semantic_functions.prompt_template import PromptTemplate
from semantic_kernel.semantic_functions.prompt_template_config import (
    PromptTemplateConfig,
)


class SemanticFunctionConfig(SKBaseModel):
    prompt_template_config: "PromptTemplateConfig"
    prompt_template: "PromptTemplate"

    @property
    def has_chat_prompt(self) -> bool:
        return isinstance(self.prompt_template, ChatPromptTemplate)
