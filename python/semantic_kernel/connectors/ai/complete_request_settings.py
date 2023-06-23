# Copyright (c) Microsoft. All rights reserved.

import pydantic as pdt

from semantic_kernel.semantic_functions.prompt_template_config import CompletionConfig


class CompleteRequestSettings(CompletionConfig):
    logprobs: int = pdt.Field(
        default=0,
        ge=0,
        description=(
            "Controls the number of logprobs to generate along with the completion."
        ),
    )
