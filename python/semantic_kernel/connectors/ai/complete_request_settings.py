# Copyright (c) Microsoft. All rights reserved.
import typing as t

import pydantic as pdt

from semantic_kernel.semantic_functions.prompt_template_config import CompletionConfig
from semantic_kernel.utils.openai_ import CompletionKwargs


class CompleteRequestSettings(CompletionConfig):
    logprobs: t.Optional[int] = pdt.Field(
        default=None,
        description=(
            "Controls the number of logprobs to generate along with the completion."
        ),
    )

    _completion_kwargs: CompletionKwargs = pdt.PrivateAttr(default=None)

    @property
    def completion_kwargs(self) -> CompletionKwargs:
        if not self._completion_kwargs:
            self._completion_kwargs = CompletionKwargs.from_orm(self)
        return self._completion_kwargs
