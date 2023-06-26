# Copyright (c) Microsoft. All rights reserved.
import typing as t

import pydantic as pdt

from semantic_kernel.semantic_functions.prompt_template_config import CompletionConfig
from semantic_kernel.utils.openai_ import ChatCompletionKwargs


class ChatRequestSettings(CompletionConfig):
    _completion_kwargs: t.Optional[ChatCompletionKwargs] = pdt.PrivateAttr(default=None)

    @property
    def completion_kwargs(self) -> ChatCompletionKwargs:
        if self._completion_kwargs is None:
            self._completion_kwargs = ChatCompletionKwargs.from_orm(self)
        return self._completion_kwargs
