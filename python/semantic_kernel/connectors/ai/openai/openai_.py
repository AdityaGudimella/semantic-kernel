# Copyright (c) Microsoft. All rights reserved.

import typing as t
from logging import Logger

import numpy as np
import pydantic as pdt

from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.chat_request_settings import ChatRequestSettings
from semantic_kernel.connectors.ai.complete_request_settings import (
    CompleteRequestSettings,
)
from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (
    EmbeddingGeneratorBase,
)
from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.settings import OpenAISettings
from semantic_kernel.utils import openai_
from semantic_kernel.utils.openai_ import generate_embeddings_async


class OpenAIChatCompletion(
    SKBaseModel, ChatCompletionClientBase, TextCompletionClientBase
):
    model_id: str = pdt.Field(
        description="OpenAI model name. See: https://platform.openai.com/docs/models"
    )
    settings: OpenAISettings = pdt.Field(
        description="OpenAI settings. See: semantic_kernel.settings.OpenAISettings"
    )
    _logger: Logger = pdt.PrivateAttr(default_factory=NullLogger)
    _messages: t.List[t.Tuple[str, str]] = pdt.PrivateAttr(default_factory=list)

    async def complete_chat_async(
        self,
        messages: t.List[openai_.RoleMessage],
        request_settings: ChatRequestSettings,
    ) -> t.Union[str, t.List[str]]:
        # TODO: tracking on token counts/etc.
        return await openai_.complete_chat_async(
            messages=messages,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.model_id,
            request_kwargs=request_settings.completion_kwargs,
        )

    async def complete_chat_stream_async(
        self,
        messages: t.List[openai_.RoleMessage],
        request_settings: ChatRequestSettings,
    ):
        return await openai_.complete_chat_stream_async(
            messages=messages,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.model_id,
            request_kwargs=request_settings.completion_kwargs,
        )

    async def complete_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ) -> t.Union[str, t.List[str]]:
        # TODO: tracking on token counts/etc.
        return await openai_.complete_async(
            prompt=prompt,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.model_id,
            request_kwargs=request_settings.completion_kwargs,
        )

    async def complete_stream_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ):
        return await openai_.complete_stream_async(
            prompt=prompt,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.model_id,
            request_kwargs=request_settings.completion_kwargs,
        )


class OpenAITextEmbedding(SKBaseModel, EmbeddingGeneratorBase):
    model_id: str = pdt.Field(
        description="OpenAI model name. See: https://platform.openai.com/docs/models"
    )
    settings: OpenAISettings = pdt.Field(
        description="OpenAI settings. See: semantic_kernel.settings.OpenAISettings"
    )
    _logger: Logger = pdt.PrivateAttr(default_factory=NullLogger)

    async def generate_embeddings_async(self, texts: t.List[str]) -> np.ndarray:
        return await generate_embeddings_async(
            input=texts,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.model_id,
        )


class OpenAITextCompletion(SKBaseModel, TextCompletionClientBase):
    model_id: str = pdt.Field(
        description="OpenAI model name. See: https://platform.openai.com/docs/models"
    )
    settings: OpenAISettings = pdt.Field(
        description="OpenAI settings. See: semantic_kernel.settings.OpenAISettings"
    )
    _logger: SKLogger = pdt.PrivateAttr(default_factory=NullLogger)

    async def complete_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ) -> t.Union[str, t.List[str]]:
        # TODO: tracking on token counts/etc.
        # TODO: complete w/ multiple...
        return await openai_.complete_async(
            prompt=prompt,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.model_id,
            request_kwargs=request_settings.completion_kwargs,
        )

    async def complete_stream_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ):
        return await openai_.complete_stream_async(
            prompt=prompt,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.model_id,
            request_kwargs=request_settings.completion_kwargs,
        )
