# Copyright (c) Microsoft. All rights reserved.
import typing as t

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
from semantic_kernel.settings import AzureOpenAISettings
from semantic_kernel.utils import openai_
from semantic_kernel.utils.openai_ import OpenAIBackends, generate_embeddings_async


class AzureTextEmbedding(SKBaseModel, EmbeddingGeneratorBase):
    deployment: str = pdt.Field(
        description=(
            "Azure OpenAI deployment name. See: ?"
            + " This value will correspond to the custom name you chose for your"
            + " deployment when you deployed a model. This value can be found under"
            + " Resource Management > Deployments in the Azure portal or, alternatively,"
            + " under Management > Deployments in Azure OpenAI Studio."
        ),
        min_length=1,
    )
    settings: AzureOpenAISettings = pdt.Field(
        description="Azure OpenAI settings. See: semantic_kernel.settings.AzureOpenAISettings"  # noqa: E501
    )
    _logger: SKLogger = pdt.PrivateAttr(default_factory=NullLogger)

    async def generate_embeddings_async(self, texts: t.List[str]) -> np.ndarray:
        return await generate_embeddings_async(
            input=texts,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.deployment,
            backend=OpenAIBackends.Azure,
        )


class AzureChatCompletion(
    SKBaseModel, ChatCompletionClientBase, TextCompletionClientBase
):
    deployment: str = pdt.Field(
        description=(
            "Azure OpenAI deployment name. See: ?"
            + " This value will correspond to the custom name you chose for your"
            + " deployment when you deployed a model. This value can be found under"
            + " Resource Management > Deployments in the Azure portal or, alternatively,"
            + " under Management > Deployments in Azure OpenAI Studio."
        ),
        min_length=1,
    )
    settings: AzureOpenAISettings = pdt.Field(
        description="Azure OpenAI settings. See: semantic_kernel.settings.AzureOpenAISettings"  # noqa: E501
    )

    async def complete_chat_async(
        self,
        messages: t.List[openai_.RoleMessage],
        request_settings: ChatRequestSettings,
    ) -> t.Union[str, t.List[str]]:
        # TODO: tracking on token counts/etc.
        return await openai_.complete_chat_async(
            messages=messages,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.deployment,
            backend=openai_.OpenAIBackends.Azure,
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
            model_or_engine=self.deployment,
            backend=openai_.OpenAIBackends.Azure,
            request_kwargs=request_settings.completion_kwargs,
        )

    async def complete_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ) -> t.Union[str, t.List[str]]:
        # TODO: tracking on token counts/etc.
        return await openai_.complete_async(
            prompt=prompt,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.deployment,
            backend=openai_.OpenAIBackends.Azure,
            request_kwargs=request_settings.completion_kwargs,
        )

    async def complete_stream_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ):
        return await openai_.complete_stream_async(
            prompt=prompt,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.deployment,
            backend=openai_.OpenAIBackends.Azure,
            request_kwargs=request_settings.completion_kwargs,
        )


class AzureTextCompletion(
    SKBaseModel,
    TextCompletionClientBase,
):
    deployment: str = pdt.Field(
        description=(
            "Azure OpenAI deployment name. See: ?"
            + " This value will correspond to the custom name you chose for your"
            + " deployment when you deployed a model. This value can be found under"
            + " Resource Management > Deployments in the Azure portal or, alternatively,"
            + " under Management > Deployments in Azure OpenAI Studio."
        )
    )
    settings: AzureOpenAISettings = pdt.Field(
        description="Azure OpenAI settings. See: semantic_kernel.settings.AzureOpenAISettings"  # noqa: E501
    )

    @pdt.validator("deployment")
    def validate_deployment(cls, v: str) -> str:
        if not v:
            raise ValueError("deployment must not be empty")
        return v

    async def complete_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ) -> t.Any:
        return await openai_.complete_async(
            prompt=prompt,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.deployment,
            backend=openai_.OpenAIBackends.Azure,
            request_kwargs=request_settings.completion_kwargs,
        )

    async def complete_stream_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ) -> t.Any:
        return await openai_.complete_stream_async(
            prompt=prompt,
            api_kwargs=self.settings.openai_api_kwargs,
            model_or_engine=self.deployment,
            backend=openai_.OpenAIBackends.Azure,
            request_kwargs=request_settings.completion_kwargs,
        )
