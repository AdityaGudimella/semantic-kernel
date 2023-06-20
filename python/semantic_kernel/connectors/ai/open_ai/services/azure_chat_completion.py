# Copyright (c) Microsoft. All rights reserved.

import typing as t

import pydantic as pdt

from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import (
    OpenAIChatCompletion,
)
from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.settings import AzureOpenAISettings


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
        )
    )
    settings: AzureOpenAISettings = pdt.Field(
        description="Azure OpenAI settings. See: semantic_kernel.settings.AzureOpenAISettings"  # noqa: E501
    )
    _openai_chat_completion: OpenAIChatCompletion = pdt.PrivateAttr()

    def __init__(self, **data: t.Any) -> None:
        super().__init__(**data)
        self._openai_chat_completion = OpenAIChatCompletion(
            model_id=self.deployment,
            settings=self.settings.openai_settings,
        )

    # TODO: Figure out expected return type hint
    async def complete_chat_async(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return super().complete_chat_async(*args, **kwargs)

    # TODO: Figure out expected return type hint
    async def complete_chat_stream_async(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return super().complete_chat_stream_async(*args, **kwargs)

    # TODO: Figure out expected return type hint
    async def complete_async(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return super().complete_async(*args, **kwargs)

    # TODO: Figure out expected return type hint
    async def complete_stream_async(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return super().complete_stream_async(*args, **kwargs)
