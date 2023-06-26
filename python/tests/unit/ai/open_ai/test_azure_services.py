# Copyright (c) Microsoft. All rights reserved.

import typing as t
from unittest.mock import AsyncMock

import pydantic as pdt
import pytest
from pytest_mock import MockerFixture

from semantic_kernel.connectors.ai.chat_request_settings import ChatRequestSettings
from semantic_kernel.connectors.ai.complete_request_settings import (
    CompleteRequestSettings,
)
from semantic_kernel.connectors.ai.openai.azure_ import (
    AzureChatCompletion,
    AzureTextCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.logging_ import SKLogger
from semantic_kernel.settings import AzureOpenAISettings


@pytest.mark.parametrize("deployment_name", ["", None])
@pytest.mark.parametrize(
    "service",
    [
        AzureChatCompletion,
        AzureTextCompletion,
        AzureTextEmbedding,
    ],
)
def test_initialization_with_bad_params_raises_error(
    deployment_name: str,
    service: t.Union[
        t.Type[AzureChatCompletion],
        t.Type[AzureTextCompletion],
        t.Type[AzureTextEmbedding],
    ],
    azure_openai_settings: AzureOpenAISettings,
) -> None:
    with pytest.raises(pdt.ValidationError):
        service(deployment=deployment_name, settings=azure_openai_settings)


class _DotDict(dict):
    def __getattr__(self, name: str) -> t.Any:
        if name in self:
            if isinstance(self[name], dict):
                return _DotDict(self[name])
            elif isinstance(self[name], list):
                return [_DotDict(x) if isinstance(x, dict) else x for x in self[name]]
        return self[name]


class TestAPICalls:
    @pytest.fixture()
    def mock_embedding(self, mocker: MockerFixture) -> t.Iterator[AsyncMock]:
        mock_embedding = AsyncMock(
            return_value={"data": [{"embedding": [1, 2, 3]}]},
        )
        mocker.patch(
            "semantic_kernel.utils.openai_.openai.Embedding.acreate", mock_embedding
        )
        yield mock_embedding

    @pytest.mark.asyncio
    async def test_azure_embedding_api_call(
        self,
        mock_embedding: AsyncMock,
        mock_azure_openai_settings: AzureOpenAISettings,
        test_logger: SKLogger,
    ) -> None:
        deployment_name = "test_deployment"
        texts = ["hello world"]

        azure_text_embedding = AzureTextEmbedding(
            deployment=deployment_name,
            settings=mock_azure_openai_settings,
            logger=test_logger,
        )

        await azure_text_embedding.generate_embeddings_async(texts)

        mock_embedding.assert_called_once_with(
            engine=deployment_name,
            api_key=mock_azure_openai_settings.api_key.get_secret_value(),
            api_type=mock_azure_openai_settings.api_type.value,
            api_base=mock_azure_openai_settings.endpoint,
            api_version=mock_azure_openai_settings.api_version,
            organization=None,
            input=texts,
        )

    @pytest.fixture()
    def mock_completion(self, mocker: MockerFixture) -> t.Iterator[AsyncMock]:
        mock_completion = AsyncMock(
            return_value=_DotDict({"choices": [{"text": "test"}]}),
        )
        mocker.patch(
            "semantic_kernel.utils.openai_.openai.Completion.acreate", mock_completion
        )
        yield mock_completion

    @pytest.mark.asyncio
    async def test_azure_completion_api_call(
        self,
        mock_completion: AsyncMock,
        mock_azure_openai_settings: AzureOpenAISettings,
    ) -> None:
        deployment_name = "test_deployment"
        prompt = "hello world"
        complete_request_settings = CompleteRequestSettings()
        azure_text_completion = AzureTextCompletion(
            deployment=deployment_name,
            settings=mock_azure_openai_settings,
        )

        await azure_text_completion.complete_async(prompt, complete_request_settings)

        mock_completion.assert_called_once_with(
            engine=deployment_name,
            api_key=mock_azure_openai_settings.api_key.get_secret_value(),
            api_type=mock_azure_openai_settings.api_type.value,
            api_base=mock_azure_openai_settings.endpoint,
            api_version=mock_azure_openai_settings.api_version,
            best_of=1,
            echo=False,
            logit_bias=None,
            logprobs=None,
            suffix=None,
            user=None,
            organization=None,
            prompt=prompt,
            temperature=complete_request_settings.temperature,
            max_tokens=complete_request_settings.max_tokens,
            top_p=complete_request_settings.top_p,
            presence_penalty=complete_request_settings.presence_penalty,
            frequency_penalty=complete_request_settings.frequency_penalty,
            stop=None,
            n=complete_request_settings.number_of_responses,
            stream=False,
        )

    @pytest.fixture()
    def mock_chat_completion(self, mocker: MockerFixture) -> t.Iterator[AsyncMock]:
        mock_chat_completion = AsyncMock(
            return_value=_DotDict({"choices": [{"message": {"content": "test"}}]}),
        )
        mocker.patch(
            "semantic_kernel.utils.openai_.openai.ChatCompletion.acreate",
            mock_chat_completion,
        )
        yield mock_chat_completion

    @pytest.mark.asyncio
    async def test_azure_chat_completion_api_call(
        self,
        mock_chat_completion: AsyncMock,
        mock_azure_openai_settings: AzureOpenAISettings,
        test_logger: SKLogger,
    ) -> None:
        deployment_name = "test_deployment"
        messages = [{"role": "user", "message": "hello world"}]
        complete_request_settings = ChatRequestSettings()

        azure_chat_completion = AzureChatCompletion(
            deployment=deployment_name,
            settings=mock_azure_openai_settings,
            logger=test_logger,
        )

        await azure_chat_completion.complete_chat_async(
            messages, complete_request_settings  # type: ignore
        )

        mock_chat_completion.assert_called_once_with(
            engine=deployment_name,
            api_key=mock_azure_openai_settings.api_key.get_secret_value(),
            api_type=mock_azure_openai_settings.api_type.value,
            api_base=mock_azure_openai_settings.endpoint,
            api_version=mock_azure_openai_settings.api_version,
            logit_bias=None,
            stop=None,
            user=None,
            organization=None,
            messages=messages,
            temperature=complete_request_settings.temperature,
            max_tokens=complete_request_settings.max_tokens,
            top_p=complete_request_settings.top_p,
            presence_penalty=complete_request_settings.presence_penalty,
            frequency_penalty=complete_request_settings.frequency_penalty,
            n=complete_request_settings.number_of_responses,
            stream=False,
        )
