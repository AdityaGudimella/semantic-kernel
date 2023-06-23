# Copyright (c) Microsoft. All rights reserved.

from logging import Logger
from unittest.mock import AsyncMock, patch

import pydantic as pdt
import pytest

from semantic_kernel.connectors.ai.complete_request_settings import (
    CompleteRequestSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)
from semantic_kernel.logging_ import SKLogger
from semantic_kernel.settings import AzureOpenAISettings


def test_azure_chat_completion_init_with_empty_deployment_name() -> None:
    # deployment_name = "test_deployment"
    endpoint = "https://test-endpoint.com"
    api_key = "test_api_key"
    api_version = "2023-03-15-preview"
    logger = Logger("test_logger")

    with pytest.raises(
        ValueError, match="The deployment name cannot be `None` or empty"
    ):
        AzureChatCompletion(
            deployment_name="",
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            logger=logger,
        )


@pytest.mark.parametrize(
    "endpoint", ["https://test-endpoint.com", "test-endpoint.com/", "", None]
)
@pytest.mark.parametrize("api_key", ["test_api_key", "", None])
def test_settings_validation(endpoint: str, api_key: str) -> None:
    """Ensure that the settings are validated on init."""
    if endpoint == "https://test-endpoint.com" and api_key == "test_api_key":
        pytest.skip("Valid settings")
    with pytest.raises(pdt.ValidationError):
        AzureOpenAISettings(
            api_key=api_key,
            api_version="api_version",
            endpoint=endpoint,
        )


@pytest.mark.parametrize("deployment_name", ["", None])
def test_azure_chat_completion_validation(
    deployment_name: str,
    azure_openai_settings: AzureOpenAISettings,
) -> None:
    """Ensure that the settings are validated on init."""
    with pytest.raises(pdt.ValidationError):
        AzureChatCompletion(
            deployment_name=deployment_name,
            settings=azure_openai_settings,
        )


@pytest.mark.asyncio
async def test_azure_chat_completion_call_with_parameters(
    mock_azure_openai_settings: AzureOpenAISettings,
    test_logger: SKLogger,
) -> None:
    mock_openai = AsyncMock()
    with patch(
        "semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion.openai",
        new=mock_openai,
    ):
        deployment_name = "test_deployment"
        prompt = "hello world"
        messages = [{"role": "user", "content": prompt}]
        complete_request_settings = CompleteRequestSettings()

        azure_chat_completion = AzureChatCompletion(
            deployment=deployment_name,
            settings=mock_azure_openai_settings,
            logger=test_logger,
        )

        await azure_chat_completion.complete_async(prompt, complete_request_settings)

        mock_openai.ChatCompletion.acreate.assert_called_once_with(
            engine=deployment_name,
            api_key=mock_azure_openai_settings.api_key,
            api_type=mock_azure_openai_settings.api_type,
            api_base=mock_azure_openai_settings.endpoint,
            api_version=mock_azure_openai_settings.api_version,
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
