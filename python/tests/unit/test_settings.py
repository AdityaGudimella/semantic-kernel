"""Tests for `semantic_kernel.settings`."""
import asyncio
from logging import getLevelName

import numpy as np
import pytest

from semantic_kernel.connectors.ai.open_ai.services.azure_text_embedding import (
    AzureTextEmbedding,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_text_embedding import (
    OpenAITextEmbedding,
)
from semantic_kernel.settings import KernelSettings, OpenAISettings, load_settings


def test_load_settings() -> None:
    """I should be able to load the settings in the test environment.

    If this test fails, a majority of the other tests will fail as well.
    """
    settings = load_settings()
    assert isinstance(settings, KernelSettings)
    assert settings.openai.api_key, "OpenAI API key not set."
    assert getLevelName(settings.logging.get_logger("test").level) == "DEBUG"


@pytest.mark.asyncio
async def test_openai_settings(kernel_settings: KernelSettings) -> None:
    """I should be able to laod the OpenAI settings in the test environment.

    If this test fails, a majority of the other tests depending on OpenAI API will fail
    as well.
    """
    test_model = OpenAITextEmbedding(
        model_id="text-embedding-ada-002", settings=kernel_settings.openai
    )
    result = await test_model.generate_embeddings_async(["test"])
    assert isinstance(result, np.ndarray)


@pytest.mark.asyncio
async def test_azure_openai_settings(kernel_settings: KernelSettings) -> None:
    """I should be able to laod the OpenAI settings in the test environment.

    If this test fails, a majority of the other tests depending on AzureOpenAI API will
    fail as well.
    """
    azure_openai_settings = kernel_settings.azure_openai
    assert azure_openai_settings is not None
    test_model = AzureTextEmbedding(
        deployment="text-embedding-ada-002", settings=azure_openai_settings
    )
    result = await test_model.generate_embeddings_async(["test"])
    assert isinstance(result, np.ndarray)
