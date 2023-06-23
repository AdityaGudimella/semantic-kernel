# Copyright (c) Microsoft. All rights reserved.

import typing as t
from logging import Logger

import numpy as np
import pydantic as pdt

from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (
    EmbeddingGeneratorBase,
)
from semantic_kernel.logging_ import NullLogger
from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.settings import OpenAISettings
from semantic_kernel.utils.openai_ import OpenAIBackends, generate_embeddings_async


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
            settings=self.settings.openai_api_kwargs,
            model_or_engine=self.model_id,
            backend=OpenAIBackends.Azure,
        )
