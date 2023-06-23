# Copyright (c) Microsoft. All rights reserved.
import typing as t

import numpy as np
import pydantic as pdt

from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (
    EmbeddingGeneratorBase,
)
from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.settings import AzureOpenAISettings
from semantic_kernel.utils.openai_ import OpenAIBackends, generate_embeddings_async


class AzureTextEmbedding(SKBaseModel, EmbeddingGeneratorBase):
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
    _logger: SKLogger = pdt.PrivateAttr(default_factory=NullLogger)

    async def generate_embeddings_async(self, texts: t.List[str]) -> np.ndarray:
        return await generate_embeddings_async(
            input=texts,
            settings=self.settings.openai_api_kwargs,
            model_or_engine=self.deployment,
            backend=OpenAIBackends.Azure,
        )
