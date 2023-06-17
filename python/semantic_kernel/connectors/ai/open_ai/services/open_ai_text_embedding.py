# Copyright (c) Microsoft. All rights reserved.

from logging import Logger
from typing import Any, List

import openai
import pydantic as pdt
from numpy import array, ndarray

from semantic_kernel.connectors.ai.ai_exception import AIException
from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (
    EmbeddingGeneratorBase,
)
from semantic_kernel.logging_ import NullLogger
from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.settings import OpenAISettings


class OpenAITextEmbedding(SKBaseModel, EmbeddingGeneratorBase):
    model_id: str = pdt.Field(
        description="OpenAI model name. See: https://platform.openai.com/docs/models"
    )
    settings: OpenAISettings = pdt.Field(
        description="OpenAI settings. See: semantic_kernel.settings.OpenAISettings"
    )
    _logger: Logger = pdt.PrivateAttr(default_factory=NullLogger)

    async def generate_embeddings_async(self, texts: List[str]) -> ndarray:
        model_args = {}
        if self.settings.api_type in ["azure", "azure_ad"]:
            model_args["engine"] = self.model_id
        else:
            model_args["model"] = self.model_id

        try:
            response: Any = await openai.Embedding.acreate(
                **model_args,
                api_key=self.settings.api_key,
                api_type=self.settings.api_type,
                api_base=self.settings.endpoint,
                api_version=self.settings.api_version,
                organization=self.settings.org_id,
                input=texts,
            )

            # make numpy arrays from the response
            raw_embeddings = [array(x["embedding"]) for x in response["data"]]
            return array(raw_embeddings)
        except Exception as ex:
            raise AIException(
                AIException.ErrorCodes.ServiceError,
                "OpenAI service failed to generate embeddings",
                ex,
            )
