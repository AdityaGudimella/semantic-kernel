# Copyright (c) Microsoft. All rights reserved.

import typing as t

import numpy as np
import pydantic as pdt

from semantic_kernel.connectors.ai.ai_exception import AIException
from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (
    EmbeddingGeneratorBase,
)
from semantic_kernel.connectors.ai.hugging_face.services.base import HFBaseModel
from semantic_kernel.optional_packages import ensure_installed
from semantic_kernel.optional_packages.sentence_transformers import SentenceTransformer


class HuggingFaceTextEmbedding(HFBaseModel, EmbeddingGeneratorBase):
    _generator: t.Optional[SentenceTransformer] = pdt.PrivateAttr(default=None)

    @property
    def generator(self) -> SentenceTransformer:
        if self._generator is None:
            ensure_installed(
                "sentence_transformers",
                error_message="Please install sentence_transformers to use HuggingFaceTextEmbedding.",  # noqa: E501
            )
            self._generator = SentenceTransformer(
                model_name_or_path=self.model_id, device=self._torch_device
            )
        return self._generator

    async def generate_embeddings_async(self, texts: t.List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.

        Arguments:
            texts {List[str]} -- Texts to generate embeddings for.

        Returns:
            ndarray -- Embeddings for the texts.
        """
        try:
            self._log.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.generator.encode(texts)
            return np.asarray(embeddings)
        except Exception as e:
            raise AIException("Hugging Face embeddings failed", e) from e
