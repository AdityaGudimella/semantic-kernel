# Copyright (c) Microsoft. All rights reserved.

from abc import abstractmethod
from typing import List

from numpy import ndarray

from semantic_kernel.pydantic_ import PydanticField


class EmbeddingGeneratorBase(PydanticField):
    @abstractmethod
    async def generate_embeddings_async(self, texts: List[str]) -> ndarray:
        pass
