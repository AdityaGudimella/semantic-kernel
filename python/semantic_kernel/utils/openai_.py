"""Utility functions to connect to OpenAI API."""
import enum
import typing as t

import numpy as np
import openai


class OpenAIBackends(str, enum.Enum):
    """Backends supported by OpenAI."""

    Azure = "azure"
    OpenAI = "openai"


class OpenAIAPIKwargs(t.TypedDict):
    """OpenAI settings."""

    api_key: str
    api_type: t.Optional[str]
    api_base: t.Optional[str]
    api_version: t.Optional[str]
    organization: t.Optional[str]


async def generate_embeddings_async(
    input: t.List[str],
    settings: OpenAIAPIKwargs,
    model_or_engine: str,
    backend: OpenAIBackends = OpenAIBackends.OpenAI,
) -> np.ndarray:
    """Wrapper around OpenAI Embedding.acreate()."""
    model_key = "model" if backend == OpenAIBackends.OpenAI else "engine"
    kwargs = {
        model_key: model_or_engine,
        **settings,
        "input": input,
    }

    try:
        response = await openai.Embedding.acreate(**kwargs)

        if not isinstance(response, dict):
            raise ConnectionError("OpenAI service failed to generate embeddings")
        if "data" not in response:
            raise ConnectionError("OpenAI service failed to generate embeddings")
    except Exception as ex:
        raise ConnectionError("OpenAI service failed to generate embeddings") from ex
    # Convert list[list[float]] to np.ndarray
    return np.stack([np.asarray(x["embedding"]) for x in response["data"]])
