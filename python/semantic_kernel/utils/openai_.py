"""Utility functions to connect to OpenAI API."""
import enum
import typing as t

import numpy as np
import openai
import pydantic as pdt
import typing_extensions as te


class OpenAIBackends(str, enum.Enum):
    """Backends supported by OpenAI."""

    Azure = "azure"
    OpenAI = "openai"


class OpenAIAPIKwargs(pdt.BaseModel):
    """OpenAI settings."""

    api_key: str
    api_type: t.Optional[str] = None
    api_base: t.Optional[str] = None
    api_version: t.Optional[str] = None
    organization: t.Optional[str] = None

    class Config:
        """Pydantic config."""

        allow_mutation = False
        extra = "forbid"
        orm_mode = True


async def generate_embeddings_async(
    input: t.List[str],
    api_kwargs: OpenAIAPIKwargs,
    model_or_engine: str,
    backend: OpenAIBackends = OpenAIBackends.OpenAI,
) -> np.ndarray:
    """Wrapper around OpenAI Embedding.acreate()."""
    model_key = "model" if backend == OpenAIBackends.OpenAI else "engine"
    kwargs = {
        model_key: model_or_engine,
        **api_kwargs.dict(),
        "input": input,
    }

    try:
        response = t.cast(dict, await openai.Embedding.acreate(**kwargs))
    except Exception as ex:
        raise ConnectionError("OpenAI service failed to generate embeddings") from ex
    if "data" not in response:
        raise ConnectionError(
            f"OpenAI service returned unexpected response: {response}"
        )
    # Convert list[list[float]] to np.ndarray
    return np.stack([np.asarray(x["embedding"]) for x in response["data"]])


# A sequence of tokens for the AI model.
_Tokens: te.TypeAlias = t.List[float]
_WordsOrTokens: te.TypeAlias = t.Union[str, t.List[str], _Tokens, t.List[_Tokens]]


class ChatCompletionKwargs(pdt.BaseModel):
    """Settings for OpenAI's ChatCompletion.create().

    See: https://platform.openai.com/docs/api-reference/chat/create
    """

    max_tokens: int = pdt.Field(
        default=16,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    temperature: t.Optional[float] = pdt.Field(
        default=None,
        description="What sampling temperature to use.",
        ge=0,
        le=2,
    )
    top_p: float = pdt.Field(
        default=1.0,
        description="Alternative to sampling with temperature, called nucleus sampling.",  # noqa: E501
        ge=0,
        le=1,
    )
    n: int = pdt.Field(
        default=1,
        description="How many completions to generate for each prompt.",
        gt=0,
    )
    stop: t.Optional[_WordsOrTokens] = pdt.Field(
        default=None,
        description="Up to 4 sequences where the API will stop generating further tokens.",  # noqa: E501
        max_items=4,
    )
    presence_penalty: float = pdt.Field(
        default=0.0,
        description=(
            "What penalty to apply if a token is already present at the end of the prompt."  # noqa: E501
        ),
        le=2,
        ge=-2,
    )
    frequency_penalty: float = pdt.Field(
        default=0.0,
        description=(
            "What penalty to apply if a token has already been generated in the text."
        ),
        le=2,
        ge=-2,
    )
    logit_bias: t.Optional[t.Dict[str, float]] = pdt.Field(
        default=None,
        description=(
            "Logit bias to insert for each token (somehow similar to temperature)."
            + " Used to modify the likelihood of specified tokens appearing in the"
            + " completion. The keys are tokens and the values are the bias values."
        ),
    )
    user: t.Optional[str] = pdt.Field(
        default=None,
        description="The user ID to impersonate. Used to detect and monitor abuse.",
    )

    class Config:
        """Pydantic config."""

        allow_mutation = False
        extra = "forbid"
        orm_mode = True


class CompletionKwargs(ChatCompletionKwargs):
    """Settings for OpenAI's Completion.create().

    See: https://platform.openai.com/docs/api-reference/completions/create
    """

    suffix: t.Optional[str] = pdt.Field(
        default=None,
        description="A suffix that comes after a completion of inserted text.",
    )
    logprobs: t.Optional[int] = pdt.Field(
        default=None,
        description=(
            "Include the log probabilities on the logprobs most likely tokens,"
            + " as well the chosen tokens."
        ),
        ge=None,
    )
    echo: bool = pdt.Field(
        default=False,
        description="Echo back the prompt in addition to the completion.",
    )
    best_of: int = pdt.Field(
        default=1,
        description="Generates best_of completions server-side and returns the 'best'.",
        gt=0,
    )


# TODO(ADI): Figure out what the types.
def _ensure_valid_response(response: t.Any) -> t.Any:
    if "choices" not in response:
        raise ConnectionError("OpenAI service failed to generate completion")
    return response


async def _complete_async(
    prompt: str,
    api_kwargs: OpenAIAPIKwargs,
    model_or_engine: str,
    backend: OpenAIBackends = OpenAIBackends.OpenAI,
    request_kwargs: t.Optional[CompletionKwargs] = None,
    stream: bool = False,
    # TODO(ADI): Figure out what the return type is.
) -> t.Any:
    """Wrapper around OpenAI Completion.acreate().

    Args:
        prompt: The prompt to generate completions for.
        api_kwargs: OpenAI API settings.
        model_or_engine: The model or engine to use.
            If backend is OpenAIBackends.OpenAI, this is the model ID.
            If backend is OpenAIBackends.Azure, this is the engine ID.
        backend: The backend to use.
        request_kwargs: Settings for OpenAI's `Completion.create()`.
    """
    request_kwargs = request_kwargs or CompletionKwargs()
    model_key = "model" if backend == OpenAIBackends.OpenAI else "engine"
    kwargs = {
        model_key: model_or_engine,
        **api_kwargs.dict(),
        "prompt": prompt,
        **request_kwargs.dict(),
        "stream": stream,
    }

    try:
        return _ensure_valid_response(await openai.Completion.acreate(**kwargs))
    except Exception as ex:
        raise ConnectionError("OpenAI service failed to generate completion") from ex


async def complete_async(
    prompt: str,
    api_kwargs: OpenAIAPIKwargs,
    model_or_engine: str,
    backend: OpenAIBackends = OpenAIBackends.OpenAI,
    request_kwargs: t.Optional[CompletionKwargs] = None,
) -> t.Union[str, t.List[str]]:
    """Wrapper around OpenAI Completion.acreate() for non-streaming completion."""
    response = await _complete_async(
        prompt=prompt,
        api_kwargs=api_kwargs,
        model_or_engine=model_or_engine,
        backend=backend,
        request_kwargs=request_kwargs,
    )
    if not response["choices"]:
        raise ConnectionError("OpenAI service failed to generate completion")
    choices = [choice.text for choice in response.choices]
    return choices[0] if len(choices) == 1 else choices


async def complete_stream_async(
    prompt: str,
    api_kwargs: OpenAIAPIKwargs,
    model_or_engine: str,
    backend: OpenAIBackends = OpenAIBackends.OpenAI,
    request_kwargs: t.Optional[CompletionKwargs] = None,
    # TODO(ADI): Figure out what the return type is.
) -> t.Any:
    """Wrapper around OpenAI Completion.acreate() for streaming completion."""
    request_kwargs = request_kwargs or CompletionKwargs()
    response = await _complete_async(
        prompt=prompt,
        api_kwargs=api_kwargs,
        model_or_engine=model_or_engine,
        backend=backend,
        request_kwargs=request_kwargs,
    )
    async for chunk in response:
        if request_kwargs.n == 1:
            yield chunk.choices[0].text
        else:
            for choice in chunk.choices:
                completions = [""] * request_kwargs.n
                completions[choice.index] = choice.text
                yield completions


class RoleMessage(t.TypedDict):
    role: str
    message: str


async def _complete_chat_async(
    messages: t.List[RoleMessage],
    api_kwargs: OpenAIAPIKwargs,
    model_or_engine: str,
    backend: OpenAIBackends = OpenAIBackends.OpenAI,
    request_kwargs: t.Optional[ChatCompletionKwargs] = None,
    stream: bool = False,
    # TODO(ADI): Figure out what the return type is.
) -> t.Any:
    """Wrapper around OpenAI ChatCompletion.acreate()."""
    if not messages:
        raise ValueError("messages cannot be empty")
    request_kwargs = request_kwargs or ChatCompletionKwargs()
    model_key = "model" if backend == OpenAIBackends.OpenAI else "engine"
    kwargs = {
        model_key: model_or_engine,
        **api_kwargs.dict(),
        "messages": messages,
        **request_kwargs.dict(),
        "stream": stream,
    }
    try:
        return _ensure_valid_response(await openai.ChatCompletion.acreate(**kwargs))
    except Exception as ex:
        raise ConnectionError("OpenAI service failed to generate completion") from ex


async def complete_chat_async(
    messages: t.List[RoleMessage],
    api_kwargs: OpenAIAPIKwargs,
    model_or_engine: str,
    backend: OpenAIBackends = OpenAIBackends.OpenAI,
    request_kwargs: t.Optional[ChatCompletionKwargs] = None,
    # TODO(ADI): Figure out what the return type is.
) -> t.Any:
    """Wrapper around OpenAI ChatCompletion.acreate() for non-streaming completion."""
    response = await _complete_chat_async(
        messages=messages,
        api_kwargs=api_kwargs,
        model_or_engine=model_or_engine,
        backend=backend,
        request_kwargs=request_kwargs,
    )
    if not response["choices"]:
        raise ConnectionError("OpenAI service failed to generate completion")
    if len(response.choices) == 1:
        return response.choices[0].message.content
    return [choice.message.content for choice in response.choices]


async def complete_chat_stream_async(
    messages: t.List[RoleMessage],
    api_kwargs: OpenAIAPIKwargs,
    model_or_engine: str,
    backend: OpenAIBackends = OpenAIBackends.OpenAI,
    request_kwargs: t.Optional[ChatCompletionKwargs] = None,
    # TODO(ADI): Figure out what the return type is.
) -> t.Any:
    """Wrapper around OpenAI ChatCompletion.acreate() for streaming completion."""

    def _parse_choices(chunk):
        message = ""
        if "role" in chunk.choices[0].delta:
            message += f"{chunk.choices[0].delta.role}: "
        if "content" in chunk.choices[0].delta:
            message += chunk.choices[0].delta.content
        index = chunk.choices[0].index
        return message, index

    request_kwargs = request_kwargs or ChatCompletionKwargs()
    response = await _complete_chat_async(
        messages=messages,
        api_kwargs=api_kwargs,
        model_or_engine=model_or_engine,
        backend=backend,
        request_kwargs=request_kwargs,
        stream=True,
    )
    async for chunk in response:
        text, index = _parse_choices(chunk)
        if request_kwargs.n == 1:
            yield text
        else:
            completions = [""] * request_kwargs.n
            completions[index] = text
            yield completions
