# Copyright (c) Microsoft. All rights reserved.

from typing import Any, List, Union

import openai
import pydantic as pdt

from semantic_kernel.connectors.ai.ai_exception import AIException
from semantic_kernel.connectors.ai.complete_request_settings import (
    CompleteRequestSettings,
)
from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.pydantic_ import SKBaseModel
from semantic_kernel.settings import OpenAISettings


class OpenAITextCompletion(SKBaseModel, TextCompletionClientBase):
    model_id: str = pdt.Field(
        description="OpenAI model name. See: https://platform.openai.com/docs/models"
    )
    settings: OpenAISettings = pdt.Field(
        description="OpenAI settings. See: semantic_kernel.settings.OpenAISettings"
    )
    _logger: SKLogger = pdt.PrivateAttr(default_factory=NullLogger)

    async def complete_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ) -> Union[str, List[str]]:
        # TODO: tracking on token counts/etc.
        response = await self._send_completion_request(prompt, request_settings, False)

        if len(response.choices) == 1:
            return response.choices[0].text
        else:
            return [choice.text for choice in response.choices]

    # TODO: complete w/ multiple...

    async def complete_stream_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ):
        response = await self._send_completion_request(prompt, request_settings, True)

        async for chunk in response:
            if request_settings.number_of_responses > 1:
                for choice in chunk.choices:
                    completions = [""] * request_settings.number_of_responses
                    completions[choice.index] = choice.text
                    yield completions
            else:
                yield chunk.choices[0].text

    async def _send_completion_request(
        self, prompt: str, request_settings: CompleteRequestSettings, stream: bool
    ):
        """
        Completes the given prompt. Returns a single string completion.
        Cannot return multiple completions. Cannot return logprobs.

        Arguments:
            prompt {str} -- The prompt to complete.
            request_settings {CompleteRequestSettings} -- The request settings.

        Returns:
            str -- The completed text.
        """
        if not prompt:
            raise ValueError("The prompt cannot be `None` or empty")
        if request_settings is None:
            raise ValueError("The request settings cannot be `None`")

        if request_settings.max_tokens < 1:
            raise AIException(
                AIException.ErrorCodes.InvalidRequest,
                "The max tokens must be greater than 0, "
                f"but was {request_settings.max_tokens}",
            )

        if request_settings.logprobs != 0:
            raise AIException(
                AIException.ErrorCodes.InvalidRequest,
                "complete_async does not support logprobs, "
                f"but logprobs={request_settings.logprobs} was requested",
            )

        model_args = {}
        if self.settings.api_type in ["azure", "azure_ad"]:
            model_args["engine"] = self.model_id
        else:
            model_args["model"] = self.model_id

        try:
            response: Any = await openai.Completion.acreate(
                **model_args,
                api_key=self.settings.api_key,
                api_type=self.settings.api_type,
                api_base=self.settings.endpoint,
                api_version=self.settings.api_version,
                organization=self.settings.org_id,
                prompt=prompt,
                temperature=request_settings.temperature,
                top_p=request_settings.top_p,
                presence_penalty=request_settings.presence_penalty,
                frequency_penalty=request_settings.frequency_penalty,
                max_tokens=request_settings.max_tokens,
                stream=stream,
                n=request_settings.number_of_responses,
                stop=(
                    request_settings.stop_sequences
                    if request_settings.stop_sequences is not None
                    and len(request_settings.stop_sequences) > 0
                    else None
                ),
            )
        except Exception as ex:
            raise AIException(
                AIException.ErrorCodes.ServiceError,
                "OpenAI service failed to complete the prompt",
                ex,
            )
        return response
