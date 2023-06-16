# Copyright (c) Microsoft. All rights reserved.

from logging import Logger
from typing import Any, List, Tuple, Union

import openai
import pydantic as pdt

from semantic_kernel.connectors.ai.ai_exception import AIException
from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.chat_request_settings import ChatRequestSettings
from semantic_kernel.connectors.ai.complete_request_settings import (
    CompleteRequestSettings,
)
from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.logging_ import NullLogger
from semantic_kernel.settings import OpenAISettings


class OpenAIChatCompletion(
    pdt.BaseModel, ChatCompletionClientBase, TextCompletionClientBase
):
    model_id: str = pdt.Field(
        description="OpenAI model name. See: https://platform.openai.com/docs/models"
    )
    settings: OpenAISettings = pdt.Field(
        description="OpenAI settings. See: semantic_kernel.settings.OpenAISettings"
    )
    _logger: Logger = pdt.PrivateAttr(NullLogger())
    _messages: List[Tuple[str, str]] = pdt.PrivateAttr([])

    async def complete_chat_async(
        self, messages: List[Tuple[str, str]], request_settings: ChatRequestSettings
    ) -> Union[str, List[str]]:
        # TODO: tracking on token counts/etc.
        response = await self._send_chat_request(messages, request_settings, False)

        if len(response.choices) == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]

    async def complete_chat_stream_async(
        self, messages: List[Tuple[str, str]], request_settings: ChatRequestSettings
    ):
        response = await self._send_chat_request(messages, request_settings, True)

        # parse the completion text(s) and yield them
        async for chunk in response:
            text, index = _parse_choices(chunk)
            # if multiple responses are requested, keep track of them
            if request_settings.number_of_responses > 1:
                completions = [""] * request_settings.number_of_responses
                completions[index] = text
                yield completions
            # if only one response is requested, yield it
            else:
                yield text

    async def complete_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ) -> Union[str, List[str]]:
        """
        Completes the given prompt.

        Arguments:
            prompt {str} -- The prompt to complete.
            request_settings {CompleteRequestSettings} -- The request settings.

        Returns:
            str -- The completed text.
        """
        prompt_to_message = [("user", prompt)]
        chat_settings = ChatRequestSettings(
            temperature=request_settings.temperature,
            top_p=request_settings.top_p,
            presence_penalty=request_settings.presence_penalty,
            frequency_penalty=request_settings.frequency_penalty,
            max_tokens=request_settings.max_tokens,
            number_of_responses=request_settings.number_of_responses,
        )
        response = await self._send_chat_request(
            prompt_to_message, chat_settings, False
        )

        if len(response.choices) == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]

    async def complete_stream_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ):
        prompt_to_message = [("user", prompt)]
        chat_settings = ChatRequestSettings(
            temperature=request_settings.temperature,
            top_p=request_settings.top_p,
            presence_penalty=request_settings.presence_penalty,
            frequency_penalty=request_settings.frequency_penalty,
            max_tokens=request_settings.max_tokens,
            number_of_responses=request_settings.number_of_responses,
        )
        response = await self._send_chat_request(prompt_to_message, chat_settings, True)

        # parse the completion text(s) and yield them
        async for chunk in response:
            text, index = _parse_choices(chunk)
            # if multiple responses are requested, keep track of them
            if request_settings.number_of_responses > 1:
                completions = [""] * request_settings.number_of_responses
                completions[index] = text
                yield completions
            # if only one response is requested, yield it
            else:
                yield text

    async def _send_chat_request(
        self,
        messages: List[Tuple[str, str]],
        request_settings: ChatRequestSettings,
        stream: bool,
    ):
        """
        Completes the given user message with an asynchronous stream.

        Arguments:
            user_message {str} -- The message (from a user) to respond to.
            request_settings {ChatRequestSettings} -- The request settings.

        Returns:
            str -- The completed text.
        """
        if request_settings is None:
            raise ValueError("The request settings cannot be `None`")

        if request_settings.max_tokens < 1:
            raise AIException(
                AIException.ErrorCodes.InvalidRequest,
                "The max tokens must be greater than 0, "
                f"but was {request_settings.max_tokens}",
            )

        if len(messages) <= 0:
            raise AIException(
                AIException.ErrorCodes.InvalidRequest,
                "To complete a chat you need at least one message",
            )

        if messages[-1][0] != "user":
            raise AIException(
                AIException.ErrorCodes.InvalidRequest,
                "The last message must be from the user",
            )

        model_args = {}
        if self.settings.api_type in ["azure", "azure_ad"]:
            model_args["engine"] = self.model_id
        else:
            model_args["model"] = self.model_id

        formatted_messages = [
            {"role": role, "content": message} for role, message in messages
        ]

        try:
            response: Any = await openai.ChatCompletion.acreate(
                **model_args,
                api_key=self.settings.api_key,
                api_type=self.settings.api_type,
                api_base=self.settings.endpoint,
                api_version=self.settings.api_version,
                organization=self.settings.org_id,
                messages=formatted_messages,
                temperature=request_settings.temperature,
                top_p=request_settings.top_p,
                presence_penalty=request_settings.presence_penalty,
                frequency_penalty=request_settings.frequency_penalty,
                max_tokens=request_settings.max_tokens,
                n=request_settings.number_of_responses,
                stream=stream,
            )
        except Exception as ex:
            raise AIException(
                AIException.ErrorCodes.ServiceError,
                "OpenAI service failed to complete the chat",
                ex,
            ) from ex

        # TODO: tracking on token counts/etc.

        return response


def _parse_choices(chunk):
    message = ""
    if "role" in chunk.choices[0].delta:
        message += chunk.choices[0].delta.role + ": "
    if "content" in chunk.choices[0].delta:
        message += chunk.choices[0].delta.content

    index = chunk.choices[0].index
    return message, index
