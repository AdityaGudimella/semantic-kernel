# Copyright (c) Microsoft. All rights reserved.

import typing as t
from threading import Thread

import pydantic as pdt

from semantic_kernel.connectors.ai.ai_exception import AIException
from semantic_kernel.connectors.ai.complete_request_settings import (
    CompleteRequestSettings,
)
from semantic_kernel.connectors.ai.hugging_face.services.base import HFBaseModel
from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.optional_packages import ensure_installed
from semantic_kernel.optional_packages.transformers import Pipeline, transformers


class HuggingFaceTextCompletion(HFBaseModel, TextCompletionClientBase):
    task: str = pdt.Field(
        default="text2text-generation",
        description=(
            "Model completion task type, options are: "
            + "- summarization: takes a long text and returns a shorter summary. "
            + "- text-generation: takes incomplete text and returns a set of completion candidates. "
            + "- text2text-generation (default): takes an input prompt and returns a completion. "
            + "text2text-generation is the default as it behaves more like GPT-3+."
        ),
    )
    _generator: t.Optional[Pipeline] = pdt.PrivateAttr(default=None)

    @property
    def generator(self) -> Pipeline:
        if self._generator is None:
            ensure_installed(
                "transformers",
                error_message="Please install transformers to use HuggingFaceTextCompletion.",  # noqa: E501
            )
            self._generator = transformers.pipeline(
                task=self.task,
                model=self.model_id,
                device=self._torch_device,
                framework="pt",
            )
        return self._generator

    async def complete_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ) -> t.Union[str, t.List[str]]:
        try:
            generation_config = transformers.GenerationConfig(
                temperature=request_settings.temperature,
                top_p=request_settings.top_p,
                max_new_tokens=request_settings.max_tokens,
                pad_token_id=50256,  # EOS token
            )

            results = self.generator(
                prompt,
                do_sample=True,
                num_return_sequences=request_settings.number_of_responses,
                generation_config=generation_config,
            )

            completions = []
            if self._task in ("text-generation", "text2text-generation"):
                completions.extend(response["generated_text"] for response in results)
                return completions[0] if len(completions) == 1 else completions
            elif self._task == "summarization":
                completions.extend(response["summary_text"] for response in results)
                return completions[0] if len(completions) == 1 else completions
            else:
                raise AIException(
                    AIException.ErrorCodes.InvalidConfiguration,
                    "Unsupported hugging face pipeline task: only \
                        text-generation, text2text-generation, and summarization are supported.",
                )

        except Exception as e:
            raise AIException("Hugging Face completion failed", e) from e

    async def complete_stream_async(
        self, prompt: str, request_settings: CompleteRequestSettings
    ):
        """
        Streams a text completion using a Hugging Face model.
        Note that this method does not support multiple responses.

        Arguments:
            prompt {str} -- Prompt to complete.
            request_settings {CompleteRequestSettings} -- Request settings.

        Yields:
            str -- Completion result.
        """
        if request_settings.number_of_responses > 1:
            raise AIException(
                AIException.ErrorCodes.InvalidConfiguration,
                "HuggingFace TextIteratorStreamer does not stream multiple responses in a parseable format. \
                    If you need multiple responses, please use the complete_async method.",
            )
        try:
            generation_config = transformers.GenerationConfig(
                temperature=request_settings.temperature,
                top_p=request_settings.top_p,
                max_new_tokens=request_settings.max_tokens,
                pad_token_id=50256,  # EOS token
            )

            tokenizer = transformers.AutoTokenizer.from_pretrained(self._model_id)
            streamer = transformers.TextIteratorStreamer(tokenizer)
            args = {prompt}
            kwargs = {
                "num_return_sequences": request_settings.number_of_responses,
                "generation_config": generation_config,
                "streamer": streamer,
                "do_sample": True,
            }

            # See https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py#L159
            thread = Thread(target=self.generator, args=args, kwargs=kwargs)
            thread.start()

            for new_text in streamer:
                yield new_text

            thread.join()

        except Exception as e:
            raise AIException("Hugging Face completion failed", e) from e
