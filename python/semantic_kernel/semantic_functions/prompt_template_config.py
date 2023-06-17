# Copyright (c) Microsoft. All rights reserved.

import typing as t

import pydantic as pdt


class CompletionConfig(pdt.BaseModel):
    temperature: float = pdt.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        #  Higher values result in more diverse text, while lower values result in more
        # predictable text
        description="Controls the 'creativity' of the generated text.",
    )
    top_p: float = pdt.Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        # Higher values result in higher quality text, while lower values result in
        # lower quality text.
        description="Controls the 'quality' of the generated text.",
    )
    presence_penalty: float = pdt.Field(
        default=0.0,
        ge=0.0,
        description=(
            "Controls the degree to which the model avoids repeating the same text as the input prompt."  # noqa: E501
        ),
    )
    frequency_penalty: float = pdt.Field(
        default=0.0,
        ge=0.0,
        description=(
            "Controls the degree to which the model avoids using the same words as the input prompt."  # noqa: E501
        ),
    )
    max_tokens: int = pdt.Field(
        default=256,
        ge=1,
        description=(
            "Controls the maximum number of tokens (words or subwords) in the generated text."  # noqa: E501
        ),
    )
    number_of_responses: int = pdt.Field(
        default=1,
        ge=1,
        description="Controls the number of responses to generate for each prompt.",
    )
    stop_sequences: t.List[str] = pdt.Field(
        default_factory=list,
        description=(
            "When encountered in the generated text, will cause the generation to stop."
        ),
    )

    def update(self, **kwargs: t.Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class InputParameter(pdt.BaseModel):
    name: str = pdt.Field(..., description="The name of the input parameter.")
    description: str = pdt.Field(
        ..., description="A description of the input parameter."
    )
    default_value: str = pdt.Field(
        ..., description="The default value of the input parameter."
    )


class InputConfig(pdt.BaseModel):
    parameters: t.List[InputParameter] = pdt.Field(
        default_factory=list,
        description="A list of input parameters for the prompt template.",
    )

    @pdt.validator("parameters")
    def validate_parameters(cls, v: t.Any) -> t.List[InputParameter]:
        if not isinstance(v, list):
            raise TypeError("parameters must be a list")
        for parameter in v:
            if not isinstance(parameter, InputParameter):
                raise TypeError("parameters must be a list of InputParameter")
        return v


class PromptTemplateConfig(pdt.BaseModel):
    schema: int = 1
    type: str = "completion"
    description: str = ""
    completion: CompletionConfig = pdt.Field(default_factory=CompletionConfig)
    default_services: t.List[str] = pdt.Field(default_factory=list)
    input: t.Optional[InputConfig] = pdt.Field(default=None)

    @classmethod
    def from_completion_parameters(
        cls,
        temperature: float = 0.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        max_tokens: int = 256,
        number_of_responses: int = 1,
        stop_sequences: t.Optional[t.List[str]] = None,
    ) -> "PromptTemplateConfig":
        if stop_sequences is None:
            stop_sequences = []
        return cls(
            completion=CompletionConfig(
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                number_of_responses=number_of_responses,
                stop_sequences=stop_sequences,
            )
        )
