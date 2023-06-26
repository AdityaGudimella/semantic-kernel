# Copyright (c) Microsoft. All rights reserved.

import typing as t

import pytest

import semantic_kernel as sk
import semantic_kernel.connectors.ai.openai as sk_oai
from semantic_kernel.connectors.ai.openai import openai_
from semantic_kernel.logging_ import LoggerSettings, SKLogger
from semantic_kernel.settings import (
    AzureOpenAISettings,
    KernelSettings,
    OpenAISettings,
    load_settings,
)


@pytest.fixture()
def kernel_settings() -> KernelSettings:
    """Returns the default kernel settings used for testing."""
    return load_settings()


@pytest.fixture()
def test_logger_settings(kernel_settings: KernelSettings) -> LoggerSettings:
    """Returns the default logger settings used in tests."""
    return kernel_settings.logging


@pytest.fixture()
def openai_settings(kernel_settings: KernelSettings) -> OpenAISettings:
    """Returns the default OpenAI settings."""
    return kernel_settings.openai


@pytest.fixture()
def azure_openai_settings(kernel_settings: KernelSettings) -> AzureOpenAISettings:
    """Returns the default Azure OpenAI config."""
    result = kernel_settings.azure_openai
    assert result
    return result


@pytest.fixture()
def mock_azure_openai_settings() -> AzureOpenAISettings:
    """Returns a mock Azure OpenAI config that will not work with actual API."""
    return AzureOpenAISettings(
        api_key="test_api_key",  # pyright: ignore[reportGeneralTypeIssues]
        endpoint="https://test-endpoint.com",
        api_version="2023-03-15-preview",
    )


@pytest.fixture()
def test_logger(test_logger_settings: LoggerSettings) -> SKLogger:
    """Returns a logger to be used in testing.

    Configure your test logging settings here.
    """
    return SKLogger(name="test_logger", settings=test_logger_settings)


@pytest.fixture()
def service_type() -> t.Optional[str]:
    """Returns the default service type."""
    return None


@pytest.fixture(scope="session")
def create_kernel():
    return sk.Kernel()


KernelServiceType = t.Optional[t.Literal["chat", "text_completion"]]


@pytest.fixture()
def kernel(
    service_type: KernelServiceType, openai_settings: OpenAISettings
) -> sk.Kernel:
    """Returns the default kernel instance."""
    kernel = sk.Kernel()
    if not service_type:
        return kernel
    if service_type == "chat":
        kernel.add_chat_service(
            service_type,
            sk_oai.OpenAIChatCompletion(
                model_id="gpt-3.5-turbo", settings=openai_settings
            ),
        )
    else:
        kernel.add_text_completion_service(
            service_type,
            openai_.OpenAITextCompletion(
                model_id="text-davinci-003", settings=openai_settings
            ),
        )
    return kernel


@pytest.fixture()
def prompt_config() -> sk.PromptTemplateConfig:
    """Returns the default prompt config."""
    return sk.PromptTemplateConfig.from_completion_parameters(
        max_tokens=2000, temperature=0.7, top_p=0.8
    )


@pytest.fixture()
def prompt_template(
    kernel: sk.Kernel,
    prompt_config: sk.PromptTemplateConfig,
    service_type: KernelServiceType,
) -> sk.PromptTemplate:
    if service_type == "chat":
        prompt_template = sk.ChatPromptTemplate(
            template="{{$user_input}}",
            template_engine=kernel.prompt_template_engine,
            prompt_config=prompt_config,
        )

        system_message = """
        You are a chat bot. Your name is Mosscap and
        you have one goal: figure out what people need.
        Your full name, should you need to know it, is
        Splendid Speckled Mosscap. You communicate
        effectively, but you tend to answer with long
        flowery prose.
        """
        prompt_template.add_system_message(system_message)
        prompt_template.add_user_message("Hi there, who are you?")
        prompt_template.add_assistant_message(
            "I am Mosscap, a chat bot. I'm trying to figure out what people need."
        )
        return prompt_template
    elif service_type == "text_completion":
        sk_prompt = """
        ChatBot can have a conversation with you about any topic.
        It can give explicit instructions or say 'I don't know'
        when it doesn't know the answer.

        {{$chat_history}}
        User:> {{$user_input}}
        ChatBot:>
        """
        return sk.PromptTemplate(
            template=sk_prompt,
            template_engine=kernel.prompt_template_engine,
            prompt_config=prompt_config,
        )
    else:
        raise ValueError(f"Invalid service type: {service_type}")


@pytest.fixture()
def chat_function(
    kernel: sk.Kernel,
    prompt_config: sk.PromptTemplateConfig,
    prompt_template: sk.PromptTemplate,
) -> sk.SKFunctionBase:
    function_config = sk.SemanticFunctionConfig(
        prompt_template_config=prompt_config, prompt_template=prompt_template
    )
    return kernel.register_semantic_function("ChatBot", "Chat", function_config)


@pytest.fixture()
def chat_kernel(kernel: sk.Kernel, openai_settings: OpenAISettings) -> sk.Kernel:
    """Returns a kernel with a chat function registered."""
    kernel.add_chat_service(
        "chat-gpt",
        sk_oai.OpenAIChatCompletion(model_id="gpt-3.5-turbo", settings=openai_settings),
    )
    return kernel
