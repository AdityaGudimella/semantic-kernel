# Copyright (c) Microsoft. All rights reserved.

import os
import typing as t

import pytest

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.settings import KernelSettings, load_settings


@pytest.fixture(scope="session")
def create_kernel():
    return sk.Kernel()


@pytest.fixture(scope="session")
def get_aoai_config():
    if "Python_Integration_Tests" in os.environ:
        deployment_name = os.environ["AzureOpenAIEmbeddings__DeploymentName"]
        api_key = os.environ["AzureOpenAI__ApiKey"]
        endpoint = os.environ["AzureOpenAI__Endpoint"]
    else:
        # Load credentials from .env file
        deployment_name, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
        deployment_name = "text-embedding-ada-002"

    return deployment_name, api_key, endpoint


@pytest.fixture(scope="session")
def get_oai_config():
    if "Python_Integration_Tests" in os.environ:
        api_key = os.environ["OpenAI__ApiKey"]
        org_id = None
    else:
        # Load credentials from .env file
        api_key, org_id = sk.openai_settings_from_dot_env()

    return api_key, org_id


@pytest.fixture()
def oai_config():
    """Returns the default OpenAI config."""


@pytest.fixture()
def azure_oai_config():
    """Returns the default Azure OpenAI config."""


@pytest.fixture()
def service_type() -> t.Optional[str]:
    """Returns the default service type."""
    return None


KernelServiceType = t.Optional[t.Literal["chat", "text_completion"]]


@pytest.fixture()
def kernel_settings() -> KernelSettings:
    """Returns the default kernel settings used for testing."""
    return load_settings()


@pytest.fixture()
def kernel(service_type: KernelServiceType) -> sk.Kernel:
    """Returns the default kernel instance."""
    kernel = sk.Kernel()
    if not service_type:
        return kernel
    api_key, org_id = sk.openai_settings_from_dot_env()
    if service_type == "chat":
        kernel.add_chat_service(
            service_type, sk_oai.OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)
        )
    else:
        kernel.add_text_completion_service(
            service_type,
            sk_oai.OpenAITextCompletion("text-davinci-003", api_key, org_id),
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
            "{{$user_input}}", kernel.prompt_template_engine, prompt_config
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
            sk_prompt, kernel.prompt_template_engine, prompt_config
        )
    else:
        raise ValueError(f"Invalid service type: {service_type}")


@pytest.fixture()
def chat_function(
    kernel: sk.Kernel,
    prompt_config: sk.PromptTemplateConfig,
    prompt_template: sk.PromptTemplate,
) -> sk.SKFunctionBase:
    function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
    return kernel.register_semantic_function("ChatBot", "Chat", function_config)


@pytest.fixture()
def chat_kernel(kernel: sk.Kernel) -> sk.Kernel:
    """Returns a kernel with a chat function registered."""
    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_chat_service(
        "chat-gpt", sk_oai.OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)
    )
    return kernel
