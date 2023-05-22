"""Test serialization of SK Kernel."""

import pytest

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.serialization import from_json, to_json


@pytest.fixture()
def kernel() -> sk.Kernel:
    """Return a `Kernel`."""
    sk_prompt = """
    ChatBot can have a conversation with you about any topic.
    It can give explicit instructions or say 'I don't know'
    when it doesn't know the answer.

    {{$chat_history}}
    User:> {{$user_input}}
    ChatBot:>
    """

    kernel = sk.Kernel()

    api_key, org_id = sk.openai_settings_from_dot_env()
    kernel.add_text_completion_service(
        "davinci-003", sk_oai.OpenAITextCompletion("text-davinci-003", api_key, org_id)
    )

    prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
        max_tokens=2000, temperature=0.7, top_p=0.4
    )

    prompt_template = sk.PromptTemplate(
        sk_prompt, kernel.prompt_template_engine, prompt_config
    )

    function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
    kernel.register_semantic_function("ChatBot", "Chat", function_config)

    system_message = """
    You are a chat bot. Your name is Mosscap and
    you have one goal: figure out what people need.
    Your full name, should you need to know it, is
    Splendid Speckled Mosscap. You communicate
    effectively, but you tend to answer with long
    flowery prose.
    """

    kernel.add_chat_service(
        "chat-gpt", sk_oai.OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id)
    )

    prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
        max_tokens=2000, temperature=0.7, top_p=0.8
    )

    prompt_template = sk.ChatPromptTemplate(
        "{{$user_input}}", kernel.prompt_template_engine, prompt_config
    )

    prompt_template.add_system_message(system_message)
    prompt_template.add_user_message("Hi there, who are you?")
    prompt_template.add_assistant_message(
        "I am Mosscap, a chat bot. I'm trying to figure out what people need."
    )
    function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
    kernel.register_semantic_function("ChatBot", "Chat", function_config)

    return kernel


def test_serialization_and_deserialization(kernel: sk.Kernel) -> None:
    """Test serialization of a `Kernel` to JSON."""
    loaded_kernel = from_json(to_json(kernel))
    assert isinstance(loaded_kernel, type(kernel))
    assert isinstance(
        loaded_kernel.prompt_template_engine, type(kernel.prompt_template_engine)
    )
    assert kernel.all_chat_services() == loaded_kernel.all_chat_services()
    assert (
        kernel.all_text_completion_services()
        == loaded_kernel.all_text_completion_services()
    )
    assert (
        kernel.all_text_completion_services()
        == loaded_kernel.all_text_completion_services()
    )
