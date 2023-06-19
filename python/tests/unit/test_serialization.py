"""Test serialization of SK Kernel."""
import typing as t

import pydantic as pdt
import pytest
import typing_extensions as te

import semantic_kernel as sk
from semantic_kernel.connectors.ai.chat_request_settings import ChatRequestSettings
from semantic_kernel.connectors.ai.complete_request_settings import (
    CompleteRequestSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import (
    OpenAIChatCompletion,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_text_completion import (
    OpenAITextCompletion,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_text_embedding import (
    OpenAITextEmbedding,
)
from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.kernel import Kernel
from semantic_kernel.memory.semantic_text_memory_base import SemanticTextMemoryBase
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from semantic_kernel.pydantic_ import PydanticField
from semantic_kernel.reliability.pass_through_without_retry import (
    PassThroughWithoutRetry,
)
from semantic_kernel.reliability.retry_mechanism_base import RetryMechanismBase
from semantic_kernel.semantic_functions.prompt_template import PromptTemplate
from semantic_kernel.semantic_functions.prompt_template_config import (
    PromptTemplateConfig,
)
from semantic_kernel.serialization import from_json, to_json
from semantic_kernel.settings import KernelSettings
from semantic_kernel.skill_definition.skill_collection import SkillCollection
from semantic_kernel.skill_definition.skill_collection_base import SkillCollectionBase
from semantic_kernel.template_engine.blocks.block import Block
from semantic_kernel.template_engine.blocks.code_block import CodeBlock
from semantic_kernel.template_engine.code_tokenizer import CodeTokenizer
from semantic_kernel.template_engine.prompt_template_engine import PromptTemplateEngine
from semantic_kernel.template_engine.protocols.code_renderer import CodeRenderer
from semantic_kernel.template_engine.protocols.prompt_templating_engine import (
    PromptTemplatingEngine,
)
from semantic_kernel.template_engine.template_tokenizer import TemplateTokenizer


@pytest.fixture()
def kernel(kernel_settings: KernelSettings) -> sk.Kernel:
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
        "davinci-003",
        OpenAITextCompletion(
            model_id="text-davinci-003", settings=kernel_settings.openai
        ),
    )

    prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
        max_tokens=2000, temperature=0.7, top_p=0.4
    )

    prompt_template = sk.PromptTemplate(
        template=sk_prompt,
        template_engine=kernel.prompt_template_engine,
        prompt_config=prompt_config,
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
        "chat-gpt",
        OpenAIChatCompletion(model_id="gpt-3.5-turbo", settings=kernel_settings.openai),
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


@pytest.mark.skip(reason="Remove or move.")
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


PydanticFieldT = t.TypeVar("PydanticFieldT", bound=PydanticField)


@pytest.mark.parametrize(
    "sk_type",
    [
        ContextVariables,
        SKFunctionBase,
        RetryMechanismBase,
        SkillCollectionBase,
        PromptTemplatingEngine,
        SemanticTextMemoryBase,
        TextCompletionClientBase,
    ],
)
def test_usage_in_pydantic_fields(sk_type: t.Type[PydanticFieldT]) -> None:
    """Semantic Kernel objects should be valid Pydantic fields.

    Otherwise, they cannot be used in Pydantic models.
    """

    class TestModel(pdt.BaseModel):
        """A test model."""

        field: t.Optional[sk_type] = None

    test_model = TestModel()
    assert test_model is not None
    serialized = test_model.json()
    assert isinstance(serialized, str)
    deserialized = TestModel.parse_raw(serialized)
    assert isinstance(deserialized, TestModel)
    assert deserialized == test_model


class _Serializable(t.Protocol):
    """A serializable object."""

    def json(self) -> pdt.Json:
        """Return a JSON representation of the object."""
        raise NotImplementedError

    def parse_raw(self: te.Self, json: pdt.Json) -> te.Self:
        """Return the constructed object from a JSON representation."""
        raise NotImplementedError


@pytest.fixture()
def serializable(
    serializable_type: type[t.Any], kernel_settings: KernelSettings
) -> _Serializable:
    """Return a serializable object.

    Ideally, I would like to use the `settings` fixture directly in the `parametrize`
    mark for the `serializable` fixture, but this is not yet possible in pytest.
    See: https://github.com/pytest-dev/pytest/issues/349
    This fixture is a workaround.
    """
    cls_obj_map = {
        Kernel: Kernel(),
        OpenAITextCompletion: OpenAITextCompletion(
            model_id="text-davinci-003", settings=kernel_settings.openai
        ),
        OpenAIChatCompletion: OpenAIChatCompletion(
            model_id="gpt-3.5-turbo", settings=kernel_settings.openai
        ),
        OpenAITextEmbedding: OpenAITextEmbedding(
            model_id="text-embedding-ada-002", settings=kernel_settings.openai
        ),
        PromptTemplateConfig: PromptTemplateConfig(),
        PromptTemplateEngine: PromptTemplateEngine(),
        PromptTemplate: PromptTemplate(
            template="",
            template_engine=PromptTemplateEngine(),
            prompt_config=PromptTemplateConfig(),
        ),
        PassThroughWithoutRetry: PassThroughWithoutRetry(),
        SkillCollection: SkillCollection(),
        TemplateTokenizer: TemplateTokenizer(),
        Block: Block(),
        ChatRequestSettings: ChatRequestSettings(),
        CodeBlock: CodeBlock(),
        CodeTokenizer: CodeTokenizer(),
        CodeRenderer: CodeRenderer(),
        CompleteRequestSettings: CompleteRequestSettings(),
        ContextVariables: ContextVariables(),
    }
    return cls_obj_map[serializable_type]


@pytest.mark.parametrize(
    "serializable_type",
    [
        # pytest.param(Kernel, marks=pytest.mark.xfail(reason="Not implemented")),
        # Kernel,
        Block,
        ChatRequestSettings,
        CodeBlock,
        CodeRenderer,
        CodeTokenizer,
        CompleteRequestSettings,
        ContextVariables,
        OpenAIChatCompletion,
        OpenAITextCompletion,
        OpenAITextEmbedding,
        PromptTemplateConfig,
        PromptTemplateEngine,
        PromptTemplate,
        PassThroughWithoutRetry,
        SkillCollection,
        TemplateTokenizer,
    ],
)
def test_serialization(serializable: _Serializable) -> None:
    """Test serialization of an object to JSON."""
    serialized = serializable.json()
    assert isinstance(serialized, str), serialized
    deserialized = serializable.parse_raw(serialized)
    try:
        assert serializable == deserialized
    except AssertionError:
        # If your class does not implement any equality checks, we want to ensure that
        # we're not doing an id comparison and instead check the attributes.
        if vars(serializable) != vars(deserialized):
            raise
