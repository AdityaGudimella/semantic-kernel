"""Test serialization of SK Kernel."""
import typing as t

import pydantic as pdt
import pytest
import typing_extensions as te

import semantic_kernel as sk
from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.chat_request_settings import ChatRequestSettings
from semantic_kernel.connectors.ai.complete_request_settings import (
    CompleteRequestSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_text_completion import (
    AzureTextCompletion,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_text_embedding import (
    AzureTextEmbedding,
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
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.memory.semantic_text_memory_base import SemanticTextMemoryBase
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from semantic_kernel.planning.basic_planner import BasicPlanner
from semantic_kernel.pydantic_ import PydanticField
from semantic_kernel.reliability.pass_through_without_retry import (
    PassThroughWithoutRetry,
)
from semantic_kernel.reliability.retry_mechanism_base import RetryMechanismBase
from semantic_kernel.semantic_functions.chat_prompt_template import ChatPromptTemplate
from semantic_kernel.semantic_functions.prompt_template import PromptTemplate
from semantic_kernel.semantic_functions.prompt_template_config import (
    PromptTemplateConfig,
)
from semantic_kernel.serialization import from_json, to_json
from semantic_kernel.settings import KernelSettings
from semantic_kernel.skill_definition.skill_collection import SkillCollection
from semantic_kernel.skill_definition.skill_collection_base import SkillCollectionBase
from semantic_kernel.template_engine.blocks.block import Block
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.blocks.code_block import CodeBlock
from semantic_kernel.template_engine.code_tokenizer import CodeTokenizer
from semantic_kernel.template_engine.prompt_template_engine import PromptTemplateEngine
from semantic_kernel.template_engine.protocols.code_renderer import CodeRenderer
from semantic_kernel.template_engine.protocols.prompt_templating_engine import (
    PromptTemplatingEngine,
)
from semantic_kernel.template_engine.protocols.text_renderer import TextRenderer
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
        BasicPlanner,
        BlockTypes,
        ContextVariables,
        ChatCompletionClientBase,
        MemoryStoreBase,
        PromptTemplatingEngine,
        RetryMechanismBase,
        SemanticTextMemoryBase,
        SKFunctionBase,
        SkillCollectionBase,
        TextCompletionClientBase,
        TextRenderer,
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
        AzureTextCompletion: AzureTextCompletion(
            deployment="text-davinci-003", settings=kernel_settings.azure_openai
        ),
        AzureChatCompletion: AzureChatCompletion(
            deployment="gpt-3.5-turbo", settings=kernel_settings.azure_openai
        ),
        AzureTextEmbedding: AzureTextEmbedding(
            deployment="text-embedding-ada-002", settings=kernel_settings.azure_openai
        ),
        Block: Block(),
        ChatPromptTemplate: ChatPromptTemplate(
            template="",
            template_engine=PromptTemplateEngine(),
            prompt_config=PromptTemplateConfig(),
        ),
        ChatRequestSettings: ChatRequestSettings(),
        CodeBlock: CodeBlock(),
        CodeTokenizer: CodeTokenizer(),
        CodeRenderer: CodeRenderer(),
        CompleteRequestSettings: CompleteRequestSettings(),
        ContextVariables: ContextVariables(),
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
        PassThroughWithoutRetry: PassThroughWithoutRetry(),
        PromptTemplateConfig: PromptTemplateConfig(),
        PromptTemplateEngine: PromptTemplateEngine(),
        PromptTemplate: PromptTemplate(
            template="",
            template_engine=PromptTemplateEngine(),
            prompt_config=PromptTemplateConfig(),
        ),
        SkillCollection: SkillCollection(),
        TemplateTokenizer: TemplateTokenizer(),
    }
    return cls_obj_map[serializable_type]


def _recursive_eq(
    exp: t.Union[pdt.BaseModel, pdt.BaseConfig, pdt.BaseSettings, t.Dict[str, t.Any]],
    act: t.Union[pdt.BaseModel, pdt.BaseConfig, pdt.BaseSettings, t.Dict[str, t.Any]],
) -> t.Union[t.Literal[True], t.NoReturn]:
    """Recursively check equality of two objects.

    This is required for the following reasons:

    1. Classes that don't implement an `__eq__` method need to be compared by their
         attributes.
    2. Pydantic `SecretField` objects are not serialized, and so should not be compared
            for equality.

    Args:
        exp: The expected object.
        act: The actual object.

    Returns:
        True if the objects are equal, otherwise raises a `pytest.fail` exception.
    """
    if exp == act:
        return True
    if not isinstance(
        exp, (pdt.BaseModel, pdt.BaseConfig, pdt.BaseSettings, pdt.SecretField, dict)
    ):
        pytest.fail(f"Expected: {exp}, but got: {act}")
    if isinstance(exp, pdt.SecretField) and isinstance(act, pdt.SecretField):
        # Pydantic `SecretField` objects are not serialized, and so should not be
        # compared for equality.
        return True
    if isinstance(exp, (pdt.BaseModel, pdt.BaseConfig, pdt.BaseSettings)):
        if not isinstance(act, (pdt.BaseModel, pdt.BaseConfig, pdt.BaseSettings)):
            pytest.fail(
                f"Expected object of type {type(exp)}, but got object of type {type(act)}"
            )
        exp = exp.dict()
        act = act.dict()
    for key in exp:
        if key not in act:
            pytest.fail(
                f"Expected key {key} not in actual object with keys: {list(act.keys())}"
            )
    if len(exp) != len(act):
        pytest.fail(
            "Expected and actual objects have different numbers of attributes: "
            + f"{len(exp)} != {len(act)} "
            + f"with types: exp -> {type(exp)}, act -> {type(act)} "
            + f"and values: exp -> {exp}, act -> {act}"
        )
    for key in exp:
        if key not in act:
            pytest.fail(f"Expected object has attribute {key} that actual does not.")
        if not _recursive_eq(exp[key], act[key]):
            pytest.fail(
                "Expected and actual objects have different values for attribute "
                + f"{key}: {exp[key]} != {act[key]} "
                + f"with types: exp -> {type(exp[key])}, act -> {type(act[key])}"
            )
    return True


@pytest.mark.parametrize(
    "serializable_type",
    [
        # pytest.param(Kernel, marks=pytest.mark.xfail(reason="Not implemented")),
        # Kernel,
        Block,
        ChatPromptTemplate,
        ChatRequestSettings,
        CodeBlock,
        CodeRenderer,
        CodeTokenizer,
        CompleteRequestSettings,
        ContextVariables,
        OpenAIChatCompletion,
        OpenAITextCompletion,
        OpenAITextEmbedding,
        PassThroughWithoutRetry,
        PromptTemplateConfig,
        PromptTemplateEngine,
        PromptTemplate,
        SkillCollection,
        TemplateTokenizer,
        AzureChatCompletion,
        AzureTextCompletion,
        AzureTextEmbedding,
    ],
)
def test_serialization(serializable: _Serializable) -> None:
    """Test serialization of an object to JSON."""
    serialized = serializable.json()
    assert isinstance(serialized, str), serialized
    deserialized = serializable.parse_raw(serialized)
    assert _recursive_eq(serializable, deserialized)
