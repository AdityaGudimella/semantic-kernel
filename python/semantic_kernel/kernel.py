# Copyright (c) Microsoft. All rights reserved.

import glob
import importlib
import inspect
import os
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import uuid4

import pydantic as pdt

from semantic_kernel.connectors.ai.ai_exception import AIException
from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.connectors.ai.chat_request_settings import ChatRequestSettings
from semantic_kernel.connectors.ai.complete_request_settings import (
    CompleteRequestSettings,
)
from semantic_kernel.connectors.ai.embeddings.embedding_generator_base import (
    EmbeddingGeneratorBase,
)
from semantic_kernel.connectors.ai.text_completion_client_base import (
    TextCompletionClientBase,
)
from semantic_kernel.kernel_exception import KernelException
from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.memory.null_memory import NullMemory
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.semantic_text_memory_base import SemanticTextMemoryBase
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.orchestration.sk_context import SKContext
from semantic_kernel.orchestration.sk_function import SKFunction
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from semantic_kernel.pydantic_ import SKBaseModel, SKGenericModel
from semantic_kernel.reliability.pass_through_without_retry import (
    PassThroughWithoutRetry,
)
from semantic_kernel.reliability.retry_mechanism_base import RetryMechanismBase
from semantic_kernel.semantic_functions.prompt_template import PromptTemplate
from semantic_kernel.semantic_functions.prompt_template_config import (
    CompletionConfig,
    PromptTemplateConfig,
)
from semantic_kernel.semantic_functions.semantic_function_config import (
    SemanticFunctionConfig,
)
from semantic_kernel.skill_definition.read_only_skill_collection import (
    ReadOnlySkillCollection,
    SkillCollectionsT,
)
from semantic_kernel.skill_definition.skill_collection import SkillCollection
from semantic_kernel.skill_definition.skill_collection_base import SkillCollectionBase
from semantic_kernel.template_engine.prompt_template_engine import PromptTemplateEngine
from semantic_kernel.template_engine.protocols.prompt_templating_engine import (
    PromptTemplatingEngine,
)
from semantic_kernel.utils.validation import validate_function_name, validate_skill_name

T = TypeVar("T")


ServiceT = TypeVar(
    "ServiceT",
    bound=Union[
        TextCompletionClientBase, ChatCompletionClientBase, EmbeddingGeneratorBase
    ],
)


Service = Callable[["Kernel"], ServiceT]


class Services(SKBaseModel):
    """A collection of services that are used by the kernel."""

    text_completion: Dict[str, Service[TextCompletionClientBase]] = pdt.Field(
        default_factory=dict,
        description="A collection of text completion services that are used by the kernel.",  # noqa: E501
    )
    chat_completion: Dict[str, Service[ChatCompletionClientBase]] = pdt.Field(
        default_factory=dict,
        description="A collection of chat completion services that are used by the kernel.",  # noqa: E501
    )
    embedding_generator: Dict[str, Service[EmbeddingGeneratorBase]] = pdt.Field(
        default_factory=dict,
        description="A collection of embedding generators that are used by the kernel.",
    )


class DefaultServices(SKBaseModel):
    """A collection of default services that are used by the kernel."""

    text_completion: Optional[str] = pdt.Field(
        default=None,
        description="The default text completion service that is used by the kernel.",
    )
    chat_completion: Optional[str] = pdt.Field(
        default=None,
        description="The default chat completion service that is used by the kernel.",
    )
    embedding_generator: Optional[str] = pdt.Field(
        default=None,
        description="The default embedding generator that is used by the kernel.",
    )


class Kernel(SKGenericModel, Generic[SkillCollectionsT]):
    logger: SKLogger = pdt.Field(
        default_factory=NullLogger,
        description="The logger that is used by the kernel.",
    )
    skill_collection: SkillCollectionsT = pdt.Field(
        default=None,
        description="The skill collection that contains all the skills that are loaded into the kernel.",  # noqa: E501
    )
    prompt_template_engine: PromptTemplatingEngine = pdt.Field(
        default=None,
        description="The prompt template engine that is used to generate prompts for the user.",  # noqa: E501
    )
    memory: SemanticTextMemoryBase = pdt.Field(
        default_factory=NullMemory,
        description="The memory that is used by the kernel to store information.",
    )
    services: Services = pdt.Field(
        default_factory=Services,
        description="A collection of services that are used by the kernel.",
    )
    default_services: DefaultServices = pdt.Field(
        default_factory=DefaultServices,
        description="A collection of default services that are used by the kernel.",
    )
    _retry_mechanism: RetryMechanismBase = pdt.Field(
        default_factory=PassThroughWithoutRetry,
        description="The retry mechanism that is used by the kernel.",
    )

    @pdt.validator(
        "skill_collection",
        pre=True,
        # NOTE: `allow_reuse` is required because SkillCollectionsT defines it's own
        # validator.
        allow_reuse=True,
    )
    def skill_collection_validator(cls, v: Any, **kwargs) -> SkillCollectionBase:
        """Validate the skill collection."""
        assert kwargs.get("logger") is not None
        return SkillCollection(logger=kwargs["logger"]) if v is None else v

    @pdt.validator(
        "prompt_template_engine",
        pre=True,
        # NOTE: `allow_reuse` is required because PromptTemplatingEngine defines it's
        # own validator.
        allow_reuse=True,
    )
    def prompt_template_engine_validator(
        cls, v: Any, **kwargs: Any
    ) -> PromptTemplateEngine:
        """Validate the skill collection."""
        assert kwargs.get("logger") is not None
        return PromptTemplateEngine(logger=kwargs["logger"]) if v is None else v

    @property
    def skills(self) -> ReadOnlySkillCollection[SkillCollection]:
        return self.skill_collection.read_only_skill_collection

    def register_semantic_function(
        self,
        skill_name: Optional[str],
        function_name: str,
        function_config: SemanticFunctionConfig,
    ) -> SKFunctionBase:
        if skill_name is None or skill_name == "":
            skill_name = SkillCollection.GLOBAL_SKILL
        assert skill_name is not None  # for type checker

        validate_skill_name(skill_name)
        validate_function_name(function_name)

        function = self._create_semantic_function(
            skill_name, function_name, function_config
        )
        self.skill_collection.add_semantic_function(function)

        return function

    async def run_async(
        self,
        *functions: Any,
        input_context: Optional[SKContext] = None,
        input_vars: Optional[ContextVariables] = None,
        input_str: Optional[str] = None,
    ) -> SKContext:
        # if the user passed in a context, prioritize it, but merge with any other inputs
        if input_context is not None:
            context = input_context
            if input_vars is not None:
                context.variables = ContextVariables(
                    **{**input_vars, **context.variables}
                )

            if input_str is not None:
                context.variables = ContextVariables(input_str, **context._variables)

        # if the user did not pass in a context, prioritize an input string, and merge that with input context variables
        else:
            if input_str is not None and input_vars is None:
                variables = ContextVariables(input_str)
            elif input_str is None and input_vars is not None:
                variables = input_vars
            elif input_str is not None and input_vars is not None:
                variables = ContextVariables(input_str)
                variables.update(input_vars)
            else:
                variables = ContextVariables()
            context = SKContext(
                variables,
                self._memory,
                self.skill_collection.read_only_skill_collection,
                self.logger,
            )

        pipeline_step = 0
        for func in functions:
            assert isinstance(func, SKFunctionBase), (
                "All func arguments to Kernel.run*(inputs, func1, func2, ...) "
                "must be SKFunctionBase instances"
            )

            if context.error_occurred:
                self.logger.error(
                    f"Something went wrong in pipeline step {pipeline_step}. "
                    f"Error description: '{context.last_error_description}'"
                )
                return context

            pipeline_step += 1

            try:
                context = await func.invoke_async(input=None, context=context)

                if context.error_occurred:
                    self.logger.error(
                        f"Something went wrong in pipeline step {pipeline_step}. "
                        f"During function invocation: '{func.skill_name}.{func.name}'. "
                        f"Error description: '{context.last_error_description}'"
                    )
                    return context
            except Exception as ex:
                self.logger.error(
                    f"Something went wrong in pipeline step {pipeline_step}. "
                    f"During function invocation: '{func.skill_name}.{func.name}'. "
                    f"Error description: '{str(ex)}'"
                )
                context.fail(str(ex), ex)
                return context

        return context

    def func(self, skill_name: str, function_name: str) -> SKFunctionBase:
        if self.skills.has_native_function(skill_name, function_name):
            return self.skills.get_native_function(skill_name, function_name)

        return self.skills.get_semantic_function(skill_name, function_name)

    def use_memory(
        self,
        storage: MemoryStoreBase,
        embeddings_generator: Optional[EmbeddingGeneratorBase] = None,
    ) -> None:
        if embeddings_generator is None:
            service_id = self.get_text_embedding_generation_service_id()
            if not service_id:
                raise ValueError("The embedding service id cannot be `None` or empty")

            if embeddings_service := self.get_ai_service(
                EmbeddingGeneratorBase, service_id
            ):
                embeddings_generator = embeddings_service(self)

            else:
                raise ValueError(f"AI configuration is missing for: {service_id}")

        if storage is None:
            raise ValueError("The storage instance provided cannot be `None`")
        if embeddings_generator is None:
            raise ValueError("The embedding generator cannot be `None`")

        self.register_memory(SemanticTextMemory(storage, embeddings_generator))

    def register_memory(self, memory: SemanticTextMemoryBase) -> None:
        self._memory = memory

    def register_memory_store(self, memory_store: MemoryStoreBase) -> None:
        self.use_memory(memory_store)

    def create_new_context(self) -> SKContext:
        return SKContext(
            ContextVariables(),
            self._memory,
            self.skills,
            self.logger,
        )

    def import_skill(
        self, skill_instance: Any, skill_name: str = ""
    ) -> Dict[str, SKFunctionBase]:
        if not skill_name.strip():
            skill_name = SkillCollection.GLOBAL_SKILL
            self.logger.debug(f"Importing skill {skill_name} into the global namespace")
        else:
            self.logger.debug(f"Importing skill {skill_name}")

        functions = [
            SKFunction.from_native_method(candidate, skill_name, self.logger)
            for _, candidate in inspect.getmembers(skill_instance, inspect.ismethod)
            if hasattr(candidate, "__sk_function__")
        ]
        self.logger.debug(f"Methods imported: {len(functions)}")

        # Uniqueness check on function names
        function_names = [f.name for f in functions]
        if len(function_names) != len(set(function_names)):
            raise KernelException(
                KernelException.ErrorCodes.FunctionOverloadNotSupported,
                "Overloaded functions are not supported, "
                "please differentiate function names.",
            )

        skill = {}
        for function in functions:
            function.set_default_skill_collection(self.skills)
            self.skill_collection.add_native_function(function)
            skill[function.name] = function

        return skill

    def get_ai_service(
        self, type: Type[T], service_id: Optional[str] = None
    ) -> Callable[["Kernel"], T]:
        matching_type = {}
        if type == TextCompletionClientBase:
            service_id = service_id or self._default_text_completion_service
            matching_type = self._text_completion_services
        elif type == ChatCompletionClientBase:
            service_id = service_id or self._default_chat_service
            matching_type = self._chat_services
        elif type == EmbeddingGeneratorBase:
            service_id = service_id or self._default_text_embedding_generation_service
            matching_type = self._text_embedding_generation_services
        else:
            raise ValueError(f"Unknown AI service type: {type.__name__}")

        if service_id not in matching_type:
            raise ValueError(
                f"{type.__name__} service with service_id '{service_id}' not found"
            )

        return matching_type[service_id]

    def all_text_completion_services(self) -> List[str]:
        return list(self._text_completion_services.keys())

    def all_chat_services(self) -> List[str]:
        return list(self._chat_services.keys())

    def all_text_embedding_generation_services(self) -> List[str]:
        return list(self._text_embedding_generation_services.keys())

    def add_text_completion_service(
        self,
        service_id: str,
        service: Union[
            TextCompletionClientBase, Callable[["Kernel"], TextCompletionClientBase]
        ],
        overwrite: bool = True,
    ) -> "Kernel":
        if not service_id:
            raise ValueError("service_id must be a non-empty string")
        if not overwrite and service_id in self._text_completion_services:
            raise ValueError(
                f"Text service with service_id '{service_id}' already exists"
            )

        self._text_completion_services[service_id] = (
            service if isinstance(service, Callable) else lambda _: service
        )
        if self._default_text_completion_service is None:
            self._default_text_completion_service = service_id

        return self

    def add_chat_service(
        self,
        service_id: str,
        service: Union[
            ChatCompletionClientBase, Callable[["Kernel"], ChatCompletionClientBase]
        ],
        overwrite: bool = True,
    ) -> "Kernel":
        if not service_id:
            raise ValueError("service_id must be a non-empty string")
        if not overwrite and service_id in self._chat_services:
            raise ValueError(
                f"Chat service with service_id '{service_id}' already exists"
            )

        self._chat_services[service_id] = (
            service if isinstance(service, Callable) else lambda _: service
        )
        if self._default_chat_service is None:
            self._default_chat_service = service_id

        if isinstance(service, TextCompletionClientBase):
            self.add_text_completion_service(service_id, service)
            if self._default_text_completion_service is None:
                self._default_text_completion_service = service_id

        return self

    def add_text_embedding_generation_service(
        self,
        service_id: str,
        service: Union[
            EmbeddingGeneratorBase, Callable[["Kernel"], EmbeddingGeneratorBase]
        ],
        overwrite: bool = False,
    ) -> "Kernel":
        if not service_id:
            raise ValueError("service_id must be a non-empty string")
        if not overwrite and service_id in self._text_embedding_generation_services:
            raise ValueError(
                f"Embedding service with service_id '{service_id}' already exists"
            )

        self._text_embedding_generation_services[service_id] = (
            service if isinstance(service, Callable) else lambda _: service
        )
        if self._default_text_embedding_generation_service is None:
            self._default_text_embedding_generation_service = service_id

        return self

    def set_default_text_completion_service(self, service_id: str) -> "Kernel":
        if service_id not in self._text_completion_services:
            raise ValueError(
                f"AI service with service_id '{service_id}' does not exist"
            )

        self._default_text_completion_service = service_id
        return self

    def set_default_chat_service(self, service_id: str) -> "Kernel":
        if service_id not in self._chat_services:
            raise ValueError(
                f"AI service with service_id '{service_id}' does not exist"
            )

        self._default_chat_service = service_id
        return self

    def set_default_text_embedding_generation_service(
        self, service_id: str
    ) -> "Kernel":
        if service_id not in self._text_embedding_generation_services:
            raise ValueError(
                f"AI service with service_id '{service_id}' does not exist"
            )

        self._default_text_embedding_generation_service = service_id
        return self

    def get_text_completion_service_service_id(
        self, service_id: Optional[str] = None
    ) -> str:
        if service_id is None or service_id not in self._text_completion_services:
            if self._default_text_completion_service is None:
                raise ValueError("No default text service is set")
            return self._default_text_completion_service

        return service_id

    def get_chat_service_service_id(self, service_id: Optional[str] = None) -> str:
        if service_id is None or service_id not in self._chat_services:
            if self._default_chat_service is None:
                raise ValueError("No default chat service is set")
            return self._default_chat_service

        return service_id

    def get_text_embedding_generation_service_id(
        self, service_id: Optional[str] = None
    ) -> str:
        if (
            service_id is None
            or service_id not in self._text_embedding_generation_services
        ):
            if self._default_text_embedding_generation_service is None:
                raise ValueError("No default embedding service is set")
            return self._default_text_embedding_generation_service

        return service_id

    def remove_text_completion_service(self, service_id: str) -> "Kernel":
        if service_id not in self._text_completion_services:
            raise ValueError(
                f"AI service with service_id '{service_id}' does not exist"
            )

        del self._text_completion_services[service_id]
        if self._default_text_completion_service == service_id:
            self._default_text_completion_service = next(
                iter(self._text_completion_services), None
            )
        return self

    def remove_chat_service(self, service_id: str) -> "Kernel":
        if service_id not in self._chat_services:
            raise ValueError(
                f"AI service with service_id '{service_id}' does not exist"
            )

        del self._chat_services[service_id]
        if self._default_chat_service == service_id:
            self._default_chat_service = next(iter(self._chat_services), None)
        return self

    def remove_text_embedding_generation_service(self, service_id: str) -> "Kernel":
        if service_id not in self._text_embedding_generation_services:
            raise ValueError(
                f"AI service with service_id '{service_id}' does not exist"
            )

        del self._text_embedding_generation_services[service_id]
        if self._default_text_embedding_generation_service == service_id:
            self._default_text_embedding_generation_service = next(
                iter(self._text_embedding_generation_services), None
            )
        return self

    def clear_all_text_completion_services(self) -> "Kernel":
        self._text_completion_services = {}
        self._default_text_completion_service = None
        return self

    def clear_all_chat_services(self) -> "Kernel":
        self._chat_services = {}
        self._default_chat_service = None
        return self

    def clear_all_text_embedding_generation_services(self) -> "Kernel":
        self._text_embedding_generation_services = {}
        self._default_text_embedding_generation_service = None
        return self

    def clear_all_services(self) -> "Kernel":
        self._text_completion_services = {}
        self._chat_services = {}
        self._text_embedding_generation_services = {}

        self._default_text_completion_service = None
        self._default_chat_service = None
        self._default_text_embedding_generation_service = None

        return self

    def _create_semantic_function(
        self,
        skill_name: str,
        function_name: str,
        function_config: SemanticFunctionConfig,
    ) -> SKFunctionBase:
        function_type = function_config.prompt_template_config.type
        if function_type != "completion":
            raise AIException(
                AIException.ErrorCodes.FunctionTypeNotSupported,
                f"Function type not supported: {function_type}",
            )

        function = SKFunction.from_semantic_config(
            skill_name, function_name, function_config
        )
        function.request_settings.update(
            **function_config.prompt_template_config.completion.dict()
        )

        # Connect the function to the current kernel skill
        # collection, in case the function is invoked manually
        # without a context and without a way to find other functions.
        function.set_default_skill_collection(self.skills)

        if function_config.has_chat_prompt:
            service = self.get_ai_service(
                ChatCompletionClientBase,
                function_config.prompt_template_config.default_services[0]
                if len(function_config.prompt_template_config.default_services) > 0
                else None,
            )

            function.set_chat_configuration(
                ChatRequestSettings(
                    **function_config.prompt_template_config.completion.dict()
                )
            )

            if service is None:
                raise AIException(
                    AIException.ErrorCodes.InvalidConfiguration,
                    "Could not load chat service, unable to prepare semantic function. "
                    "Function description: "
                    "{function_config.prompt_template_config.description}",
                )

            function.set_chat_service(lambda: service(self))
        else:
            service = self.get_ai_service(
                TextCompletionClientBase,
                function_config.prompt_template_config.default_services[0]
                if len(function_config.prompt_template_config.default_services) > 0
                else None,
            )

            function.set_ai_configuration(
                CompleteRequestSettings(
                    **function_config.prompt_template_config.completion.dict()
                )
            )

            if service is None:
                raise AIException(
                    AIException.ErrorCodes.InvalidConfiguration,
                    "Could not load text service, unable to prepare semantic function. "
                    "Function description: "
                    "{function_config.prompt_template_config.description}",
                )

            function.set_ai_service(lambda: service(self))

        return function

    def import_native_skill_from_directory(
        self, parent_directory: str, skill_directory_name: str
    ) -> Dict[str, SKFunctionBase]:
        MODULE_NAME = "native_function"

        validate_skill_name(skill_directory_name)

        skill_directory = os.path.abspath(
            os.path.join(parent_directory, skill_directory_name)
        )
        native_py_file_path = os.path.join(skill_directory, f"{MODULE_NAME}.py")

        if not os.path.exists(native_py_file_path):
            raise ValueError(
                f"Native Skill Python File does not exist: {native_py_file_path}"
            )

        skill_name = os.path.basename(skill_directory)
        try:
            spec = importlib.util.spec_from_file_location(
                MODULE_NAME, native_py_file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            class_name = next(
                (
                    name
                    for name, cls in inspect.getmembers(module, inspect.isclass)
                    if cls.__module__ == MODULE_NAME
                ),
                None,
            )
            if class_name:
                skill_obj = getattr(module, class_name)()
                return self.import_skill(skill_obj, skill_name)
        except Exception:
            pass

        return {}

    def import_semantic_skill_from_directory(
        self, parent_directory: str, skill_directory_name: str
    ) -> Dict[str, SKFunctionBase]:
        CONFIG_FILE = "config.json"
        PROMPT_FILE = "skprompt.txt"

        validate_skill_name(skill_directory_name)

        skill_directory = os.path.join(parent_directory, skill_directory_name)
        skill_directory = os.path.abspath(skill_directory)

        if not os.path.exists(skill_directory):
            raise ValueError(f"Skill directory does not exist: {skill_directory_name}")

        skill = {}

        directories = glob.glob(f"{skill_directory}/*/")
        for directory in directories:
            dir_name = os.path.dirname(directory)
            function_name = os.path.basename(dir_name)
            prompt_path = os.path.join(directory, PROMPT_FILE)

            # Continue only if the prompt template exists
            if not os.path.exists(prompt_path):
                continue

            config = PromptTemplateConfig()
            config_path = os.path.join(directory, CONFIG_FILE)
            with open(config_path, "r") as config_file:
                config = config.from_json(config_file.read())

            # Load Prompt Template
            with open(prompt_path, "r") as prompt_file:
                template = PromptTemplate(
                    prompt_file.read(), self.prompt_template_engine, config
                )

            # Prepare lambda wrapping AI logic
            function_config = SemanticFunctionConfig(config, template)

            skill[function_name] = self.register_semantic_function(
                skill_directory_name, function_name, function_config
            )

        return skill

    def create_semantic_function(
        self,
        prompt_template: str,
        function_name: Optional[str] = None,
        skill_name: Optional[str] = None,
        description: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        number_of_responses: int = 1,
        stop_sequences: Optional[List[str]] = None,
    ) -> "SKFunctionBase":
        function_name = (
            function_name
            if function_name is not None
            else f"f_{str(uuid4()).replace('-', '_')}"
        )

        config = PromptTemplateConfig(
            description=(
                description
                if description is not None
                else "Generic function, unknown purpose"
            ),
            type="completion",
            completion=CompletionConfig(
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                number_of_responses=number_of_responses,
                stop_sequences=stop_sequences if stop_sequences is not None else [],
            ),
        )

        validate_function_name(function_name)
        if skill_name is not None:
            validate_skill_name(skill_name)

        template = PromptTemplate(
            template=prompt_template,
            template_engine=self.prompt_template_engine,
            prompt_config=config,
        )
        function_config = SemanticFunctionConfig(config, template)

        return self.register_semantic_function(
            skill_name, function_name, function_config
        )
