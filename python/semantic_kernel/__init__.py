# Copyright (c) Microsoft. All rights reserved.

import typing as t
from pathlib import Path

from semantic_kernel import core_skills, memory
from semantic_kernel.kernel import Kernel
from semantic_kernel.logging_ import NullLogger, SKLogger
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.orchestration.sk_context import SKContext
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from semantic_kernel.semantic_functions.chat_prompt_template import ChatPromptTemplate
from semantic_kernel.semantic_functions.prompt_template import PromptTemplate
from semantic_kernel.semantic_functions.prompt_template_config import (
    PromptTemplateConfig,
)
from semantic_kernel.semantic_functions.semantic_function_config import (
    SemanticFunctionConfig,
)
from semantic_kernel.settings import (
    AzureOpenAISettings,
    KernelSettings,
    OpenAISettings,
    load_settings,
)

__version__ = "0.2.8.dev"

REPO_ROOT: t.Final[Path] = Path(__file__).parent.parent.parent
PYTHON_REPO_ROOT: t.Final[Path] = REPO_ROOT / "python"
PYTHON_PACKAGE_ROOT: t.Final[Path] = PYTHON_REPO_ROOT / "semantic_kernel"

__all__ = [
    "Kernel",
    "SKLogger",
    "NullLogger",
    "PromptTemplateConfig",
    "PromptTemplate",
    "ChatPromptTemplate",
    "SemanticFunctionConfig",
    "ContextVariables",
    "SKFunctionBase",
    "SKContext",
    "memory",
    "core_skills",
    "__version__",
    "REPO_ROOT",
    "AzureOpenAISettings",
    "KernelSettings",
    "OpenAISettings",
    "load_settings",
]
