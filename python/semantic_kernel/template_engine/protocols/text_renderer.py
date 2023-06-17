# Copyright (c) Microsoft. All rights reserved.

import typing as t

import pydantic as pdt

from semantic_kernel.orchestration.context_variables import ContextVariables


class TextRenderer(pdt.BaseModel):
    """
    Protocol for static (text) blocks that don't need async rendering.
    """

    def render(self, variables: t.Optional[ContextVariables] = None) -> str:
        """
        Render the block using only the given variables.

        :param variables: Optional variables used to render the block
        :return: Rendered content
        """
        raise NotImplementedError("Subclasses must implement this method.")
