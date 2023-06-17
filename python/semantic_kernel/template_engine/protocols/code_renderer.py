# Copyright (c) Microsoft. All rights reserved.

import pydantic as pdt

from semantic_kernel.orchestration.sk_context import SKContext


class CodeRenderer(pdt.BaseModel):
    """
    Protocol for dynamic code blocks that need async IO to be rendered.
    """

    async def render_code_async(self, context: SKContext) -> str:
        """
        Render the block using the given context.

        :param context: SK execution context
        :return: Rendered content
        """
        raise NotImplementedError("Subclasses must implement this method.")
