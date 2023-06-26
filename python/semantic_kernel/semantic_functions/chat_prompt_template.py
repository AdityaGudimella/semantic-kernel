# Copyright (c) Microsoft. All rights reserved.

from typing import TYPE_CHECKING, List, Tuple

import pydantic as pdt

from semantic_kernel.semantic_functions.prompt_template import PromptTemplate

if TYPE_CHECKING:
    from semantic_kernel.orchestration.sk_context import SKContext


class ChatPromptTemplate(PromptTemplate):
    _messages: List[Tuple[str, PromptTemplate]] = pdt.PrivateAttr()

    async def render_async(self, context: "SKContext") -> str:
        raise NotImplementedError(
            "Can't call render_async on a ChatPromptTemplate.\n"
            + "Use render_messages_async instead."
        )

    def add_system_message(self, message: str) -> None:
        self.add_message("system", message)

    def add_user_message(self, message: str) -> None:
        self.add_message("user", message)

    def add_assistant_message(self, message: str) -> None:
        self.add_message("assistant", message)

    def add_message(self, role: str, message: str) -> None:
        self._messages.append(
            (
                role,
                PromptTemplate(
                    template=message,
                    template_engine=self.template_engine,
                    prompt_config=self.prompt_config,
                ),
            )
        )

    async def render_messages_async(
        self, context: "SKContext"
    ) -> List[Tuple[str, str]]:
        rendered_messages = []
        for role, message in self._messages:
            rendered_messages.append((role, await message.render_async(context)))

        latest_user_message = await self.template_engine.render_async(
            self.template, context
        )
        rendered_messages.append(("user", latest_user_message))

        return rendered_messages
