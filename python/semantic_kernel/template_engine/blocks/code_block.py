# Copyright (c) Microsoft. All rights reserved.

import copy
from typing import Any, List, Optional, Tuple

import pydantic as pdt

from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from semantic_kernel.skill_definition.read_only_skill_collection import (
    ReadOnlySkillCollection,
    SkillCollectionsT,
)
from semantic_kernel.template_engine.blocks.block import Block
from semantic_kernel.template_engine.blocks.block_types import BlockTypes
from semantic_kernel.template_engine.blocks.function_id_block import FunctionIdBlock
from semantic_kernel.template_engine.protocols.code_renderer import CodeRenderer


class CodeBlock(Block, CodeRenderer):
    type: BlockTypes = BlockTypes.CODE
    tokens: List[Block] = pdt.Field(default_factory=list)

    @pdt.validator("tokens")
    def validate_tokens(cls, tokens: List[Block]) -> List[Block]:
        """Validate tokens list.

        Args:
            tokens (List[Block]): List of tokens to validate
        """
        if len(tokens) > 1:
            if tokens[0].type != BlockTypes.FUNCTION_ID:
                raise ValueError(f"Invalid first token: {tokens[0]}")
            elif tokens[1].type not in (
                BlockTypes.VARIABLE,
                BlockTypes.VALUE,
            ):
                raise ValueError(f"Invalid second token: {tokens[1]}")
        if len(tokens) > 2:
            raise ValueError(f"Invalid number of tokens: {len(tokens)}")
        return tokens

    @pdt.validator("tokens", each_item=True)
    def validate_each_token(cls, token: Any) -> Block:
        """Validate each token.

        Args:
            v (Any): An element of the tokens list
        """
        if not token.is_valid():
            raise ValueError(f"Invalid token: {token}")
        return token

    def is_valid(self) -> Tuple[bool, str]:
        error_msg = ""

        for token in self.tokens:
            is_valid, error_msg = token.is_valid()
            if not is_valid:
                self.logger.error(error_msg)
                return False, error_msg

        if len(self.tokens) > 1:
            if self.tokens[0].type != BlockTypes.FUNCTION_ID:
                error_msg = f"Unexpected second token found: {self.tokens[1].content}"
                self.logger.error(error_msg)
                return False, error_msg

            if (
                self.tokens[1].type != BlockTypes.VALUE
                and self.tokens[1].type != BlockTypes.VARIABLE
            ):
                error_msg = "Functions support only one parameter"
                self.logger.error(error_msg)
                return False, error_msg

        if len(self.tokens) > 2:
            error_msg = f"Unexpected second token found: {self.tokens[1].content}"
            self.logger.error(error_msg)
            return False, error_msg

        self._validated = True

        return True, ""

    async def render_code_async(self, context):
        if not self._validated:
            is_valid, error = self.is_valid()
            if not is_valid:
                raise ValueError(error)

        self.logger.debug(f"Rendering code: `{self.content}`")

        if self.tokens[0].type in (BlockTypes.VALUE, BlockTypes.VARIABLE):
            return self.tokens[0].render(context.variables)

        if self.tokens[0].type == BlockTypes.FUNCTION_ID:
            return await self._render_function_call_async(self.tokens[0], context)

        raise ValueError(f"Unexpected first token type: {self.tokens[0].type}")

    async def _render_function_call_async(self, f_block: FunctionIdBlock, context):
        if not context.skills:
            raise ValueError("Skill collection not set")

        function = self._get_function_from_skill_collection(context.skills, f_block)

        if not function:
            error_msg = f"Function `{f_block.content}` not found"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        variables_clone = copy.copy(context.variables)

        if len(self.tokens) > 1:
            self.logger.debug(f"Passing variable/value: `{self.tokens[1].content}`")
            input_value = self.tokens[1].render(variables_clone)
            variables_clone.update(input_value)

        result = await function.invoke_async(
            variables=variables_clone, memory=context.memory, log=self.logger
        )

        if result.error_occurred:
            error_msg = (
                f"Function `{f_block.content}` execution failed. "
                f"{result.last_exception.__class__.__name__}: "
                f"{result.last_error_description}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        return result.result

    def _get_function_from_skill_collection(
        self,
        skills: ReadOnlySkillCollection[SkillCollectionsT],
        f_block: FunctionIdBlock,
    ) -> Optional[SKFunctionBase]:
        if not f_block.skill_name and skills.has_function(None, f_block.function_name):
            return skills.get_function(None, f_block.function_name)

        if f_block.skill_name and skills.has_function(
            f_block.skill_name, f_block.function_name
        ):
            return skills.get_function(f_block.skill_name, f_block.function_name)

        return None
