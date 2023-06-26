# Copyright (c) Microsoft. All rights reserved.

import pytest

from semantic_kernel.template_engine.blocks.block import Block
from semantic_kernel.template_engine.blocks.block_types import BlockTypes


def test_init_content_whitespace():
    """When I initialize a block with whitespace content, it should be stripped"""
    block = Block(content="  test content  ")
    assert block.content == "test content"


@pytest.fixture()
def block() -> Block:
    return Block()


def test_type_property(block: Block):
    assert block.type == BlockTypes.UNDEFINED
