import typing as t

import pytest

import semantic_kernel as sk


@pytest.fixture()
def chat(
    kernel: sk.Kernel, chat_function: sk.SKFunctionBase
) -> t.Callable[[str], t.Awaitable[sk.SKContext]]:
    async def chat(user_input: str) -> sk.SKContext:
        context_vars = sk.ContextVariables()
        context_vars["user_input"] = user_input
        return await kernel.run_async(chat_function, input_vars=context_vars)

    return chat


@pytest.mark.asyncio
@pytest.mark.parametrize("service_type", ["chat", "text_completion"])
@pytest.mark.parametrize("user_input", ["What is your name?"])
async def test_chat(
    chat: t.Callable[[str], t.Awaitable[sk.SKContext]], user_input: str
) -> None:
    response = await chat(user_input)
    assert not response.error_occurred, response.result
    assert isinstance(response.result, str), response.result
