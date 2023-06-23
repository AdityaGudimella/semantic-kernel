# Copyright (c) Microsoft. All rights reserved.

import pytest
from test_utils import retry

import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.settings import OpenAISettings


@pytest.mark.asyncio
async def test_oai_text_completion_with_skills(
    setup_tldr_function_for_oai_models, openai_settings: OpenAISettings
):
    kernel, sk_prompt, text_to_summarize = setup_tldr_function_for_oai_models

    print("* Service: OpenAI Text Completion")
    print("* Endpoint: OpenAI")
    print("* Model: text-davinci-003")

    kernel.add_chat_service(
        "davinci-003",
        sk_oai.OpenAITextCompletion(
            model_id="text-davinci-003", settings=openai_settings
        ),
    )

    # Create the semantic function
    tldr_function = kernel.create_semantic_function(
        sk_prompt, max_tokens=200, temperature=0, top_p=0.5
    )

    summary = await retry(
        lambda: kernel.run_async(tldr_function, input_str=text_to_summarize)
    )
    output = str(summary).strip()
    print(f"TLDR using input string: '{output}'")
    assert "First Law" not in output and (
        "human" in output or "Human" in output or "preserve" in output
    )
    assert len(output) < 100
