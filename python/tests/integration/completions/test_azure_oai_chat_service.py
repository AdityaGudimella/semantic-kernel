# Copyright (c) Microsoft. All rights reserved.

import os

import pytest
from test_utils import retry

import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.settings import AzureOpenAISettings


@pytest.mark.asyncio
async def test_azure_e2e_chat_completion_with_skill(
    setup_tldr_function_for_oai_models, azure_openai_settings: AzureOpenAISettings
):
    kernel, sk_prompt, text_to_summarize = setup_tldr_function_for_oai_models

    if "Python_Integration_Tests" in os.environ:
        deployment_name = os.environ["AzureOpenAIChat__DeploymentName"]
    else:
        deployment_name = "gpt-35-turbo"

    print("* Service: Azure OpenAI Chat Completion")
    print(f"* Endpoint: {azure_openai_settings.endpoint}")
    print(f"* Deployment: {deployment_name}")

    # Configure LLM service
    kernel.add_chat_service(
        "chat_completion",
        sk_oai.AzureChatCompletion(
            deployment=deployment_name,
            settings=azure_openai_settings,
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
