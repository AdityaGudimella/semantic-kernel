# Copyright (c) Microsoft. All rights reserved.

import os

import pytest
from test_utils import retry

import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.core_skills.conversation_summary_skill import (
    ConversationSummarySkill,
)


@pytest.mark.asyncio
async def test_azure_summarize_conversation_using_skill(
    setup_summarize_conversation_using_skill,
    azure_openai_settings: sk.AzureOpenAISettings,
):
    kernel, chatTranscript = setup_summarize_conversation_using_skill

    if "Python_Integration_Tests" in os.environ:
        deployment_name = os.environ["AzureOpenAI__DeploymentName"]
    else:
        deployment_name = "text-davinci-003"

    kernel.add_text_completion_service(
        "text_completion",
        sk_oai.AzureTextCompletion(
            deployment=deployment_name,
            settings=azure_openai_settings,
        ),
    )

    conversationSummarySkill = kernel.import_skill(
        ConversationSummarySkill(kernel), "conversationSummary"
    )

    summary = await retry(
        lambda: kernel.run_async(
            conversationSummarySkill["SummarizeConversation"], input_str=chatTranscript
        )
    )

    output = str(summary).strip().lower()
    print(output)
    assert "john" in output and "jane" in output
    assert len(output) < len(chatTranscript)


@pytest.mark.asyncio
async def test_oai_summarize_conversation_using_skill(
    setup_summarize_conversation_using_skill, openai_settings: sk.OpenAISettings
):
    kernel, chatTranscript = setup_summarize_conversation_using_skill

    kernel.add_text_completion_service(
        "davinci-003",
        sk_oai.OpenAITextCompletion(
            model_id="text-davinci-003", settings=openai_settings
        ),
    )

    conversationSummarySkill = kernel.import_skill(
        ConversationSummarySkill(kernel), "conversationSummary"
    )

    summary = await retry(
        lambda: kernel.run_async(
            conversationSummarySkill["SummarizeConversation"], input_str=chatTranscript
        )
    )

    output = str(summary).strip().lower()
    print(output)
    assert "john" in output and "jane" in output
    assert len(output) < len(chatTranscript)
