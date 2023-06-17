"""Tests for `semantic_kernel.settings`."""

from semantic_kernel.settings import KernelSettings, load_settings


def test_load_settings() -> None:
    """I should be able to load the settings in the test environment.

    If this test fails, a majority of the other tests will fail as well.
    """
    settings = load_settings()
    assert isinstance(settings, KernelSettings)
    assert settings.openai.api_key, "OpenAI API key not set."
    assert settings.logging.get_logger("test").level == "DEBUG"
