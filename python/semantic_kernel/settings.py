"""Settings to configure the semantic kernel."""
import typing as t
from pathlib import Path

import pydantic as pdt
from pydantic.env_settings import SettingsSourceCallable
from yaml import safe_load

from semantic_kernel.logging_ import LoggerSettings


def yaml_config_source(config: pdt.BaseSettings) -> dict[str, str]:
    """Config source that loads variables from a yaml file in home directory.

    This function is used by pydantic to automatically load settings from a yaml file
    in the settings path.
    """
    settings_path = getattr(
        config, "SETTINGS_PATH", KernelSettings.DEFAULT_SETTINGS_PATH
    )
    return safe_load(settings_path.read_text()) if settings_path.exists() else {}


class OpenAISettings(pdt.BaseSettings):
    """Settings to configure the OpenAI API."""

    api_key: str = pdt.Field(
        description="OpenAI API key. See: https://platform.openai.com/account/api-keys",
    )
    org_id: t.Optional[str] = pdt.Field(
        None,  # Only required if your account belongs to multiple organizations.
        description="OpenAI organization ID. See: https://platform.openai.com/account/org-settings",  # noqa: E501
    )
    api_type: t.Optional[str] = pdt.Field(
        None,
        description="OpenAI API type. See: ?",
    )
    api_version: t.Optional[str] = pdt.Field(
        None,
        description="OpenAI API version. See: ?",
    )
    endpoint: t.Optional[str] = pdt.Field(
        None,
        description="OpenAI API endpoint. See: ?",
    )

    class Config:
        """Pydantic configuration."""

        # One shouldn't be able to modify the settings once they're loaded.
        allow_mutation = False


class KernelSettings(pdt.BaseSettings):
    """Settings to configure a semantic kernel `Kernel` object.

    If you have a yaml file at `DEFAULT_SETTINGS_PATH` with the correct configuration,
    you can initialize the KernelSettings by just calling KernelSettings(). This also
    works if some of the settings are defined in the yaml file, and others are passed
    in as keyword arguments or set as environment variables etc.
    `/.semantic_kernel/settings.yaml`:
    open_ai:
        api_key: "..."
        org_id: "..."
    """

    DEFAULT_SETTINGS_PATH: t.ClassVar[pdt.FilePath] = (
        Path.home() / ".semantic_kernel" / "settings.yaml"
    )

    settings_path: Path = pdt.Field(
        DEFAULT_SETTINGS_PATH,
        description="Path to the directory containing the settings file.",
    )
    openai: OpenAISettings
    logging: LoggerSettings = pdt.Field(
        default_factory=LoggerSettings,
        description="Settings to configure the logging.",
    )

    class Config:  # type: ignore
        """Pydantic configuration."""

        # One shouldn't be able to modify the settings once they're loaded.
        allow_mutation = False

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            """Customise the config sources.

            Reorder the config sources so that we prefer yaml files over environment
            variables.
            """
            # Prioritize in the order:
            return (
                # First: values passed into __init__
                init_settings,
                # Second: values present in cls.SETTINGS_PATH file if present
                yaml_config_source,
                # Third: values present in environment variables
                env_settings,
                # Finally: values present in file secrets if any
                file_secret_settings,
            )


def load_settings() -> KernelSettings:
    """Load the settings for a semantic kernel.

    Convenience method so that I don't have to add a type: ignore everywhere.
    """
    return KernelSettings()  # type: ignore
