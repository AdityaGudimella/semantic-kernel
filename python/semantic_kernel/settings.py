"""Settings to configure the semantic kernel."""
import enum
import typing as t
from pathlib import Path

import pydantic as pdt
from pydantic.env_settings import SettingsSourceCallable
from yaml import safe_load

from semantic_kernel.logging_ import LoggerSettings
from semantic_kernel.optional_packages.chromadb import Settings as ChromaDBSettings
from semantic_kernel.utils.openai_ import OpenAIAPIKwargs


class SKBaseSettings(pdt.BaseSettings):
    class Config:
        """Base Pydantic configuration for all SemanticKernel Settings classes."""

        # One shouldn't be able to modify the settings once they're loaded.
        allow_mutation = False


def yaml_config_source(config: pdt.BaseSettings) -> dict[str, str]:
    """Config source that loads variables from a yaml file in home directory.

    This function is used by pydantic to automatically load settings from a yaml file
    in the settings path.
    """
    settings_path = getattr(
        config, "SETTINGS_PATH", KernelSettings.DEFAULT_SETTINGS_PATH
    )
    return safe_load(settings_path.read_text()) if settings_path.exists() else {}


class OpenAISettings(SKBaseSettings):
    """Settings to configure the OpenAI API."""

    api_key: pdt.SecretStr = pdt.Field(
        description="OpenAI API key. See: https://platform.openai.com/account/api-keys",
    )
    org_id: t.Optional[str] = pdt.Field(
        None,
        description=(
            "OpenAI organization ID."
            + " See: https://platform.openai.com/account/org-settings"
            + " Only required if your account belongs to multiple organizations."
        ),
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
    _openai_api_kwargs: OpenAIAPIKwargs = pdt.PrivateAttr()

    def __init__(self, **data: t.Any) -> None:
        super().__init__(**data)
        self._openai_api_kwargs = OpenAIAPIKwargs(
            api_key=self.api_key.get_secret_value(),
            api_type=self.api_type,
            api_base=self.endpoint,
            api_version=self.api_version,
            organization=self.org_id,
        )

    @property
    def openai_api_kwargs(self) -> OpenAIAPIKwargs:
        """Get the kwargs used for the OpenAI API calls."""
        return self._openai_api_kwargs


class AzureAPIType(str, enum.Enum):
    """Azure API type."""

    Azure = "azure"
    AzureAD = "azure_ad"


class AzureOpenAISettings(SKBaseSettings):
    """Settings to configure the Azure OpenAI API."""

    api_key: pdt.SecretStr = pdt.Field(
        description=(
            "Azure OpenAI API key. See: ?"
            + " This value can be found in the Keys & Endpoint section when examining"
            + " your resource in the Azure portal. You can use either KEY1 or KEY2."
        )
    )
    endpoint: str = pdt.Field(
        description=(
            "Azure OpenAI API endpoint."
            + " This value can be found in the Keys & Endpoint section when examining"
            + " your resource from the Azure portal."
        )
    )
    api_version: str = pdt.Field(
        default="2023-03-15-preview",
        description="Azure OpenAI API version. See: ?",
    )
    ad_auth: bool = pdt.Field(
        default=False,
        description="Whether to use Azure Active Directory authentication.",
    )
    _openai_settings: OpenAISettings = pdt.PrivateAttr()
    _openai_api_kwargs: OpenAIAPIKwargs = pdt.PrivateAttr()

    @pdt.validator("endpoint")
    def validate_endpoint(cls, v: str) -> str:
        """Validate the endpoint."""
        if not v.startswith("https://"):
            raise ValueError("Endpoint must start with 'https://'")
        return v

    def __init__(self, **data: t.Any) -> None:
        super().__init__(**data)
        openai_kwargs = {
            k: v for k, v in data.items() if k in OpenAISettings.__fields__
        }
        self._openai_settings = OpenAISettings(**openai_kwargs)
        self._openai_api_kwargs = OpenAIAPIKwargs(
            api_key=self.api_key.get_secret_value(),
            api_type="azure",
            api_base=self.endpoint,
            api_version=self.api_version,
            organization=None,
        )

    @property
    def api_type(self) -> AzureAPIType:
        """Get the Azure API type."""
        return AzureAPIType.AzureAD if self.ad_auth else AzureAPIType.Azure

    @property
    def openai_settings(self) -> OpenAISettings:
        """Get the OpenAI settings."""
        return self._openai_settings

    @property
    def openai_api_kwargs(self) -> OpenAIAPIKwargs:
        """Get the kwargs used for the OpenAI API calls."""
        return self._openai_api_kwargs


class KernelSettings(SKBaseSettings):
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
        default=DEFAULT_SETTINGS_PATH,
        description="Path to the directory containing the settings file.",
    )
    openai: OpenAISettings
    azure_openai: t.Optional[AzureOpenAISettings] = pdt.Field(
        default=None,
        description="Settings to configure the Azure OpenAI API.",
    )
    chroma_db: ChromaDBSettings = pdt.Field(
        default_factory=ChromaDBSettings,
        description="Settings to configure the ChromaDB.",
    )
    logging: LoggerSettings = pdt.Field(
        default_factory=LoggerSettings,
        description="Settings to configure the logging.",
    )

    class Config:  # type: ignore
        """Pydantic configuration."""

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
