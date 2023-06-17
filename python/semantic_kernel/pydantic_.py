import abc
import typing as t

import pydantic as pdt
import typing_extensions as te


class PydanticABC(abc.ABC):
    """An Abstract Base Class that can be used as a pydantic field."""

    @classmethod
    def __get_validators__(cls) -> t.Iterable[t.Callable]:
        """Gets the validators for the class."""
        yield cls.validate

    @classmethod
    def validate(cls, v: t.Any) -> t.Any:
        """Does no validation, just returns the value."""
        return v


class Serializable(abc.ABC):
    """Serialization protocol followed by pydantic `BaseModel`s.

    If you want your custom class to be serializable when it's used as a pydanitc field
    type, you must subclass this class and implement the `json` method.
    """

    @abc.abstractmethod
    def json(self) -> str:
        """Serialize the model to JSON."""
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    @abc.abstractmethod
    def parse_raw(
        cls: t.Type[te.Self], b: bytes, *, content_type: t.Optional[str] = None
    ) -> te.Self:
        """Parse a raw byte string into a model."""
        raise NotImplementedError("Subclasses must implement this method.")


class SKBaseModel(pdt.BaseModel):
    """Base class for all pydantic models in the SK."""

    class Config:
        """Base configuration for all pydantic models in the SK."""

        json_encoders: t.Dict[t.Type[t.Any], t.Callable[[t.Any], t.Any]] = {
            Serializable: lambda v: v.json(),
        }
