import abc
import typing as t

import pydantic as pdt
import typing_extensions as te
from pydantic.generics import GenericModel


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


_JSON_ENCODERS: t.Final[t.Dict[t.Type[t.Any], t.Callable[[t.Any], str]]] = {
    Serializable: lambda v: v.json(),
}


class SKBaseModel(pdt.BaseModel):
    """Base class for all pydantic models in the SK."""

    class Config:
        """Pydantic configuration."""

        json_encoders = _JSON_ENCODERS


class SKGenericModel(GenericModel):
    """Base class for all pydantic `GenericModel`s in the SK."""

    class Config:
        """Pydantic configuration."""

        json_encoders = _JSON_ENCODERS
