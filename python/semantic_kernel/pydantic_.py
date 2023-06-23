import abc
import json
import typing as t

import numpy as np
import pydantic as pdt
import typing_extensions as te
from pydantic.fields import ModelField
from pydantic.generics import GenericModel
from pydantic.parse import Protocol
from pydantic.types import StrBytes
from pydantic.utils import to_lower_camel

from semantic_kernel.logging_ import NullLogger, SKLogger


class PydanticField(abc.ABC):
    """Subclass this class to make your class a valid pydantic field type.

    This class is a no-op, but it's necessary to make pydantic recognize your class as
    a valid field type. See https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types
    for more information.

    - If you want to add validation to your class, you can do so by implementing the
    `__get_validators__` class method. See
    https://pydantic-docs.helpmanual.io/usage/validators/ for more information.
    - If you want to add serialization to your class, you can do so by implementing the
    `json` and `parse_raw` methods. See
    https://pydantic-docs.helpmanual.io/usage/exporting_models/#json for more information.
    """

    @classmethod
    def __get_validators__(cls) -> t.Generator[t.Callable[..., t.Any], None, None]:
        """Gets the validators for the class."""
        yield cls.no_op_validate

    @classmethod
    def no_op_validate(cls, v: t.Any) -> t.Any:
        """Does no validation, just returns the value."""
        return v

    def json(self) -> str:
        """Serialize the model to JSON."""
        return ""

    @classmethod
    def parse_raw(
        cls: t.Type[te.Self],
        b: StrBytes,
        *,
        content_type: str = None,
        encoding: str = "utf8",
        proto: Protocol = None,
        allow_pickle: bool = False,
    ) -> te.Self:
        """Parse a raw byte string into a model."""
        return cls()


_JSON_ENCODERS: t.Final[t.Dict[t.Type[t.Any], t.Callable[[t.Any], str]]] = {
    PydanticField: lambda v: v.json(),
    np.ndarray: lambda v: json.dumps(v.tolist()),
}


class SKBaseModel(pdt.BaseModel):
    """Base class for all pydantic models in the SK."""

    logger: SKLogger = pdt.Field(default_factory=NullLogger)

    @pdt.validator("logger", pre=True)
    def _validate_logger(cls, v: t.Any) -> SKLogger:
        """Validate the logger."""
        if v is None:
            return NullLogger()
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Expected logger to be an SKLogger, but got {type(v)}."
                ) from e
        if isinstance(v, dict):
            return SKLogger(**v)
        if isinstance(v, SKLogger):
            return v
        raise ValueError(f"Expected logger to be an SKLogger, but got {type(v)}.")

    class Config:
        """Pydantic configuration."""

        json_encoders = _JSON_ENCODERS
        # Prevent mutation of models after they are created.
        # This seems to be the default behavior in SemanticKernel
        allow_mutation = False
        # See the `allow_population_by_field_name` section of
        # https://docs.pydantic.dev/latest/usage/model_config/#options
        allow_population_by_field_name = True
        # Alias `snake_case`d to lowerCamelCase
        # Eg: `external_source_name` -> `externalSourceName`
        alias_generator = to_lower_camel


class SKGenericModel(GenericModel):
    """Base class for all pydantic `GenericModel`s in the SK."""

    class Config:
        """Pydantic configuration."""

        json_encoders = _JSON_ENCODERS
        # Prevent mutation of models after they are created.
        # This seems to be the default behavior in SemanticKernel
        allow_mutation = False


ShapeT = t.TypeVar("ShapeT")
DTypeT = t.TypeVar("DTypeT")


class PydanticNDArray(PydanticField, np.ndarray[ShapeT, np.dtype[DTypeT]]):
    """Use this only to annotate numpy arrays in pydantic models."""

    @classmethod
    def __modify_schema__(
        cls, field_schema: dict[str, t.Any], field: t.Optional[ModelField]
    ) -> None:
        if field and field.sub_fields:
            type_with_potential_subtype = f"np.ndarray[{field.sub_fields[0]}]"
        else:
            type_with_potential_subtype = "np.ndarray"
        # Originally: field_schema.update({"type": type_with_potential_subtype})
        field_schema["type"] = type_with_potential_subtype

    @classmethod
    def __get_validators__(cls) -> t.Generator[t.Callable[..., t.Any], None, None]:
        """Gets the validators for the class."""
        yield cls.validate

    @classmethod
    def validate(cls, v: t.Any) -> t.Any:
        """Does no validation, just returns the value."""
        if isinstance(v, str):
            v = json.loads(v)
        if not isinstance(v, np.ndarray):
            try:
                v = np.asarray(v)
            except Exception as e:
                raise ValueError(f"Could not convert {v} to a numpy array.") from e
        return v

    def json(self) -> str:
        return json.dumps(self.tolist())

    @classmethod
    def parse_raw(
        cls: t.Type[te.Self],
        b: StrBytes,
        *,
        content_type: str = None,
        encoding: str = "utf8",
        proto: Protocol = None,
        allow_pickle: bool = False,
    ) -> te.Self:
        """Parse a raw byte string into a model."""
        if isinstance(b, bytes):
            return cls(np.asarray(json.loads(b.decode("utf-8"))))
        return cls(np.asarray(json.loads(b)))
