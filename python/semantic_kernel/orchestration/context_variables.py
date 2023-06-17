# Copyright (c) Microsoft. All rights reserved.
import collections.abc as cabc
import json as json_
import typing as t

from semantic_kernel.pydantic_ import Serializable


class ContextVariables(cabc.MutableMapping[str, str], Serializable):
    """Variables that can be used to store and retrieve data.

    This class is a MutableMapping subclass, which means it behaves like a dictionary.
    """

    _MAIN_KEY: t.Final[str] = "input"

    @classmethod
    def __get_validators__(cls) -> t.Iterable[t.Callable]:
        """Gets the validators for the class."""
        yield cls.validate

    @classmethod
    def validate(cls, v: t.Any) -> "ContextVariables":
        """Validates the value.

        Args:
            v: The value to validate.

        Returns:
            The validated value.
        """
        if isinstance(v, cls):
            return v
        if isinstance(v, str):
            try:
                loaded = json_.loads(v)
            except json_.JSONDecodeError as e:
                raise ValueError("Invalid JSON string") from e
            if isinstance(loaded, cabc.Mapping):
                return cls(**loaded)
            if isinstance(loaded, str):
                return cls(content=loaded)
        if isinstance(v, cabc.Mapping):
            return cls(**v)
        raise TypeError(f"Invalid type: {type(v)}")

    def __init__(self, content: str = "input", **kwargs: str) -> None:
        """Initializes a new instance of the ContextVariables class.

        Args:
            content: The initial value for the "input" variable.
            **kwargs: Additional variables to be added to the collection.
        """
        self._variables = {self._MAIN_KEY: content, **kwargs}

    @property
    def input(self) -> str:
        """Gets the value of the "input" variable.

        Returns:
            The value of the "input" variable.
        """
        return self._variables[self._MAIN_KEY]

    def __contains__(self, key: object) -> bool:
        """Checks whether the collection contains a variable with the specified name.

        Args:
            key: The name of the variable to check.

        Returns:
            `True` if the collection contains a variable with the specified name;
            otherwise, `False`.
        """
        return key in self._variables

    def __getitem__(self, key: str) -> str:
        """Gets the value of the variable with the specified name.

        Args:
            key: The name of the variable to get.

        Returns:
            The value of the variable.
        """
        return self._variables[key]

    def __setitem__(self, key: str, value: str) -> None:
        """Sets the value of the variable with the specified name.

        Args:
            key: The name of the variable to set.
            value: The new value for the variable.
        """
        if not key:
            raise ValueError("The variable name cannot be `None` or empty")
        key = key.lower()
        self._variables[key] = value

    def __delitem__(self, key: str) -> None:
        """Deletes the variable with the specified name.

        Args:
            key: The name of the variable to delete.
        """
        del self._variables[key]

    def __iter__(self) -> t.Iterator[str]:
        """Returns an iterator over the names of the variables in the collection.

        Returns:
            An iterator over the names of the variables.
        """
        return iter(self._variables)

    def __len__(self) -> int:
        """Returns the number of variables in the collection.

        Returns:
            The number of variables in the collection.
        """
        return len(self._variables)

    def __repr__(self) -> str:
        """Returns a string representation of the ContextVariables object.

        Returns:
            A string representation of the ContextVariables object.
        """
        return f"ContextVariables({self._variables})"

    def to_dict(self) -> t.Dict[str, str]:
        """Converts the collection to a dictionary.

        Returns:
            A dictionary containing the variables in the collection.
        """
        return dict(self._variables)

    def json(self, **kwargs: t.Any) -> str:
        """Returns the JSON representation of the object.

        Args:
            **kwargs: Additional arguments to pass to json.dumps().

        Returns:
            The JSON representation of the object.
        """
        return json_.dumps(self.to_dict(), **kwargs)

    @classmethod
    def parse_raw(
        cls, b: t.Union[str, bytes], *, content_type: t.Optional[str] = None
    ) -> "ContextVariables":
        """Parses the raw data.

        Args:
            b: The raw data to parse.
            content_type: The content type of the raw data.

        Returns:
            The parsed data.
        """
        return cls(**json_.loads(b))

    class Encoder(json_.JSONEncoder):
        """JSON encoder for ContextVariables objects."""

        def default(self, obj: t.Any) -> t.Any:
            """Encodes the object to JSON.

            Args:
                obj: The object to encode.

            Returns:
                The JSON representation of the object.
            """
            if isinstance(obj, ContextVariables):
                return obj.to_dict()
            return super().default(obj)
