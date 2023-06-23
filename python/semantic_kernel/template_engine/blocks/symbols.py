# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.pydantic_ import PydanticField


class Symbols(PydanticField):
    BLOCK_STARTER = "{"
    BLOCK_ENDER = "}"

    VAR_PREFIX = "$"

    DBL_QUOTE = '"'
    SGL_QUOTE = "'"
    ESCAPE_CHAR = "\\"

    SPACE = " "
    TAB = "\t"
    NEW_LINE = "\n"
    CARRIAGE_RETURN = "\r"
