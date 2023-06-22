# Copyright (c) Microsoft. All rights reserved.

import typing as t
from typing import TYPE_CHECKING

import pydantic as pdt

from semantic_kernel.pydantic_ import SKGenericModel

if TYPE_CHECKING:
    from semantic_kernel.skill_definition.skill_collection_base import (
        SkillCollectionBase,
    )


SkillCollectionsT = t.TypeVar("SkillCollectionsT", bound="SkillCollectionBase")


class ReadOnlySkillCollection(SKGenericModel, t.Generic[SkillCollectionsT]):
    _skill_collection: SkillCollectionsT = pdt.Field(
        alias="skill_collection", description="The skill collection."
    )

    def __getattr__(self, name: str):
        return getattr(self._skill_collection, name)
