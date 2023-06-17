# Copyright (c) Microsoft. All rights reserved.

import typing as t
import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_kernel.skill_definition.skill_collection_base import (
        SkillCollectionBase,
    )


SkillCollectionsT = t.TypeVar("SkillCollectionsT", bound="SkillCollectionBase")


class ReadOnlySkillCollection(t.Generic[SkillCollectionsT]):
    def __init__(
        self,
        skill_collection: t.Union[
            weakref.ReferenceType[SkillCollectionsT], SkillCollectionsT
        ],
    ) -> None:
        if not isinstance(skill_collection, weakref.ReferenceType):
            skill_collection = weakref.ref(skill_collection)
        self._skill_collection = skill_collection

    def __getattr__(self, name: str):
        return getattr(self._skill_collection, name)
