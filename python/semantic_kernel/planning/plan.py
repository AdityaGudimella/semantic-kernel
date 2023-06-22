import typing as t

import pydantic as pdt

from semantic_kernel.orchestration.sk_context import SKContext
from semantic_kernel.pydantic_ import SKGenericModel
from semantic_kernel.skill_definition.read_only_skill_collection import SkillCollectionT


class Plan(SKGenericModel, t.Generic[SkillCollectionT]):
    goal: str = pdt.Field(description="The goal that wants to be achieved")
    prompt: str = pdt.Field(description="The prompt to be used to generate the plan")
    generated_plan: SKContext[SkillCollectionT] = pdt.Field(
        description="The generated plan that consists of a list of steps to complete the goal"
    )
