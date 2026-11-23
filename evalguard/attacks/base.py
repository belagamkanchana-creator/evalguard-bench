"""Base class for attacks: an attack turns a scenario into payload text hidden in external content."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evalguard.core.models import Scenario


@dataclass(frozen=True)
class AttackContext:
    """What the attacker knows: the harmful action they want. Not the user's request."""

    scenario_id: str
    attacker_goal: str
    harmful_tool: str
    harmful_parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> AttackContext:
        return cls(
            scenario_id=scenario.scenario_id,
            attacker_goal=scenario.attacker_goal,
            harmful_tool=scenario.harmful_action.tool_name,
            harmful_parameters=dict(scenario.harmful_action.parameters),
        )


class Attack(ABC):
    name: str = "base"
    family: str = "base"

    @abstractmethod
    def build_payloads(self, ctx: AttackContext) -> list[str]:
        """Payload strings to hide in tool returns. More than one string means the payload is split
        across consecutive tool returns."""
