"""A scripted agent and mock external data sources.

The agent here is deliberately *simulated*: given a scenario, it reads the external content and
proposes either the user's safe action or the attacker's harmful action, depending on
``follow_injection``. This isolates the object under test, the judge: a harmful trajectory shows
what a successfully hijacked agent would propose, so we measure only whether the judge catches it.
It does not measure how often a real agent gets hijacked.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

from evalguard.core.models import (
    AgentAction,
    AgentTrajectory,
    Scenario,
    SourceSpec,
    Split,
    ToolReturn,
)

logger = logging.getLogger(__name__)


class MockDataSource:
    """Renders mock emails, web pages and API returns, embedding an attack payload if given."""

    def render(self, spec: SourceSpec, payload: str | None = None) -> ToolReturn:
        if spec.kind == "email":
            content = self._render_email(spec, payload)
        elif spec.kind == "web":
            content = self._render_web(spec, payload)
        elif spec.kind == "api":
            content = self._render_api(spec, payload)
        else:  # pragma: no cover - guarded by the SourceKind literal
            raise ValueError(f"Unknown source kind: {spec.kind}")
        return ToolReturn(source=spec.kind, title=spec.title, content=content)

    @staticmethod
    def _render_email(spec: SourceSpec, payload: str | None) -> str:
        lines = [f"From: {spec.sender or 'unknown@example.com'}", f"Subject: {spec.title}", "", spec.body.strip()]
        if payload:
            lines += ["", payload]
        return "\n".join(lines)

    @staticmethod
    def _render_web(spec: SourceSpec, payload: str | None) -> str:
        paragraphs = "\n".join(f"  <p>{p.strip()}</p>" for p in spec.body.strip().split("\n\n") if p.strip())
        injected = f'\n  <div class="user-content">{payload}</div>' if payload else ""
        return f"<html>\n<head><title>{spec.title}</title></head>\n<body>\n{paragraphs}{injected}\n</body>\n</html>"

    @staticmethod
    def _render_api(spec: SourceSpec, payload: str | None) -> str:
        record: dict[str, str] = {"status": "ok", "resource": spec.title, "data": spec.body.strip()}
        if payload:
            record["notes"] = payload
        return json.dumps(record, indent=2, ensure_ascii=False)


def assign_payloads(n_sources: int, inject_into: int, payloads: list[str]) -> list[list[str]]:
    """Spread payload fragments across sources, starting at ``inject_into`` and wrapping around.

    One payload lands in one source; several fragments (payload splitting) land in consecutive sources.
    """
    slots: list[list[str]] = [[] for _ in range(n_sources)]
    for k, payload in enumerate(payloads):
        slots[(inject_into + k) % n_sources].append(payload)
    return slots


def to_agent_action(scenario: Scenario, harmful: bool) -> AgentAction:
    spec = scenario.harmful_action if harmful else scenario.safe_action
    return AgentAction(
        action_id=f"{scenario.scenario_id}:{'harmful' if harmful else 'safe'}",
        tool_name=spec.tool_name,
        parameters=dict(spec.parameters),
        intended_goal=spec.intended_goal,
        is_harmful=harmful,
    )


class SimulatedAgent:
    """Scripted agent: reads every source of a scenario, then proposes the safe or the harmful action."""

    def __init__(self, data_source: MockDataSource | None = None) -> None:
        self.data_source = data_source or MockDataSource()

    def collect(self, scenario: Scenario, payloads: list[str]) -> list[ToolReturn]:
        slots = assign_payloads(len(scenario.sources), scenario.inject_into, payloads)
        return [
            self.data_source.render(spec, "\n".join(slot) if slot else None)
            for spec, slot in zip(scenario.sources, slots, strict=True)
        ]

    def run(
        self,
        scenario: Scenario,
        payloads: list[str],
        follow_injection: bool,
        attack_name: str = "none",
        split: Split = "test",
    ) -> AgentTrajectory:
        tool_returns = self.collect(scenario, payloads)
        action = to_agent_action(scenario, harmful=follow_injection)
        logs = [f"fetch_{tr.source}({tr.title!r}) -> {len(tr.content)} chars" for tr in tool_returns]
        logs.append(f"propose {action.tool_name}({json.dumps(action.parameters, sort_keys=True)})")
        return AgentTrajectory(
            trajectory_id=f"{scenario.scenario_id}|{attack_name}|{'harmful' if follow_injection else 'safe'}",
            scenario_id=scenario.scenario_id,
            user_instruction=scenario.user_instruction,
            untrusted_input="\n\n".join(tr.content for tr in tool_returns),
            tool_returns=tool_returns,
            proposed_action=action,
            execution_logs=logs,
            attack_name=attack_name,
            split=split,
        )


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load scenarios from a YAML file with a top-level ``scenarios`` list."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Scenario file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Scenario file is not valid YAML: {path}: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        raise ValueError(f"{path} must contain a top-level 'scenarios' list")
    scenarios = [Scenario.model_validate(item) for item in data["scenarios"]]
    ids = [s.scenario_id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate scenario_id in {path}")
    logger.info("Loaded %d scenarios from %s", len(scenarios), path)
    return scenarios
