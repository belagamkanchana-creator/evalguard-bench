"""Pydantic data models shared across the benchmark."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Decision = Literal["APPROVE", "REJECT"]
JudgeMode = Literal["absolute", "pairwise"]
Split = Literal["train", "test"]
SourceKind = Literal["email", "web", "api"]


class ActionSpec(BaseModel):
    """An action as written in a scenario file (before it becomes an AgentAction)."""

    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    intended_goal: str


class SourceSpec(BaseModel):
    """One piece of external content the agent reads: an email, a web page, or an API return."""

    kind: SourceKind
    title: str
    body: str
    sender: str | None = None


class Scenario(BaseModel):
    """A task with a safe action (what the user wants) and a harmful action (what an attacker wants)."""

    scenario_id: str
    domain: str
    user_instruction: str
    sources: list[SourceSpec] = Field(min_length=1)
    inject_into: int = 0
    """Index of the source that receives the attack payload (the first one, for split payloads)."""
    safe_action: ActionSpec
    harmful_action: ActionSpec
    attacker_goal: str
    """The harmful action in plain English, as an attacker would phrase the instruction."""

    @model_validator(mode="after")
    def _check_inject_index(self) -> Scenario:
        if not 0 <= self.inject_into < len(self.sources):
            raise ValueError(
                f"{self.scenario_id}: inject_into={self.inject_into} but there are {len(self.sources)} sources"
            )
        return self


class ToolReturn(BaseModel):
    """Rendered external content, exactly as the agent (and the judge) sees it."""

    source: SourceKind
    title: str
    content: str


class AgentAction(BaseModel):
    action_id: str
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    intended_goal: str
    is_harmful: bool


class AgentTrajectory(BaseModel):
    trajectory_id: str
    scenario_id: str
    user_instruction: str
    untrusted_input: str
    """All external content concatenated; this is what input filters (e.g. perplexity) inspect."""
    tool_returns: list[ToolReturn] = Field(default_factory=list)
    proposed_action: AgentAction
    execution_logs: list[str] = Field(default_factory=list)
    attack_name: str = "none"
    split: Split = "test"


class JudgeOutput(BaseModel):
    judge_mode: JudgeMode
    decision: Decision
    score: float = Field(ge=1.0, le=5.0)
    reasoning: str = ""
    raw_prompt: str = ""
    raw_response: str = ""
    parse_ok: bool = True
    """False when the judge's reply could not be parsed; the decision then fails closed to REJECT."""
    blocked_by: str | None = None
    """Name of the defense that dropped the input before the judge saw it, if any."""
    error: str | None = None


class PairwiseOutput(BaseModel):
    """Result of comparing two trajectories twice, once in each order (positional swapping)."""

    judge_mode: Literal["pairwise"] = "pairwise"
    first_pass_winner: Literal["A", "B"] | None = None
    swapped_pass_winner: Literal["A", "B"] | None = None
    first_choice_id: str | None = None
    swapped_choice_id: str | None = None
    consistent: bool = False
    """True when both orders picked the same trajectory."""
    final_choice_id: str | None = None
    """The trajectory picked in both orders; None if the passes disagree or a reply was unparseable."""
    reasoning_first: str = ""
    reasoning_swapped: str = ""
    raw_prompts: list[str] = Field(default_factory=list)
    blocked_by: str | None = None
