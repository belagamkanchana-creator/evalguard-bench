import json
import os

import pytest

from evalguard.core.judge import (
    JudgeError,
    JudgeSettings,
    LLMJudge,
    OpenAIBackend,
    make_backend,
    parse_absolute,
    parse_pairwise,
)
from evalguard.core.models import AgentTrajectory
from evalguard.defenses.base import Defense


@pytest.mark.parametrize("raw, expected", [
    ('{"reasoning": "ok", "score": 5}', 5.0),
    ('```json\n{"reasoning": "bad", "score": 2}\n```', 2.0),
    ('Thinking... {"score": 4, "reasoning": "draft"} final: {"reasoning": "no", "score": 1}', 1.0),
    ("Reasoning: fine\nScore: 4", 4.0),
    ('{"score": "3"}', 3.0),
])
def test_parse_absolute(raw, expected):
    assert parse_absolute(raw)[0] == expected


@pytest.mark.parametrize("raw", ["", "no idea", '{"score": 9}', '{"reasoning": "x"}'])
def test_parse_absolute_unparseable(raw):
    assert parse_absolute(raw)[0] is None


@pytest.mark.parametrize("raw, expected", [
    ('{"reasoning": "x", "winner": "B"}', "B"),
    ('{"winner": "trajectory a"}', "A"),
    ("Winner: B", "B"),
    ("both are fine", None),
])
def test_parse_pairwise(raw, expected):
    assert parse_pairwise(raw)[0] == expected


def test_threshold_decides(scenarios, agent, scripted):
    t = agent.run(scenarios[0], [], True)
    assert LLMJudge(scripted('{"score": 4}')).evaluate(t).decision == "APPROVE"
    assert LLMJudge(scripted('{"score": 3}')).evaluate(t).decision == "REJECT"
    strict = LLMJudge(scripted('{"score": 4}'), settings=JudgeSettings(approve_threshold=5))
    assert strict.evaluate(t).decision == "REJECT"


def test_unparseable_and_errors_fail_closed(scenarios, agent, scripted):
    t = agent.run(scenarios[0], [], True)
    out = LLMJudge(scripted("hmm")).evaluate(t)
    assert out.decision == "REJECT" and not out.parse_ok

    def boom(system, user):
        raise JudgeError("network down")
    out = LLMJudge(scripted(boom)).evaluate(t)
    assert out.decision == "REJECT" and not out.parse_ok and "network down" in out.error


class DropAll(Defense):
    name = "drop_all"

    def screen(self, trajectory: AgentTrajectory):
        return "always"


def test_screening_defense_skips_the_model(scenarios, agent, scripted):
    backend = scripted('{"score": 5}')
    out = LLMJudge(backend, [DropAll()]).evaluate(agent.run(scenarios[0], [], True))
    assert out.blocked_by == "drop_all" and out.decision == "REJECT"
    assert backend.calls == []


def test_pairwise_swapping_detects_positional_bias(scenarios, agent, scripted):
    safe, harmful = agent.run(scenarios[0], [], False), agent.run(scenarios[0], [], True)
    always_a = LLMJudge(scripted('{"winner": "A"}')).evaluate_pairwise(safe, harmful)
    assert not always_a.consistent and always_a.final_choice_id is None

    def pick_safe(system, user):
        block_a = user.split("=== TRAJECTORY B ===")[0]
        return json.dumps({"winner": "A" if scenarios[0].safe_action.tool_name in block_a else "B"})
    picker = LLMJudge(scripted(pick_safe)).evaluate_pairwise(safe, harmful)
    assert picker.consistent and picker.final_choice_id == safe.trajectory_id
    assert picker.first_pass_winner == "A" and picker.swapped_pass_winner == "B"


def test_pairwise_dropped_trajectory_loses(scenarios, agent, scripted):
    class DropHarmful(Defense):
        name = "drop_harmful"

        def screen(self, trajectory):
            return "x" if trajectory.proposed_action.is_harmful else None

    safe, harmful = agent.run(scenarios[0], [], False), agent.run(scenarios[0], [], True)
    out = LLMJudge(scripted('{"winner": "A"}'), [DropHarmful()]).evaluate_pairwise(safe, harmful)
    assert out.final_choice_id == safe.trajectory_id and out.blocked_by == "drop_harmful"


def test_make_backend_specs(monkeypatch):
    assert make_backend("mock", JudgeSettings()).name == "mock"
    with pytest.raises(ValueError):
        make_backend("gpt4", JudgeSettings())
    with pytest.raises(ValueError):
        make_backend("foo:bar", JudgeSettings())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(JudgeError):
        OpenAIBackend("gpt-4o-mini", JudgeSettings())


@pytest.mark.skipif(os.environ.get("EVALGUARD_OLLAMA_TESTS") != "1", reason="needs a local Ollama server")
def test_ollama_backend_live():  # pragma: no cover
    backend = make_backend(os.environ.get("EVALGUARD_OLLAMA_MODEL", "ollama:llama3.1:8b"), JudgeSettings())
    assert isinstance(backend.generate("Reply with OK.", "Say OK"), str)
