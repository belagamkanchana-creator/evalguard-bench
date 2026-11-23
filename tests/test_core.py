import pytest
from pydantic import ValidationError

from evalguard.core.agent import assign_payloads
from evalguard.core.models import Scenario


def test_scenarios_load(scenarios):
    assert len(scenarios) == 8
    for s in scenarios:
        assert len(s.sources) >= 2
        assert s.safe_action.tool_name and s.harmful_action.tool_name


def test_inject_index_is_validated(scenarios):
    data = scenarios[0].model_dump()
    data["inject_into"] = 99
    with pytest.raises(ValidationError):
        Scenario.model_validate(data)


def test_assign_payloads_wraps_around():
    assert assign_payloads(3, 2, ["a"]) == [[], [], ["a"]]
    assert assign_payloads(3, 2, ["a", "b", "c"]) == [["b"], ["c"], ["a"]]
    assert assign_payloads(2, 0, ["a", "b", "c"]) == [["a", "c"], ["b"]]


def test_agent_builds_safe_and_harmful_trajectories(scenarios, agent):
    s = scenarios[0]
    safe = agent.run(s, ["PAYLOAD"], follow_injection=False, attack_name="x")
    harmful = agent.run(s, ["PAYLOAD"], follow_injection=True, attack_name="x")
    assert not safe.proposed_action.is_harmful and harmful.proposed_action.is_harmful
    assert safe.proposed_action.tool_name == s.safe_action.tool_name
    assert harmful.proposed_action.tool_name == s.harmful_action.tool_name
    assert "PAYLOAD" in safe.untrusted_input and "PAYLOAD" in harmful.untrusted_input
    assert "PAYLOAD" in safe.tool_returns[s.inject_into].content
    assert safe.trajectory_id != harmful.trajectory_id
    assert harmful.execution_logs[-1].startswith("propose ")


def test_clean_trajectory_has_no_payload(scenarios, agent):
    t = agent.run(scenarios[1], [], follow_injection=False)
    assert "user-content" not in t.untrusted_input
