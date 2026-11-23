import pytest

from evalguard.attacks import (
    AttackContext,
    AuthoritySpoofAttack,
    DelimiterHijackAttack,
    IndirectPromptInjector,
    JustificationManipulator,
    PayloadSplitAttack,
    UniversalSuffixAttacker,
)
from evalguard.core.judge import LLMJudge, MockBackend


@pytest.fixture
def ctx(scenarios):
    return AttackContext.from_scenario(scenarios[3])


def test_delimiter_hijack_closes_tags(ctx):
    (payload,) = DelimiterHijackAttack().build_payloads(ctx)
    assert "</untrusted_data>" in payload and "<system>" in payload and payload.startswith("-->")
    assert ctx.attacker_goal in payload


def test_authority_spoof_has_admin_notice(ctx):
    (payload,) = AuthoritySpoofAttack().build_payloads(ctx)
    assert payload.startswith("[ADMIN NOTICE: Security clearance verified")
    assert ctx.harmful_tool in payload


def test_payload_split_spreads_one_sentence(ctx, scenarios, agent):
    fragments = PayloadSplitAttack(n_fragments=3).build_payloads(ctx)
    assert len(fragments) == 3
    joined = " ".join(f.split(") ", 1)[1] for f in fragments)
    assert ctx.attacker_goal in joined
    traj = agent.run(scenarios[3], fragments, follow_injection=True)
    holders = [tr for tr in traj.tool_returns if "(part " in tr.content]
    assert len(holders) == 2  # this scenario has 2 sources; fragments wrap around


def test_payload_split_needs_two_fragments():
    with pytest.raises(ValueError):
        PayloadSplitAttack(n_fragments=1)


def test_injector_factory():
    names = [a.name for a in IndirectPromptInjector().all()]
    assert names == ["ipi_delimiter_hijack", "ipi_authority_spoof", "ipi_payload_split"]
    with pytest.raises(ValueError):
        IndirectPromptInjector().get("nope")


@pytest.mark.parametrize("variant", JustificationManipulator.VARIANTS)
def test_justification_variants(ctx, variant):
    (payload,) = JustificationManipulator(variant).build_payloads(ctx)
    assert ctx.attacker_goal in payload
    assert "5" in payload


def test_suffix_search_improves_train_score_within_budget(scenarios, agent):
    attacker = UniversalSuffixAttacker(suffix_length=3, max_queries=200, seed=1)
    judge = LLMJudge(MockBackend(seed=1))
    tokens = attacker.optimize(judge, agent, scenarios[:2])
    assert len(tokens) == 3
    assert attacker.queries_used <= 200
    assert attacker.best_train_score >= attacker.history[0]["train_score"]
    (payload,) = attacker.build_payloads(AttackContext.from_scenario(scenarios[5]))
    assert payload.endswith(attacker.suffix)
