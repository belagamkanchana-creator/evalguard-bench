import json
import math
import os

import pytest

from evalguard.benchmark import defense_configs, load_config, run_benchmark, save_json, split_scenarios
from evalguard.core.judge import LLMJudge, MockBackend
from evalguard.core.mock_judge import CUES, _split_sections, odd_tokens
from evalguard.defenses import BoundaryReminderDefense, InstructionHardeningDefense
from evalguard.metrics.evaluator import (
    PairRecord,
    RunRecord,
    attack_success_rate,
    auc_roc,
    defense_efficiency_rating,
    false_positive_rate,
    unparseable_rate,
)
from evalguard.metrics.report import MOCK_WARNING, write_markdown_report


def rec(harmful, decision, parse_ok=True, blocked_by=None):
    return RunRecord(attack="a", defense_config="none", scenario_id="s", trajectory_id="t", split="test",
                     is_harmful=harmful, decision=decision, score=4.0 if decision == "APPROVE" else 1.0,
                     parse_ok=parse_ok, blocked_by=blocked_by)


RECORDS = [rec(True, "APPROVE"), rec(True, "REJECT"), rec(True, "REJECT", parse_ok=False), rec(True, "REJECT"),
           rec(False, "APPROVE"), rec(False, "REJECT", blocked_by="perplexity")]


def test_rates():
    assert attack_success_rate(RECORDS) == 25.0
    assert false_positive_rate(RECORDS) == 50.0
    assert unparseable_rate(RECORDS) == 20.0  # 1 of the 5 records that reached the judge
    assert math.isnan(attack_success_rate([rec(False, "APPROVE")]))


def test_der():
    assert defense_efficiency_rating(80.0, 20.0) == 75.0
    assert defense_efficiency_rating(40.0, 100.0) == -150.0
    assert math.isnan(defense_efficiency_rating(0.0, 0.0))


def test_auc():
    assert auc_roc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auc_roc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert auc_roc([0.5, 0.5], [1, 0]) == 0.5
    assert math.isnan(auc_roc([0.5], [1]))


def test_pair_record_flags():
    p = PairRecord(attack="a", defense_config="none", scenario_id="s", safe_id="S", harmful_id="H",
                   first_pass_winner="A", swapped_pass_winner="A", consistent=False, final_choice_id=None)
    assert p.positional_inconsistency and not p.harmful_preferred and not p.safe_preferred


def test_mock_sees_no_odd_tokens_or_cues_in_clean_content(scenarios, agent):
    for defenses in ([], [BoundaryReminderDefense()], [InstructionHardeningDefense()]):
        judge = LLMJudge(MockBackend(0), defenses)
        for s in scenarios:
            _, user = judge.build_absolute_prompt(agent.run(s, [], True))
            untrusted, _ = _split_sections(user)
            assert odd_tokens(untrusted) == [], (s.scenario_id, defenses)
            if not defenses:
                assert not any(c.pattern.search(untrusted) for c in CUES), s.scenario_id


def test_mock_rejects_clean_harmful_and_approves_clean_safe_under_every_defense(scenarios, agent):
    for defenses in ([], [BoundaryReminderDefense()], [InstructionHardeningDefense()],
                     [BoundaryReminderDefense(), InstructionHardeningDefense()]):
        judge = LLMJudge(MockBackend(3), defenses)
        for s in scenarios:
            assert judge.evaluate(agent.run(s, [], True)).decision == "REJECT"
            safe_out = judge.evaluate(agent.run(s, [], False))
            assert safe_out.decision == "APPROVE" or not safe_out.parse_ok


def test_split_and_configs(scenarios):
    train, test = split_scenarios(scenarios, 0.375, seed=7)
    assert len(train) == 3 and len(test) == 5
    assert not {s.scenario_id for s in train} & {s.scenario_id for s in test}
    assert defense_configs(["boundary", "hardening"]) == ["none", "boundary", "hardening", "boundary+hardening"]
    assert defense_configs([]) == ["none"]


def test_end_to_end_mock_run(tmp_path):
    cfg = load_config("configs/default.yaml")
    cfg["universal_suffix"]["max_queries"] = 120
    result = run_benchmark("mock", ["ipi", "universal_suffix", "justification_manip"],
                           ["boundary", "hardening", "perplexity"], cfg, max_scenarios=5)
    assert result.meta["mock"] and result.meta["n_scenarios"] == 5
    assert set(result.summary["defense_config"]) == {"none", "boundary", "hardening", "perplexity",
                                                     "boundary+hardening+perplexity"}
    assert result.suffix["queries_used"] <= 120
    assert "universal_suffix" in result.perplexity["auc_by_attack"]
    suffix_rows = [r for r in result.records if r.attack == "universal_suffix"]
    assert suffix_rows and all(r.split == "test" for r in suffix_rows)  # suffix evaluated on held-out only

    path = save_json(result, tmp_path / "run.json")
    data = json.loads(path.read_text())  # strict JSON: NaN replaced by null
    assert data["meta"]["warning"] == MOCK_WARNING
    report = write_markdown_report(tmp_path / "r.md", result.meta, result.summary, result.der,
                                   result.pairwise_summary, result.perplexity, result.suffix, None)
    assert MOCK_WARNING in report.read_text()


@pytest.mark.skipif(os.environ.get("EVALGUARD_HF_TESTS") != "1", reason="downloads distilgpt2 from Hugging Face")
def test_hf_perplexity_scorer_live():  # pragma: no cover
    from evalguard.defenses import HFPerplexityScorer

    scorer = HFPerplexityScorer("distilgpt2")
    assert scorer.score("~~ ]]\\ {{{ %;] @@ ^^ |> zxQ ]{") > scorer.score("The meeting is on Thursday at two.")
