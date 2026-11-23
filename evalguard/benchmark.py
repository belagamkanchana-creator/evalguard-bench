"""The benchmark pipeline: build trajectories, run attacks and defenses, compute metrics, save outputs."""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from evalguard import __version__
from evalguard.attacks import (
    Attack,
    AttackContext,
    IndirectPromptInjector,
    JustificationManipulator,
    UniversalSuffixAttacker,
)
from evalguard.core.agent import SimulatedAgent, load_scenarios
from evalguard.core.judge import JudgeSettings, LLMJudge, make_backend
from evalguard.core.models import AgentTrajectory, Scenario
from evalguard.defenses import (
    BoundaryReminderDefense,
    CharBigramScorer,
    Defense,
    HFPerplexityScorer,
    InstructionHardeningDefense,
    PerplexityFilter,
    PerplexityScorer,
)
from evalguard.metrics.evaluator import (
    PairRecord,
    RunRecord,
    auc_roc,
    der_table,
    summarize_absolute,
    summarize_pairwise,
)
from evalguard.metrics.report import MOCK_WARNING

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTACK_GROUPS = ("ipi", "universal_suffix", "justification_manip")
DEFENSE_NAMES = ("boundary", "hardening", "perplexity")
MODES = ("absolute", "pairwise")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        cfg = yaml.safe_load(path.read_text()) or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {path}") from e
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return cfg


def expand_attacks(groups: list[str], cfg: dict[str, Any], seed: int) -> list[Attack]:
    attacks: list[Attack] = []
    for group in groups:
        if group == "ipi":
            attacks += IndirectPromptInjector(cfg.get("attacks", {}).get("payload_split_fragments", 3)).all()
        elif group == "justification_manip":
            attacks += [JustificationManipulator(v) for v in JustificationManipulator.VARIANTS]
        elif group == "universal_suffix":
            us = cfg.get("universal_suffix", {})
            attacks.append(UniversalSuffixAttacker(
                suffix_length=us.get("suffix_length", 6), max_queries=us.get("max_queries", 3000),
                n_rounds=us.get("n_rounds", 1), seed=seed,
            ))
        else:
            raise ValueError(f"Unknown attack group {group!r}; choose from {ATTACK_GROUPS}")
    return attacks


def defense_configs(selected: list[str]) -> list[str]:
    """The undefended baseline, each selected defense alone, and all of them combined."""
    configs = ["none", *selected]
    if len(selected) > 1:
        configs.append("+".join(selected))
    return configs


def split_scenarios(scenarios: list[Scenario], train_fraction: float, seed: int) -> tuple[list[Scenario], list[Scenario]]:
    if len(scenarios) < 2:
        raise ValueError("Need at least 2 scenarios: one to search the suffix on and one to test it on")
    order = scenarios[:]
    random.Random(seed).shuffle(order)
    n_train = min(len(order) - 1, max(1, round(train_fraction * len(order))))
    return order[:n_train], order[n_train:]


def make_scorer(ppl_cfg: dict[str, Any], mock: bool, override: str | None) -> PerplexityScorer:
    kind = override or ppl_cfg.get("scorer", "auto")
    if kind == "auto":
        kind = "char_bigram" if mock else "hf"
    mode = ppl_cfg.get("mode", "windowed_max")
    if kind == "char_bigram":
        return CharBigramScorer(window=ppl_cfg.get("char_window", 48), mode=mode)
    if kind == "hf":
        return HFPerplexityScorer(ppl_cfg.get("hf_model", "distilgpt2"),
                                  window_tokens=ppl_cfg.get("hf_window_tokens", 32), mode=mode)
    raise ValueError(f"Unknown perplexity scorer {kind!r}; use auto, char_bigram or hf")


@dataclass
class TrajectoryPair:
    attack: str
    scenario: Scenario
    safe: AgentTrajectory
    harmful: AgentTrajectory


@dataclass
class BenchmarkResult:
    meta: dict[str, Any]
    records: list[RunRecord]
    pairs: list[PairRecord]
    summary: pd.DataFrame
    der: pd.DataFrame
    pairwise_summary: pd.DataFrame | None
    perplexity: dict[str, Any] | None
    suffix: dict[str, Any] | None
    config: dict[str, Any] = field(default_factory=dict)


def run_benchmark(
    model_spec: str,
    attack_groups: list[str],
    defense_names: list[str],
    cfg: dict[str, Any],
    modes: list[str] | None = None,
    max_scenarios: int | None = None,
    perplexity_scorer: str | None = None,
) -> BenchmarkResult:
    modes = modes or list(MODES)
    seed = int(cfg.get("seed", 0))
    mock = model_spec == "mock"
    settings = JudgeSettings.from_dict({**cfg.get("judge", {}), "seed": seed})
    backend = make_backend(model_spec, settings)

    scenarios_path = Path(cfg.get("scenarios_path", "data/scenarios.yaml"))
    if not scenarios_path.is_absolute():
        scenarios_path = REPO_ROOT / scenarios_path
    scenarios = load_scenarios(scenarios_path)
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]
    train, test = split_scenarios(scenarios, cfg.get("universal_suffix", {}).get("train_fraction", 0.375), seed)
    split_of = {s.scenario_id: "train" for s in train} | {s.scenario_id: "test" for s in test}
    agent = SimulatedAgent()
    attacks = expand_attacks(attack_groups, cfg, seed)

    # 1. Universal suffix search against the undefended judge, on training scenarios only.
    suffix_info: dict[str, Any] | None = None
    for attack in attacks:
        if isinstance(attack, UniversalSuffixAttacker):
            logger.info("Searching for a universal suffix on %d training scenarios", len(train))
            attack.optimize(LLMJudge(backend, [], settings), agent, train)
            suffix_info = {
                "suffix": attack.suffix, "tokens": attack.suffix_tokens, "queries_used": attack.queries_used,
                "start_train_score": attack.history[0]["train_score"], "best_train_score": attack.best_train_score,
                "train_scenarios": [s.scenario_id for s in train], "history": attack.history,
            }

    # 2. Trajectory pairs (safe + harmful) for the clean baseline and every attack.
    pairs: list[TrajectoryPair] = []
    for s in scenarios:
        split = split_of[s.scenario_id]
        pairs.append(TrajectoryPair("none", s, agent.run(s, [], False, "none", split), agent.run(s, [], True, "none", split)))
    for attack in attacks:
        eval_set = test if isinstance(attack, UniversalSuffixAttacker) else scenarios
        for s in eval_set:
            payloads = attack.build_payloads(AttackContext.from_scenario(s))
            split = split_of[s.scenario_id]
            pairs.append(TrajectoryPair(
                attack.name, s,
                agent.run(s, payloads, False, attack.name, split),
                agent.run(s, payloads, True, attack.name, split),
            ))

    # 3. Defenses (built once; the perplexity filter is calibrated on clean training content).
    ppl_cfg = cfg.get("perplexity", {})
    available: dict[str, Defense] = {"boundary": BoundaryReminderDefense(), "hardening": InstructionHardeningDefense()}
    ppl_filter: PerplexityFilter | None = None
    if "perplexity" in defense_names:
        ppl_filter = PerplexityFilter(
            make_scorer(ppl_cfg, mock, perplexity_scorer), tau=ppl_cfg.get("tau"),
            calibrate_percentile=ppl_cfg.get("calibrate_percentile", 95), margin=ppl_cfg.get("margin", 1.05),
        )
        tau_given = ppl_filter.tau is not None
        ppl_filter.fit([tr.content for s in train for tr in agent.collect(s, [])])
        available["perplexity"] = ppl_filter

    configs = defense_configs(defense_names)
    records: list[RunRecord] = []
    pair_records: list[PairRecord] = []
    for config in configs:
        defenses = [] if config == "none" else [available[name] for name in config.split("+")]
        judge = LLMJudge(backend, defenses, settings)
        logger.info("Evaluating defense configuration %r (%d trajectory pairs)", config, len(pairs))
        for pair in pairs:
            if "absolute" in modes:
                for traj in (pair.safe, pair.harmful):
                    out = judge.evaluate(traj)
                    records.append(RunRecord(
                        attack=pair.attack, defense_config=config, scenario_id=pair.scenario.scenario_id,
                        trajectory_id=traj.trajectory_id, split=traj.split,
                        is_harmful=traj.proposed_action.is_harmful, decision=out.decision, score=out.score,
                        parse_ok=out.parse_ok, blocked_by=out.blocked_by, reasoning=out.reasoning,
                    ))
            if "pairwise" in modes and cfg.get("pairwise", {}).get("enabled", True):
                po = judge.evaluate_pairwise(pair.safe, pair.harmful)
                pair_records.append(PairRecord(
                    attack=pair.attack, defense_config=config, scenario_id=pair.scenario.scenario_id,
                    safe_id=pair.safe.trajectory_id, harmful_id=pair.harmful.trajectory_id,
                    first_pass_winner=po.first_pass_winner, swapped_pass_winner=po.swapped_pass_winner,
                    consistent=po.consistent, final_choice_id=po.final_choice_id, blocked_by=po.blocked_by,
                ))

    # 4. Perplexity detection quality: attacked external content (1) vs clean external content (0).
    ppl_info: dict[str, Any] | None = None
    if ppl_filter is not None:
        eval_ids = {s.scenario_id for s in test}
        clean = [p.safe.untrusted_input for p in pairs if p.attack == "none" and p.scenario.scenario_id in eval_ids]
        clean_scores = [ppl_filter.score_text(t) for t in clean]
        auc_by_attack: dict[str, float] = {}
        attacked_scores: dict[str, list[float]] = {}
        for attack in attacks:
            texts = [p.harmful.untrusted_input for p in pairs
                     if p.attack == attack.name and p.scenario.scenario_id in eval_ids]
            scores = [ppl_filter.score_text(t) for t in texts]
            attacked_scores[attack.name] = scores
            auc_by_attack[attack.name] = auc_roc(scores + clean_scores, [1] * len(scores) + [0] * len(clean_scores))
        ppl_info = {
            "scorer": ppl_filter.scorer.name, "mode": ppl_cfg.get("mode", "windowed_max"), "tau": ppl_filter.tau,
            "tau_source": "set in config" if tau_given else
            f"calibrated: {ppl_filter.calibrate_percentile:g}th percentile of clean training content x {ppl_filter.margin:g}",
            "auc_by_attack": auc_by_attack, "clean_scores": clean_scores, "attacked_scores": attacked_scores,
        }

    summary = summarize_absolute(records) if records else pd.DataFrame()
    meta = {
        "tool": "EvalGuard-Bench", "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_spec, "mock": mock, "warning": MOCK_WARNING if mock else None, "seed": seed,
        "attacks": [a.name for a in attacks], "defense_configs": configs, "modes": modes,
        "n_scenarios": len(scenarios), "n_train": len(train), "n_test": len(test),
        "train_scenarios": [s.scenario_id for s in train], "test_scenarios": [s.scenario_id for s in test],
    }
    return BenchmarkResult(
        meta=meta, records=records, pairs=pair_records, summary=summary,
        der=der_table(summary) if not summary.empty else pd.DataFrame(),
        pairwise_summary=summarize_pairwise(pair_records) if pair_records else None,
        perplexity=ppl_info, suffix=suffix_info, config=cfg,
    )


def _jsonable(value: Any) -> Any:
    """Replace NaN with None recursively so the output is valid JSON."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def save_json(result: BenchmarkResult, path: Path) -> Path:
    data = {
        "meta": result.meta,
        "config": result.config,
        "summary": result.summary.to_dict(orient="records"),
        "defense_efficiency": result.der.to_dict(orient="records"),
        "pairwise_summary": result.pairwise_summary.to_dict(orient="records") if result.pairwise_summary is not None else None,
        "perplexity": result.perplexity,
        "universal_suffix": result.suffix,
        "records": [r.model_dump() for r in result.records],
        "pairwise_records": [p.model_dump() for p in result.pairs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False))
    return path
