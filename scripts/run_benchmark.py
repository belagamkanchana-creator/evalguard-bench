#!/usr/bin/env python
"""Run EvalGuard-Bench from the command line.

Examples:
    python scripts/run_benchmark.py --mock
    python scripts/run_benchmark.py --model ollama:llama3.1:8b --attacks ipi --defenses boundary hardening
    python scripts/run_benchmark.py --model openai:gpt-4o-mini --max-scenarios 4 --suffix-budget 300
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from evalguard.benchmark import (  # noqa: E402
    ATTACK_GROUPS,
    DEFENSE_NAMES,
    MODES,
    BenchmarkResult,
    load_config,
    run_benchmark,
    save_json,
)
from evalguard.core.judge import JudgeError  # noqa: E402
from evalguard.metrics.report import MOCK_WARNING, plot_asr_heatmap, write_markdown_report  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EvalGuard-Bench: adversarial robustness of LLM judges.")
    p.add_argument("--model", default=None,
                   help="Judge: 'mock', 'hf:<model>', 'openai:<model>' or 'ollama:<model>' (default: mock)")
    p.add_argument("--mock", action="store_true", help="Use the offline simulated judge (no API key needed)")
    p.add_argument("--attacks", nargs="+", choices=ATTACK_GROUPS, default=list(ATTACK_GROUPS))
    p.add_argument("--defenses", nargs="*", choices=DEFENSE_NAMES, default=list(DEFENSE_NAMES),
                   help="Defenses to evaluate alone and combined; pass none of them for the baseline only")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--max-scenarios", type=int, default=None, help="Use only the first N scenarios (saves API cost)")
    p.add_argument("--suffix-budget", type=int, default=None, help="Max judge queries for the suffix search")
    p.add_argument("--perplexity-scorer", choices=("auto", "char_bigram", "hf"), default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.1f}"
    return str(value)


def print_tables(result: BenchmarkResult) -> None:
    tables: list[tuple[str, pd.DataFrame]] = [
        ("Absolute scoring: ASR / FPR (%)", result.summary),
        ("Defense efficiency rating (%)", result.der),
    ]
    if result.pairwise_summary is not None:
        tables.append(("Pairwise with positional swapping (%)", result.pairwise_summary))
    if result.perplexity:
        tables.append(("Perplexity filter AUC-ROC", pd.DataFrame(
            [{"attack": k, "auc_roc": v} for k, v in result.perplexity["auc_by_attack"].items()])))

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        if result.meta["mock"]:
            print(f"\n*** {MOCK_WARNING} ***\n")
        for title, df in tables:
            print(f"\n{title}\n{df.to_string(index=False)}")
        return

    console = Console()
    if result.meta["mock"]:
        console.print(Panel(MOCK_WARNING, title="MOCK RUN", style="bold yellow"))
    for title, df in tables:
        if df is None or df.empty:
            continue
        table = Table(title=title, show_lines=False)
        for col in df.columns:
            table.add_column(str(col), justify="left" if df[col].dtype == object else "right")
        for row in df.itertuples(index=False):
            table.add_row(*(_fmt(v) for v in row))
        console.print(table)
    if result.suffix:
        console.print(f"Universal suffix: [bold]{result.suffix['suffix']}[/bold] "
                      f"(train score {result.suffix['start_train_score']:.2f} -> {result.suffix['best_train_score']:.2f}, "
                      f"{result.suffix['queries_used']} queries)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("run_benchmark")

    if args.mock or args.model in (None, "mock"):
        if not args.mock and args.model is None:
            log.info("No --model given: running with the offline simulated judge (mock mode)")
        model_spec = "mock"
    else:
        model_spec = args.model

    try:
        cfg = load_config(args.config)
        if args.seed is not None:
            cfg["seed"] = args.seed
        if args.suffix_budget is not None:
            cfg.setdefault("universal_suffix", {})["max_queries"] = args.suffix_budget
        result = run_benchmark(model_spec, args.attacks, args.defenses, cfg, modes=args.modes,
                               max_scenarios=args.max_scenarios, perplexity_scorer=args.perplexity_scorer)
    except (JudgeError, ValueError, FileNotFoundError, RuntimeError) as e:
        log.error("%s", e)
        return 2

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    print_tables(result)
    json_path = save_json(result, out_dir / "benchmark_run.json")
    title = "ASR by attack and defense" + (" (MOCK - simulated judge)" if result.meta["mock"] else f" ({model_spec})")
    heatmap = None
    if not result.summary.empty:
        heatmap = plot_asr_heatmap(result.summary, out_dir / "asr_heatmap.png", title,
                                   row_order=["none", *result.meta["attacks"]],
                                   col_order=result.meta["defense_configs"])
    report = write_markdown_report(
        out_dir / "benchmark_report.md", result.meta, result.summary, result.der, result.pairwise_summary,
        result.perplexity, result.suffix, heatmap.name if heatmap else None,
    )
    log.info("Saved %s", json_path)
    log.info("Saved %s", report)
    if heatmap:
        log.info("Saved %s", heatmap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
