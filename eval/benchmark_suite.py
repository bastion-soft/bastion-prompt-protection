"""Multi-benchmark evaluation suite.

Runs one model (or several) across multiple held-out benchmarks. Prevents
"leaderboard hacking" against a single benchmark — if you only beat rogue
and lose on JailbreakBench / xTRam1-test, you didn't generalize.

Output is a single leaderboard with rows per (runner, benchmark) showing
AUC, F1, P, R, FPR@99TPR, p50/p95 latency.

Usage:
    # Default: run bastion-fast against all benchmarks
    python -m eval.benchmark_suite

    # Run multiple models for comparison
    python -m eval.benchmark_suite \\
        --runner bastion-fast \\
        --runner protectai/deberta-v3-base-prompt-injection-v2

    # Pick specific benchmarks
    python -m eval.benchmark_suite --benchmark rogue --benchmark jailbreakbench

    # Use a local model directory (e.g. just-trained Stage 3 output)
    python -m eval.benchmark_suite --runner local:/path/to/stage3/final
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from eval.data import BENCHMARK_LOADERS, EvalSet
from eval.metrics import binary_metrics
from eval.runners import BastionRunner, Runner, RunnerOutput, TransformersRunner

logger = logging.getLogger(__name__)


@dataclass
class SuiteRow:
    runner: str
    benchmark: str
    n_samples: int
    n_attack: int
    n_benign: int
    auc: float
    f1: float
    precision: float
    recall: float
    fpr_at_tpr_99: float
    fpr_at_tpr_95: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_seconds: float


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Build runners
    runners: list[Runner] = []
    for spec in args.runner:
        try:
            runners.append(_build_runner(spec))
        except Exception as exc:
            logger.warning("could not load runner %s: %s", spec, exc)
    if not runners:
        logger.error("no runners loaded")
        return 1

    # Build benchmarks
    bench_names = args.benchmark or list(BENCHMARK_LOADERS)
    benchmarks: list[EvalSet] = []
    for name in bench_names:
        try:
            benchmarks.append(BENCHMARK_LOADERS[name](limit=args.limit))
        except Exception as exc:
            logger.warning("could not load benchmark %s: %s", name, exc)

    rows: list[SuiteRow] = []
    for runner in runners:
        for bench in benchmarks:
            try:
                row = _run(runner, bench, threshold=args.threshold)
                rows.append(row)
                _print_row(row)
            except Exception as exc:
                logger.warning("runner %s failed on %s: %s", runner.name, bench.name, exc)

    _print_leaderboard(rows)

    if args.output:
        _save(rows, Path(args.output))

    return 0


def _build_runner(spec: str) -> Runner:
    if spec.startswith("bastion-prompt-protection-"):
        preset = spec.removeprefix("bastion-prompt-protection-")
        return BastionRunner(preset=preset, name=spec)
    if spec.startswith("local:"):
        path = spec.removeprefix("local:")
        return TransformersRunner(model_id=path, name=f"local:{Path(path).name}")
    return TransformersRunner(model_id=spec)


def _run(runner: Runner, bench: EvalSet, threshold: float) -> SuiteRow:
    logger.info(
        "running %s on %s (n=%d, attack=%d, benign=%d)",
        runner.name,
        bench.name,
        len(bench),
        bench.n_attack,
        bench.n_benign,
    )
    t0 = time.perf_counter()
    output: RunnerOutput = runner.score_batch(bench.texts)
    total = time.perf_counter() - t0

    scores = np.asarray(output.scores, dtype=float)
    labels = np.asarray(bench.labels, dtype=int)
    metrics = binary_metrics(scores, labels, threshold=threshold)

    p50 = statistics.median(output.latencies_ms) if output.latencies_ms else 0.0
    p95 = (
        statistics.quantiles(output.latencies_ms, n=20)[-1]
        if len(output.latencies_ms) >= 20
        else max(output.latencies_ms, default=0.0)
    )

    return SuiteRow(
        runner=runner.name,
        benchmark=bench.name,
        n_samples=len(bench),
        n_attack=bench.n_attack,
        n_benign=bench.n_benign,
        auc=metrics.auc,
        f1=metrics.f1,
        precision=metrics.precision,
        recall=metrics.recall,
        fpr_at_tpr_99=metrics.fpr_at_tpr_99,
        fpr_at_tpr_95=metrics.fpr_at_tpr_95,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        total_seconds=total,
    )


def _print_row(row: SuiteRow) -> None:
    logger.info(
        "  AUC=%.3f F1=%.3f P=%.3f R=%.3f FPR@99=%.3f p50=%.2fms",
        row.auc,
        row.f1,
        row.precision,
        row.recall,
        row.fpr_at_tpr_99,
        row.p50_latency_ms,
    )


def _print_leaderboard(rows: list[SuiteRow]) -> None:
    try:
        from tabulate import tabulate
    except ImportError:
        for r in rows:
            _print_row(r)
        return

    headers = [
        "runner",
        "benchmark",
        "n",
        "AUC",
        "F1",
        "P",
        "R",
        "FPR@99",
        "p50 ms",
    ]
    table = [
        [
            r.runner,
            _short(r.benchmark),
            r.n_samples,
            f"{r.auc:.3f}",
            f"{r.f1:.3f}",
            f"{r.precision:.3f}",
            f"{r.recall:.3f}",
            f"{r.fpr_at_tpr_99:.3f}",
            f"{r.p50_latency_ms:.1f}",
        ]
        for r in rows
    ]
    print()
    print(tabulate(table, headers=headers, tablefmt="github"))

    # Per-runner average across benchmarks — single-number leaderboard.
    print("\n## Average across benchmarks")
    by_runner: dict[str, list[SuiteRow]] = {}
    for r in rows:
        by_runner.setdefault(r.runner, []).append(r)
    avg_table = []
    for runner, runner_rows in by_runner.items():
        avg_table.append(
            [
                runner,
                len(runner_rows),
                f"{statistics.mean(r.auc for r in runner_rows):.3f}",
                f"{statistics.mean(r.f1 for r in runner_rows):.3f}",
                f"{statistics.mean(r.fpr_at_tpr_99 for r in runner_rows):.3f}",
                f"{statistics.mean(r.p50_latency_ms for r in runner_rows):.1f}",
            ]
        )
    print(
        tabulate(
            avg_table,
            headers=["runner", "n_bench", "avg AUC", "avg F1", "avg FPR@99", "avg p50 ms"],
            tablefmt="github",
        )
    )


def _short(name: str) -> str:
    """Shorten benchmark names for table display while keeping them distinct."""
    if "/" not in name:
        return name
    # Keep `org/last` so e.g. "xTRam1/test" stays distinguishable from "S-Labs/test".
    parts = name.split("/")
    if len(parts) >= 2 and parts[-1] in {"test", "validation"}:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def _save(rows: list[SuiteRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eval.benchmark_suite")
    p.add_argument(
        "--runner",
        action="append",
        default=[],
        help="Runner spec. 'bastion-fast' / 'local:<path>' / HuggingFace id. Repeatable.",
    )
    p.add_argument(
        "--benchmark",
        action="append",
        default=[],
        choices=list(BENCHMARK_LOADERS),
        help="Benchmark to run. Default: all. Repeatable.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap samples per benchmark")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output", type=str, default=None)

    args = p.parse_args()
    if not args.runner:
        args.runner = ["bastion-fast"]
    if not args.output:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.output = f"eval/results/suite-{ts}.json"
    return args


if __name__ == "__main__":
    sys.exit(main())
