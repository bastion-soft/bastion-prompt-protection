"""Benchmark Guard against held-out evaluation sets and external baselines.

Examples:
    python -m eval.benchmark --runner bastion-fast
    python -m eval.benchmark --runner bastion-fast \\
        --runner protectai/deberta-v3-base-prompt-injection-v2
    python -m eval.benchmark --dataset rogue --limit 500 \\
        --runner bastion-fast
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

from eval.data import EvalSet, load_rogue_benchmark, load_xtram1_test
from eval.metrics import binary_metrics
from eval.runners import BastionRunner, Runner, RunnerOutput, TransformersRunner

logger = logging.getLogger(__name__)

DATASETS = {
    "rogue": load_rogue_benchmark,
    "xtram1": load_xtram1_test,
}


@dataclass
class BenchmarkRow:
    runner: str
    dataset: str
    n_samples: int
    n_attack: int
    n_benign: int
    auc: float
    f1: float
    precision: float
    recall: float
    fpr_at_tpr_99: float
    fpr_at_tpr_95: float
    threshold_at_tpr_99: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_seconds: float


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    eval_sets = [DATASETS[name](limit=args.limit) for name in args.dataset]
    runners = []
    for spec in args.runner:
        try:
            runners.append(_build_runner(spec))
        except Exception as exc:
            logger.warning("could not load runner %s: %s", spec, exc)

    rows: list[BenchmarkRow] = []
    for runner in runners:
        if runner is None:
            continue
        for eval_set in eval_sets:
            try:
                row = _run(runner, eval_set, threshold=args.threshold)
                rows.append(row)
                _print_row(row)
            except Exception as exc:
                logger.warning("runner %s failed on %s: %s", runner.name, eval_set.name, exc)

    _print_leaderboard(rows)

    if args.output:
        _save(rows, Path(args.output))

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eval.benchmark")
    p.add_argument(
        "--runner",
        action="append",
        default=[],
        help=(
            "Runner spec. 'bastion-fast' / 'bastion-prompt-protection-accurate' for the local "
            "Guard, otherwise treated as a HuggingFace model id. May be repeated."
        ),
    )
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        choices=list(DATASETS),
        help="Eval dataset. May be repeated. Default: ['rogue'].",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap samples per dataset for quick smoke runs.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for F1/precision/recall.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write results JSON. Default: eval/results/<timestamp>.json.",
    )
    args = p.parse_args()

    if not args.runner:
        args.runner = ["bastion-fast"]
    if not args.dataset:
        args.dataset = ["rogue"]
    if not args.output:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.output = f"eval/results/{ts}.json"

    return args


def _build_runner(spec: str) -> Runner:
    if spec.startswith("bastion-prompt-protection-"):
        preset = spec.removeprefix("bastion-prompt-protection-")
        return BastionRunner(preset=preset, name=spec)
    return TransformersRunner(model_id=spec)


def _run(runner: Runner, eval_set: EvalSet, threshold: float) -> BenchmarkRow:
    logger.info(
        "running %s on %s (n=%d, attack=%d, benign=%d)",
        runner.name,
        eval_set.name,
        len(eval_set),
        eval_set.n_attack,
        eval_set.n_benign,
    )
    t0 = time.perf_counter()
    output: RunnerOutput = runner.score_batch(eval_set.texts)
    total = time.perf_counter() - t0

    scores = np.asarray(output.scores, dtype=float)
    labels = np.asarray(eval_set.labels, dtype=int)
    metrics = binary_metrics(scores, labels, threshold=threshold)

    p50 = statistics.median(output.latencies_ms) if output.latencies_ms else 0.0
    p95 = (
        statistics.quantiles(output.latencies_ms, n=20)[-1]
        if len(output.latencies_ms) >= 20
        else max(output.latencies_ms, default=0.0)
    )

    return BenchmarkRow(
        runner=runner.name,
        dataset=eval_set.name,
        n_samples=len(eval_set),
        n_attack=eval_set.n_attack,
        n_benign=eval_set.n_benign,
        auc=metrics.auc,
        f1=metrics.f1,
        precision=metrics.precision,
        recall=metrics.recall,
        fpr_at_tpr_99=metrics.fpr_at_tpr_99,
        fpr_at_tpr_95=metrics.fpr_at_tpr_95,
        threshold_at_tpr_99=metrics.threshold_at_tpr_99,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        total_seconds=total,
    )


def _print_row(row: BenchmarkRow) -> None:
    logger.info(
        "  AUC=%.3f F1=%.3f P=%.3f R=%.3f FPR@99TPR=%.3f p50=%.2fms p95=%.2fms total=%.1fs",
        row.auc,
        row.f1,
        row.precision,
        row.recall,
        row.fpr_at_tpr_99,
        row.p50_latency_ms,
        row.p95_latency_ms,
        row.total_seconds,
    )


def _print_leaderboard(rows: list[BenchmarkRow]) -> None:
    try:
        from tabulate import tabulate  # type: ignore[import-not-found]
    except ImportError:
        logger.info("(install tabulate for a prettier leaderboard)")
        for r in rows:
            _print_row(r)
        return

    headers = [
        "runner",
        "dataset",
        "n",
        "AUC",
        "F1",
        "P",
        "R",
        "FPR@99TPR",
        "p50 ms",
        "p95 ms",
    ]
    table = [
        [
            r.runner,
            _short(r.dataset),
            r.n_samples,
            f"{r.auc:.3f}",
            f"{r.f1:.3f}",
            f"{r.precision:.3f}",
            f"{r.recall:.3f}",
            f"{r.fpr_at_tpr_99:.3f}",
            f"{r.p50_latency_ms:.2f}",
            f"{r.p95_latency_ms:.2f}",
        ]
        for r in rows
    ]
    print()
    print(tabulate(table, headers=headers, tablefmt="github"))


def _short(name: str) -> str:
    return name.split("/")[-1] if "/" in name else name


def _save(rows: list[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", path)


if __name__ == "__main__":
    sys.exit(main())
