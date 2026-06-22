"""Run bastion-prompt-protection against open baselines on all held-out benchmarks.

Produces a markdown leaderboard suitable for the README and HF model card.

Usage (Colab T4 free tier is fine — total wall-clock ~5-10 min):

    # 1. Install + auth (Meta Prompt-Guard is gated; the rest are public)
    pip install -e ".[eval]"
    huggingface-cli login

    # 2. Run
    python -m scripts.run_leaderboard

    # 3. Outputs
    #    eval/results/leaderboard.json   — raw rows
    #    eval/results/leaderboard.md     — markdown for README / HF
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

from eval.benchmark_suite import SuiteRow, _run
from eval.data import BENCHMARK_LOADERS
from eval.runners import TransformersRunner

logger = logging.getLogger(__name__)


# Commercial models — gated on the HF Hub. The harness still lists them so a
# license holder (with a granted HF token) can score them in the same run; for
# everyone else they fail to download and are skipped with a clear message
# pointing at how to get access. See the skip handler in main().
# (Dormant for now: the multilingual BASELINES entry is commented out while the
# multilingual model is being updated; kept so re-enabling is a one-line change.)
COMMERCIAL_MODELS = {
    "bastionsoft/binary-bastion-prompt-protection-mdeberta-v3-base-v1",
}

# (display_name, hf_model_id, attack_label_id)
# attack_label_id may be int (single attack class) or list[int] (sum across
# multiple attack classes — for multi-class detectors). All competitors are
# binary with attack = LABEL_1; if any returns AUC < 0.5 its polarity is
# reversed — flip that entry's index to 0.
BASELINES: list[tuple[str, str, int | list[int]]] = [
    # --- Ours -------------------------------------------------------------
    # The free, open (AGPL) model — fully reproducible by anyone. This is the
    # one the public benchmark is about.
    (
        "bastion-prompt-protection (70M)",
        "bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1",
        1,
    ),
    # The commercial multilingual model — PARKED while the multilingual model is
    # being updated. Re-enable this entry once the new version ships.
    # (
    #     "bastion multilingual (280M, commercial)",
    #     "bastionsoft/binary-bastion-prompt-protection-mdeberta-v3-base-v1",
    #     1,
    # ),
    # --- Open-source competitors (all public, all reproducible) -----------
    ("wolf-defender (0.3B)", "patronus-studio/wolf-defender-prompt-injection", 1),
    ("wolf-defender-small (0.1B)", "patronus-studio/wolf-defender-prompt-injection-small", 1),
    ("sentinel (qualifire, 395M)", "qualifire/prompt-injection-sentinel", 1),
    ("proventra mdeberta (280M)", "proventra/mdeberta-v3-base-prompt-injection", 1),
    # PIGuard (leolee99/PIGuard) deliberately omitted: it requires
    # trust_remote_code=True to load, and a reproducible security-tool harness
    # should not execute arbitrary remote code from a third-party repo.
    ("fmops distilbert (67M)", "fmops/distilbert-prompt-injection", 1),
    ("protectai v2 (184M)", "protectai/deberta-v3-base-prompt-injection-v2", 1),
    ("deepset injection (184M)", "deepset/deberta-v3-base-injection", 1),
    ("hlyn judge (70M)", "hlyn-labs/prompt-injection-judge-deberta-70m", 1),
    # Meta Prompt-Guard is 3-class (0=BENIGN, 1=INJECTION, 2=JAILBREAK).
    # Both 1 and 2 are "attack" in our binary frame — sum their softmax probs.
    # Requires HF gated-access approval at https://huggingface.co/meta-llama/Prompt-Guard-86M
    ("meta prompt-guard (86M)", "meta-llama/Prompt-Guard-86M", [1, 2]),
]


BENCHMARK_DISPLAY = {
    "rogue": "rogue (5k)",
    "jailbreakbench": "JBB (200)",
    "xtram1_test": "xTRam1 test (2k)",
    "slabs_test": "S-Labs test (2k)",
    "deepset_test": "deepset test (116)",
}


DEFAULT_BENCHMARKS = ["rogue", "jailbreakbench", "xtram1_test", "slabs_test"]


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load benchmarks once, reuse across runners. Track (key, EvalSet) so we
    # can key the output table off the registry key, not EvalSet.name.
    bench_keys = args.benchmark or DEFAULT_BENCHMARKS
    bench_pairs: list[tuple[str, object]] = []
    name_to_key: dict[str, str] = {}
    for key in bench_keys:
        try:
            eval_set = BENCHMARK_LOADERS[key](limit=args.limit)
        except Exception as exc:
            logger.warning("could not load benchmark %s: %s", key, exc)
            continue
        bench_pairs.append((key, eval_set))
        name_to_key[eval_set.name] = key
    if not bench_pairs:
        logger.error("no benchmarks loaded")
        return 1

    rows: list[tuple[str, SuiteRow]] = []  # (bench_key, row)
    for display, model_id, attack_label in BASELINES:
        logger.info("=" * 60)
        logger.info("loading %s (%s)", display, model_id)
        try:
            runner = TransformersRunner(
                model_id=model_id,
                attack_label_id=attack_label,
                max_length=512,
                batch_size=args.batch_size,
                name=display,
            )
        except Exception as exc:
            if model_id in COMMERCIAL_MODELS:
                logger.warning(
                    "skip %s — this is a commercial, gated model. To include it, "
                    "obtain a license and access at https://bastionsoft.com, then "
                    "`huggingface-cli login` with the granted token. (%s)",
                    display,
                    exc,
                )
            else:
                logger.warning("skip %s: %s", display, exc)
            continue
        for key, bench in bench_pairs:
            try:
                rows.append(
                    (key, _run(runner, bench, threshold=args.threshold, dump_dir=args.dump_scores))
                )
            except Exception as exc:
                logger.warning("%s on %s failed: %s", display, bench.name, exc)

    if not rows:
        logger.error("no rows produced")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_order = [k for k, _ in bench_pairs]

    json_path = out_dir / "leaderboard.json"
    _write_json(rows, json_path)
    logger.info("wrote %s", json_path)

    md_path = out_dir / "leaderboard.md"
    md_path.write_text(_format_markdown(rows, bench_order))
    logger.info("wrote %s", md_path)
    print("\n" + md_path.read_text())
    return 0


def _format_markdown(rows: list[tuple[str, SuiteRow]], bench_order: list[str]) -> str:
    by_runner: dict[str, dict[str, SuiteRow]] = {}
    for key, r in rows:
        by_runner.setdefault(r.runner, {})[key] = r

    # Rank models by average AUC (descending), applied consistently to every
    # table so the same model order reads down the AUC, F1 and latency tables.
    def _avg_auc(by_bench: dict[str, SuiteRow]) -> float:
        vals = [row.auc for row in by_bench.values()]
        return statistics.mean(vals) if vals else 0.0

    order = sorted(by_runner, key=lambda name: _avg_auc(by_runner[name]), reverse=True)

    headers = [BENCHMARK_DISPLAY.get(k, k) for k in bench_order]

    lines: list[str] = []
    lines.append("## Leaderboard — AUC")
    lines.append("")
    lines.append("| Model | " + " | ".join(headers) + " | **Avg** |")
    lines.append("|" + "---|" * (len(headers) + 2))
    for runner in order:
        by_bench = by_runner[runner]
        values, aucs = [], []
        for key in bench_order:
            row = by_bench.get(key)
            if row is None:
                values.append("—")
            else:
                values.append(f"{row.auc:.3f}")
                aucs.append(row.auc)
        avg = f"**{statistics.mean(aucs):.3f}**" if aucs else "—"
        lines.append(f"| {runner} | " + " | ".join(values) + f" | {avg} |")

    lines.append("")
    lines.append("## Leaderboard — F1 @ threshold=0.5")
    lines.append("")
    lines.append("| Model | " + " | ".join(headers) + " | **Avg** |")
    lines.append("|" + "---|" * (len(headers) + 2))
    for runner in order:
        by_bench = by_runner[runner]
        values, f1s = [], []
        for key in bench_order:
            row = by_bench.get(key)
            if row is None:
                values.append("—")
            else:
                values.append(f"{row.f1:.3f}")
                f1s.append(row.f1)
        avg = f"**{statistics.mean(f1s):.3f}**" if f1s else "—"
        lines.append(f"| {runner} | " + " | ".join(values) + f" | {avg} |")

    lines.append("")
    lines.append("## Latency (p50 ms / sample, batched inference)")
    lines.append("")
    lines.append("| Model | " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * (len(headers) + 1))
    for runner in order:
        by_bench = by_runner[runner]
        values = []
        for key in bench_order:
            row = by_bench.get(key)
            values.append(f"{row.p50_latency_ms:.1f}" if row else "—")
        lines.append(f"| {runner} | " + " | ".join(values) + " |")

    lines.append("")
    lines.append(
        f"_Generated {time.strftime('%Y-%m-%d')} via `python -m scripts.run_leaderboard`._"
    )
    return "\n".join(lines) + "\n"


def _write_json(rows: list[tuple[str, SuiteRow]], path: Path) -> None:
    from dataclasses import asdict

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": [{"benchmark_key": key, **asdict(r)} for key, r in rows],
    }
    path.write_text(json.dumps(payload, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scripts.run_leaderboard")
    p.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="benchmark(s) to run; repeat for multiple. Default: DEFAULT_BENCHMARKS.",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="cap samples per benchmark (smoke testing)"
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", default="eval/results")
    p.add_argument(
        "--dump-scores",
        default=None,
        metavar="DIR",
        help="also write raw per-prompt scores+labels per (model, benchmark) to DIR "
        "(e.g. eval/results/scores) for offline operating-point analysis.",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())
