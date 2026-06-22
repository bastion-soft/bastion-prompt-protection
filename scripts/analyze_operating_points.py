"""Threshold-agnostic operating-point analysis from dumped scores.

Reads the per-prompt scores written by ``run_leaderboard --dump-scores`` (attack
sets, labelled) and ``measure_false_positives --dump-scores`` (real benign
traffic) and computes, per detector:

  * FPR at a fixed detection rate (TPR=0.95 / 0.99) on **real benign traffic** —
    the apples-to-apples comparison that doesn't depend on where anyone's 0.5
    falls: hold the catch rate constant, compare the false-alarm cost.
  * Equal-error rate (EER) and AUC (detection basis; AUC cross-checks the
    leaderboard).
  * A threshold sweep (FPR on real benign + recall on attacks) at chosen
    thresholds — the 0.45/0.5/0.55 view, free once scores are loaded.
  * A "deployment curve": TPR-on-attacks vs FPR-on-real-benign across thresholds,
    for plotting (the false-positive-axis companion to the detection ROC).

No model inference — pure post-processing, reproducible offline from the
committed score files.

    python -m scripts.analyze_operating_points
    python -m scripts.analyze_operating_points --tprs 0.95,0.99 --thresholds 0.45,0.5,0.55
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from eval.metrics import eer, fpr_at_threshold, threshold_at_tpr
from eval.scores_io import load_all_scores


def _pool(records: list[dict], runner: str, kind: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Concatenate all (scores, labels) for one runner and kind."""
    scores: list[float] = []
    labels: list[int] = []
    have_labels = False
    for rec in records:
        if rec["runner"] != runner or rec["kind"] != kind:
            continue
        scores.extend(rec["scores"])
        if rec.get("labels") is not None:
            labels.extend(rec["labels"])
            have_labels = True
    s = np.asarray(scores, dtype=float)
    return s, (np.asarray(labels, dtype=int) if have_labels else None)


def _analyze_runner(
    attack_scores: np.ndarray,
    attack_labels: np.ndarray,
    benign_scores: np.ndarray,
    tprs: list[float],
    thresholds: list[float],
) -> dict:
    from sklearn.metrics import roc_auc_score  # type: ignore[import-not-found]

    pos = attack_scores[attack_labels == 1]  # attack-only scores → recall

    fpr_at_tpr = {}
    for tpr in tprs:
        t = threshold_at_tpr(attack_scores, attack_labels, tpr)
        fpr_at_tpr[f"{tpr:g}"] = {
            "threshold": round(t, 5),
            "fpr_real_benign": round(fpr_at_threshold(benign_scores, t), 5),
        }

    sweep = {}
    for thr in thresholds:
        sweep[f"{thr:g}"] = {
            "fpr_real_benign": round(fpr_at_threshold(benign_scores, thr), 5),
            "recall_attacks": round(float(np.mean(pos >= thr)) if pos.size else 0.0, 5),
        }

    eer_val, eer_t = eer(attack_scores, attack_labels)

    # Deployment curve: TPR(attacks) vs FPR(real benign) swept over thresholds.
    grid = np.linspace(0.0, 1.0, 201)
    curve = {
        "threshold": [round(float(t), 4) for t in grid],
        "tpr_attacks": [round(float(np.mean(pos >= t)) if pos.size else 0.0, 5) for t in grid],
        "fpr_real_benign": [round(float(np.mean(benign_scores >= t)), 5) for t in grid],
    }

    return {
        "auc": round(float(roc_auc_score(attack_labels, attack_scores)), 5),
        "eer": round(eer_val, 5),
        "eer_threshold": round(eer_t, 5),
        "n_attack": int((attack_labels == 1).sum()),
        "n_benign_real": int(benign_scores.size),
        "fpr_at_tpr": fpr_at_tpr,
        "threshold_sweep": sweep,
        "deployment_curve": curve,
    }


def _format_markdown(results: dict, tprs: list[float], thresholds: list[float]) -> str:
    rows = results["rows"]
    primary = f"{tprs[0]:g}"
    # Rank by FPR at the primary detection rate (lower = better), missing last.
    order = sorted(
        rows,
        key=lambda r: r["fpr_at_tpr"].get(primary, {}).get("fpr_real_benign", 1e9),
    )

    lines: list[str] = []
    lines.append("## Operating points — false positives at a fixed detection rate")
    lines.append("")
    lines.append(
        "Each detector's threshold is set to catch the same share of attacks; we then "
        "report how much **real benign traffic** (WildChat + LMSYS) it flags at that catch "
        "rate. This holds detection constant and shows the false-alarm cost, so the "
        "comparison does not depend on where any detector's 0.5 happens to fall."
    )
    lines.append("")
    tpr_cols = " | ".join(f"FPR @ {int(float(t) * 100)}% catch" for t in tprs)
    lines.append(f"| Detector | AUC | EER | {tpr_cols} |")
    lines.append("|" + "---|" * (len(tprs) + 3))
    for r in order:
        cells = [
            f"{r['fpr_at_tpr'].get(f'{t:g}', {}).get('fpr_real_benign', float('nan')) * 100:.2f}%"
            for t in tprs
        ]
        lines.append(
            f"| {r['runner']} | {r['auc']:.3f} | {r['eer'] * 100:.1f}% | "
            + " | ".join(cells)
            + " |"
        )

    lines.append("")
    lines.append("## Threshold sweep — FPR on real benign traffic (lower = better)")
    lines.append("")
    thr_cols = " | ".join(f"{t:g}" for t in thresholds)
    lines.append(f"| Detector | {thr_cols} |")
    lines.append("|" + "---|" * (len(thresholds) + 1))
    for r in order:
        cells = [
            f"{r['threshold_sweep'].get(f'{t:g}', {}).get('fpr_real_benign', float('nan')) * 100:.2f}%"
            for t in thresholds
        ]
        lines.append(f"| {r['runner']} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Threshold sweep — recall on attacks (higher = better)")
    lines.append("")
    lines.append(f"| Detector | {thr_cols} |")
    lines.append("|" + "---|" * (len(thresholds) + 1))
    for r in order:
        cells = [
            f"{r['threshold_sweep'].get(f'{t:g}', {}).get('recall_attacks', float('nan')) * 100:.1f}%"
            for t in thresholds
        ]
        lines.append(f"| {r['runner']} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        f"_Generated {time.strftime('%Y-%m-%d')} via `python -m scripts.analyze_operating_points` "
        "from dumped per-prompt scores. No model inference; reproducible offline._"
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    tprs = [float(x) for x in args.tprs.split(",") if x.strip()]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    records = load_all_scores(args.scores_dir)
    if not records:
        print(
            f"no score files in {args.scores_dir} — run with --dump-scores first", file=sys.stderr
        )
        return 1

    runners = sorted({rec["runner"] for rec in records})
    rows = []
    curves = {}
    for runner in runners:
        attack_scores, attack_labels = _pool(records, runner, "attack")
        benign_scores, _ = _pool(records, runner, "benign")
        if attack_labels is None or attack_scores.size == 0:
            print(f"skip {runner}: no labelled attack scores", file=sys.stderr)
            continue
        if benign_scores.size == 0:
            print(f"skip {runner}: no real-benign scores", file=sys.stderr)
            continue
        res = _analyze_runner(attack_scores, attack_labels, benign_scores, tprs, thresholds)
        res["runner"] = runner
        curves[runner] = res.pop("deployment_curve")
        rows.append(res)

    if not rows:
        print("no runner had both attack and benign scores", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tprs": tprs,
        "thresholds": thresholds,
        "rows": rows,
    }
    (out_dir / "operating_points.json").write_text(json.dumps(results, indent=2))
    (out_dir / "operating_points.md").write_text(_format_markdown(results, tprs, thresholds))
    (out_dir / "det_points.json").write_text(
        json.dumps(
            {"schema_version": 1, "generated_at": results["generated_at"], "curves": curves},
            indent=2,
        )
    )
    print(f"✓ wrote operating_points.{{json,md}} and det_points.json to {out_dir}")
    print("\n" + (out_dir / "operating_points.md").read_text())
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scripts.analyze_operating_points")
    p.add_argument("--scores-dir", default="eval/results/scores")
    p.add_argument("--output-dir", default="eval/results")
    p.add_argument("--tprs", default="0.95,0.99", help="fixed detection rates (comma-separated)")
    p.add_argument(
        "--thresholds",
        default="0.2,0.45,0.5,0.55,0.8",
        help="sweep thresholds for the discrete table (comma-separated); "
        "the deployment curve always covers the full 0→1 range",
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())
