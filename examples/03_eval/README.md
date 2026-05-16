# Pattern 3 — verify accuracy yourself

The "don't trust our numbers, run the suite" path. Reproduces the full
leaderboard table from the repo README using only this repo's source —
no extras, no hidden test set.

**Use this when:**

- You want to verify our published AUC / F1 numbers against the model
  you would actually deploy.
- You are evaluating bastion against a competitor and want a fair,
  reproducible comparison.
- You are running an internal benchmark and want a baseline.

## Prerequisites

```bash
pip install -e ".[eval]"
```

This installs the SDK plus `torch`, `transformers`, `datasets`,
`scikit-learn`, `tabulate`, and `tqdm` — needed to load the public
baselines (`protectai`, `deepset`, `hlyn-labs`, Meta Prompt-Guard) and
score them.

## Run the full leaderboard

```bash
python -m scripts.run_leaderboard
```

Wall-clock: ~10 minutes on a GPU, ~30 minutes on CPU. Writes the result
to:

- `eval/results/leaderboard.json` — raw numbers (every metric, every
  cell)
- `eval/results/leaderboard.md` — the markdown tables that appear in
  the repo README

## Run against one model

If you only want to score the bastion model itself (or a single
baseline) on all four benchmarks:

```bash
python -m eval.benchmark_suite \
    --runner bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1
```

For a single benchmark, single model:

```bash
python -m eval.benchmark \
    --runner bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1 \
    --dataset rogue
```

## What gets benchmarked

| Benchmark | Source | n | Notes |
|---|---|---|---|
| `rogue` | `rogue-security/prompt-injections-benchmark` | 5,000 | Long, narrative-wrapped attacks |
| `xtram1_test` | `xTRam1/safe-guard-prompt-injection` test split | 2,060 | Standard injection, held out from training |
| `slabs_test` | `S-Labs/prompt-injection-dataset` test split | 2,101 | Security-lab curated, held out from training |
| `jailbreakbench` | `JailbreakBench/JBB-Behaviors` | 200 | Harmful-behavior elicitation |

All four were excluded from the bastion training corpus.

## Headline metrics

| Metric | Why it matters |
|---|---|
| AUC | Threshold-independent ranking quality |
| F1 @ 0.5 | Practical default-threshold quality |
| FPR @ TPR=0.99 | False-positive rate at high recall — the deployment-critical number |
| p50 / p95 latency | Adoption gate; bastion ships sub-10 ms p50 |

## What you should see

If you re-run the leaderboard against the published model, you should
get numbers within ±0.005 of the committed
[`eval/results/leaderboard.json`](../../eval/results/leaderboard.json).
Small variance is normal — model loading and CPU thread scheduling jitter
individual latency cells slightly, but AUC and F1 are deterministic to
three decimals.

If your numbers differ by more than that, something is wrong (wrong
model id, different attack-label index for a baseline, etc.) and we
want to hear about it. Open an issue.

## See also

- [`eval/README.md`](../../eval/README.md) — the eval harness layout
- [`scripts/run_leaderboard.py`](../../scripts/run_leaderboard.py) — the
  leaderboard runner

## When to use this vs another pattern

- **Pattern 1 (raw ONNX)** if you want to verify *individual predictions*
  rather than aggregate benchmark numbers.
- **Pattern 2 (SDK)** for building applications, not just measuring.
- **Pattern 4 (Docker)** for production deployment after you have
  convinced yourself the numbers are real.
