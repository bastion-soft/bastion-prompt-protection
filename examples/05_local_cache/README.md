# Advanced — Local model cache (offline / regulated)

Pre-download the model once. Run fully offline thereafter. Use this for air-gapped deployments, GDPR-strict workloads, Docker images built without runtime network access, or any environment that needs deterministic startup with no surprise downloads.

**Use this when:** the runtime cannot reach huggingface.co at request time, or you want to bake the model into a container at build time.

## Prerequisites

```bash
pip install bastion-prompt-protection
```

You need outbound HTTPS for the *initial* download. After that, no network access is required.

## Run

```bash
python examples/02_local_cache/main.py
```

The script does two things:

1. **Option A** — points `Guard` at a project-local cache directory (`./.bastion-cache/`). First call downloads the model there; subsequent calls load from disk only.
2. **Option B** — pre-downloads the model snapshot explicitly via `huggingface_hub.snapshot_download`, then sets `HF_HUB_OFFLINE=1` so any later network call hard-fails. Useful in CI / build steps where silent fallbacks would mask real issues.

## Expected output

```
Model cached under: /your/path/.bastion-cache
  risk=0.970  label=attack  stage=heuristics

Pre-downloading model snapshot...
  ↳ snapshot at: /your/path/.bastion-cache/models--bastionsoft--binary-bastion-prompt-protection-deberta-v3-xsmall-v1/snapshots/<sha>

offline-mode protect():
  risk=0.970  label=attack  stage=heuristics
  ✓ ran fully offline against the local cache
```

The second `protect()` runs after `HF_HUB_OFFLINE=1` is set. If the model weren't fully cached, that call would fail with `HFValidationError: offline mode is enabled`. The fact that it succeeded proves the cache is complete.

## How it works

| What | Code reference |
|---|---|
| Project-local cache directory | [`main.py:25-31`](main.py) |
| Pointing `Guard` at it via `GuardConfig.cache_dir` | [`main.py:33-37`](main.py) |
| Pre-downloading the full snapshot | [`main.py:46-50`](main.py) |
| Enforcing offline mode at runtime | [`main.py:54`](main.py) |

The HuggingFace Hub library uses a layered cache:

- The **cache_dir** you pass to `GuardConfig.cache_dir` becomes the root.
- Inside it, models live under `models--<org>--<repo>/snapshots/<sha>/`.
- File contents are deduplicated via symlinks.

This means you can ship the cache directory itself as a Docker volume, an artifact, or a checked-in fixture for tests — anything that gets the directory tree to the runtime works.

## Production tips

- **Bake the model into a Docker image at build time** to eliminate first-boot download latency:
  ```dockerfile
  ENV HF_HOME=/opt/bastion/cache
  RUN python -c "from huggingface_hub import snapshot_download; \
      snapshot_download('bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1')"
  ENV HF_HUB_OFFLINE=1
  ```
  See [`docker/Dockerfile.cpu`](../../docker/Dockerfile.cpu) for the full recipe.
- **For Kubernetes**, mount the cache as a `PersistentVolume` shared across pods so each replica doesn't re-download.
- **Verify the cache is complete** before relying on offline mode — `HF_HUB_OFFLINE=1` will fail loudly if anything is missing, which is what you want in CI.

## When to use this vs another pattern

- **Pattern 2 (SDK)** if you have network access and don't need
  build-time control over the model.
- **Pattern 4 (Docker)** — this advanced pattern is the foundation
  for those Docker images. The Dockerfiles internally do exactly what
  `main.py` shows.
- **Pattern 1 (raw ONNX)** if you also want to avoid the SDK dependency
  entirely.
