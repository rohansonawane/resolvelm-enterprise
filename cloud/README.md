# Cloud GPU training — DO NOT run from the Apple Silicon laptop.
# This directory is a template for an A10/A100 VM. No paid instances were created.

## What this is for

Fine-tune `Qwen/Qwen3-4B-Instruct-2507` with QLoRA SFT then DPO. Local M4 smoke uses a
tiny model or the sklearn fallback. 4B numbers must never be copied into README
unless this job actually ran.

## Steps (on a CUDA VM you already own)

```bash
# 1. Sync this repo
# 2. docker build -f cloud/Dockerfile.training -t resolvelm-train .
# 3. ./cloud/run_gpu_training.sh
```

Expected hardware: 24GB+ VRAM (A10) or 40GB (A100). Not benchmarked on this hardware.
