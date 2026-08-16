# GPU setup — train ResolveLM on a real NVIDIA box

This MacBook ran **smoke SFT** on `Qwen/Qwen2.5-0.5B-Instruct` (LoRA, MPS) plus a rule/sklearn fallback. The NVIDIA machine trains **Qwen3-4B-Instruct QLoRA SFT then DPO** on a larger derived incident corpus.

Those runs write `artifacts/gpu/` and `docs/GPU_MODEL_COMPARISON.md`. They must **not** be pasted into the Mac table in README.md.

Weights are **not** in git. Hugging Face downloads them on the GPU box (`Qwen/Qwen3-4B-Instruct-2507`, Apache-2.0).

## 1. Clone

```bash
git clone https://github.com/rohansonawane/resolvelm-enterprise.git
cd resolvelm-enterprise
```

Need: Ubuntu (or similar), NVIDIA driver, CUDA matching the PyTorch wheel, Python 3.11, [uv](https://docs.astral.sh/uv/), ~20 GiB free disk (4B weights + adapters + dataset).

VRAM:

| Card | Config |
|------|--------|
| ~24 GB+ (recommended) | `configs/cuda/sft.yaml` + `dpo.yaml` — Qwen3-4B-Instruct-2507, 4-bit QLoRA |
| ~8–12 GB | `configs/cuda/sft_1p5b.yaml` + `dpo_1p5b.yaml` — Qwen2.5-1.5B-Instruct LoRA |

## 2. Python env + CUDA PyTorch

```bash
make gpu-setup
make gpu-check
```

If `gpu-check` says `torch.cuda.is_available() is False`, the env has a CPU wheel. Install a CUDA build (example, CUDA 12.4):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install bitsandbytes
make gpu-check
```

Use the index that matches **your** CUDA (`cu118`, `cu121`, `cu124`, `cu126`). See https://pytorch.org/get-started/locally/

`bitsandbytes` is Linux CUDA only. Do not install it on Apple Silicon.

## 3. Data (derived / synthetic, larger)

```bash
make gpu-data
```

This generates a grouped-temporal split (~800 incident variants by default) under `data/processed/` and quality JSON under `artifacts/gpu/`. It is **not** a proprietary ticket dump. Patterns follow public Loghub Hadoop/HDFS/OpenStack style templates.

## 4. Train (SFT then DPO)

```bash
# Default: 4B QLoRA (needs bitsandbytes + enough VRAM)
make gpu-train

# First run on a smaller card
make gpu-train GPU_SFT=configs/cuda/sft_1p5b.yaml GPU_DPO=configs/cuda/dpo_1p5b.yaml
```

`scripts/check_gpu.py` refuses to start on Apple Silicon. CUDA configs do **not** fall back to sklearn if Transformers fails.

The first 4B download is several gigabytes. Later runs reuse the Hugging Face cache.

## 5. Evaluate, serve, dashboard

```bash
make gpu-eval
# 1.5B adapters:
make gpu-eval GPU_EVAL=configs/cuda/evaluation_1p5b.yaml

make gpu-serve
# other terminal
make ui
```

API: http://127.0.0.1:8101 — `gpu-serve` loads the DPO adapter with 4-bit weights when `RESOLVELM_LOAD_IN_4BIT=true`. Dashboard: http://localhost:3101.

Measured GPU numbers live in `artifacts/gpu/` and `docs/GPU_MODEL_COMPARISON.md`. If a cell is missing, it was not run. Do not invent it.

## What “real” means here

| | Mac smoke | This GPU path |
|---|---|---|
| Data | 24 families × 2 variants | 200 families × 4 variants |
| Base model | Qwen2.5-0.5B-Instruct (or rule/sklearn) | Qwen3-4B-Instruct-2507 (or 1.5B first-run) |
| Method | LoRA on MPS / fallback | 4-bit QLoRA SFT → DPO on CUDA |
| Metrics file | `artifacts/` | `artifacts/gpu/` |
| Time | minutes | tens of minutes → hours |

Still a derived corpus. JSON schema accuracy on this generator is **not** an on-call SLA.

Hub ID note: there is no `Qwen/Qwen3-4B-Instruct` repo. The instruct checkpoint is `Qwen/Qwen3-4B-Instruct-2507`.
