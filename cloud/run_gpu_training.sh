#!/usr/bin/env bash
set -euo pipefail
# CUDA training driver. Refuse to run on Apple Silicon.
if [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Refusing to run cloud GPU training on Apple Silicon. Use a CUDA VM."
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found. Aborting."
  exit 2
fi
python scripts/prepare_dataset.py --config configs/cuda/data.yaml
python scripts/train_sft.py --config configs/cuda/sft.yaml
python scripts/train_dpo.py --config configs/cuda/dpo.yaml
python scripts/evaluate.py --config configs/cuda/evaluation.yaml
