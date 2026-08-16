# Resume bullets

Filled **after** `make smoke` from real artifacts. Until those files exist, do not quote percentages.

Placeholder (replace from `artifacts/model_comparison.csv`):

- Built ResolveLM, an incident-diagnosis post-training stack (SFT + DPO + FastAPI + Next.js) with grouped-split leakage controls and schema-validated JSON output.
- Measured BASE vs SFT vs DPO on a locked fixture test split (n and accuracies: see artifacts after smoke; 4B CUDA: not benchmarked on this hardware).
- Shipped a fallback trainer that still writes loss curves when Transformers/TRL cannot fit in 16 GB unified memory.
