# Model comparison

Locked test split (n=10 smoke incidents). No hyperparameter tuning on test.
Values copied from `artifacts/model_comparison.csv`.

SFT used `Qwen/Qwen2.5-0.5B-Instruct` + LoRA on MPS. DPO training ran (`train_loss=0.691156804561615`) but **DPO evaluation was not completed** after a host Python crash (LightGBM OpenMP SIGSEGV in a sibling project). Production CUDA Qwen3-4B numbers: **Not benchmarked on this hardware.**

| model | backend | n | root_cause_accuracy | root_cause_macro_f1 | severity_accuracy | json_parse_success | pydantic_schema_success | supported_evidence_rate | action_f1 | latency_p50_ms |
|-------|---------|---|---------------------|---------------------|-------------------|--------------------|-------------------------|-------------------------|-----------|----------------|
| base | rule | 10 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0255 |
| sft | transformers | 10 | 0.0 | 0.0 | 0.0 | 0.9 | 0.0 | 0.0 | 0.0 | 15944.31 |
| dpo | trl_dpo_train_only | Not benchmarked on this hardware. | Not benchmarked on this hardware. | Not benchmarked on this hardware. | Not benchmarked on this hardware. | Not benchmarked on this hardware. | Not benchmarked on this hardware. | Not benchmarked on this hardware. | Not benchmarked on this hardware. | Not benchmarked on this hardware. |
