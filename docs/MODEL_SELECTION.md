# Model selection

## Local (executed or attempted on this M4)

**Primary attempt:** `Qwen/Qwen2.5-0.5B-Instruct`

- License: Apache-2.0, ungated on Hugging Face.
- Size: on the order of 0.5B parameters — small enough to *try* LoRA on 16 GB unified memory; 4B is not.
- Instruction-tuned, so the chat template exists (we do not invent special tokens).
- If download or MPS RAM fails, smoke still trains the TF-IDF/SGD fallback and serves `RuleBasedBackend`. That outcome is recorded in `artifacts/sft_train_metrics.json`, not hidden.

**Not selected locally:** `Qwen/Qwen3-4B-Instruct` — production class, QLoRA 4-bit, needs CUDA VRAM this machine does not have. Downloading it would also be an unnecessary multi-GB hit.

## Production (documented, not executed)

`configs/cuda/` and `cloud/` target **Qwen3-4B-Instruct + QLoRA + vLLM**. Reasons:

- Fits a single modern GPU with 4-bit adapters better than 7B/14B for an internal tool.
- Instruct checkpoint matches the JSON-in/JSON-out job.
- vLLM is the intended high-QPS backend; Transformers generate is the compatibility path.

Accuracy, tokens/sec, and VRAM for 4B: **Not benchmarked on this hardware.**

## Baseline

Keyword `RuleBasedBackend` is the BASE row in comparison tables. It is a legitimate heuristic, not a fake LLM score. SFT/DPO rows are the classifier or LoRA model actually produced by `make smoke`.
