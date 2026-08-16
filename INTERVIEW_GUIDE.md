# Interview guide (derived from this repo)

Walk an interviewer through **this** codebase, not a generic LLM fine-tune story.

## 5 minutes

ResolveLM turns ops-style incidents into **strict JSON diagnoses**. Data is derived from Loghub-like failure patterns, split by incident family so variants cannot leak into test, then SFT teaches the JSON contract and DPO pushes away wrong causes, invented evidence, and dangerous remediations. Serving is FastAPI with a `ModelBackend` protocol: Transformers, vLLM (CUDA), or a rule/classifier fallback so the Mac still demos. Invalid generations become error payloads, not fake gold.

## 15 minutes

- **Data:** `generator.py` + `validate.py` + `grouped_temporal_split`. Evidence IDs must exist in the incident. DPO pairs never use test answers (`incidents_to_dpo` skips `split==test`).
- **Modeling:** `training/sft.py` tries TRL `SFTTrainer` + PEFT LoRA, then sklearn SGD. `training/dpo.py` same pattern with `DPOTrainer` / preference SGD.
- **Eval:** `evaluate.py` scores BASE vs SFT vs DPO on one locked test file. Metrics include schema success and supported-evidence rate, not just accuracy.
- **API:** `/api/v1/incidents/analyze` with size limits, timeouts, Prometheus, `request_id`.
- **Honest hardware:** 0.5B or fallback locally; 4B QLoRA lives in `configs/cuda/` only.

## 30 minutes

Trace a request: dashboard demo → FastAPI → `request_to_prompt` → backend → `parse_analysis_with_retry` → Pydantic. Then trace training: `prepare_dataset.py` writes JSONL, smoke trains 2–3 steps or fallback epochs, `loss_curve.json` is real. Discuss why grouped split beats random split for templated logs, why we refuse to coerce invalid JSON, and why vLLM is a stub on Apple Silicon.

## Topic prompts

### Architecture

Why a Protocol instead of a single HF client? How does KServe YAML relate to local FastAPI?

### Data

How do you prove no family leakage? What does `label_origin` buy you in an audit?

### Modeling

LoRA targets, why QLoRA is CUDA-only here, what the fallback trainer still proves (loss moved, adapter-equivalent artifact written).

### Eval

Why schema success can be 1.0 while root-cause accuracy is not. Why tokens/sec may be "Not benchmarked".

### Failures

Broken JSON, missing weights, oversized body, timeout — tests in `tests/integration/test_failures.py`.

### Scaling / cost

4B QLoRA batch/accum in `configs/cuda/sft.yaml`. This laptop must not be billed as that run. Cloud script refuses Darwin arm64.

### Deploy / MLOps

Compose healthchecks without `sleep 30`. MLflow aliases `candidate`/`champion`. HPA + KServe GPU example.

### Security / tradeoffs

Human review, dangerous-remediation negatives, no secret commits, MIT vs Qwen license if weights are downloaded.

## Likely hard questions

1. Why not parse JSON with regex only? (Schema: enums, evidence, confidence bounds.)
2. Why sklearn fallback is not cheating? (Different backend, labeled in artifacts, still a real trained model.)
3. How would you add RAG over runbooks without leaking test incidents into the index?
