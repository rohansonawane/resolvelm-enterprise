# Model card — ResolveLM

## Local model (this laptop)

| Item | Value |
|------|--------|
| Intended local weights | `Qwen/Qwen2.5-0.5B-Instruct` (ungated, Apache-2.0) |
| Production target | `Qwen/Qwen3-4B-Instruct-2507` with QLoRA — **not loaded here** |
| Fallback | `RuleBasedBackend` + TF-IDF/SGD `ClassifierBackend` |
| Hardware | Apple M4, 16 GB RAM, MPS, no NVIDIA |

Whether the 0.5B LoRA path actually ran is recorded in `artifacts/sft_train_metrics.json` (`transformers_used`, `hf_error` if any).

## Use

Given service, environment, symptoms, and log/metric/trace events, emit `IncidentAnalysis` JSON. Operators must treat SEV-1/SEV-2 `recommended_actions` as drafts.

## Training

- SFT: chat messages with system/user/assistant; assistant is strict JSON. Chat template applied only when a HF tokenizer exists — no invented special tokens.
- DPO: chosen = gold JSON; rejected ∈ {wrong_root_cause, unsupported_evidence, invalid_json, missing_evidence, dangerous_remediation, vague_response, severity_error, excessive_certainty}.

## Evaluation

BASE vs SFT vs DPO on the same locked test split. Metrics live in `artifacts/*_model_metrics.json` and `artifacts/model_comparison.csv`. Do not quote 4B numbers from this machine.

## Risks

- Hallucinated evidence IDs (mitigated by schema + `supported_evidence_rate` and human-review flags).
- Dangerous remediations (DPO negative class `dangerous_remediation`; still require a human).
- Domain shift from research logs to modern microservices.

## License of this repo

MIT. Upstream model licenses still apply if you download Qwen weights.
