# Security

## Reporting

Open a private report with the repository owner. Do not file public issues that include production logs, tokens, or customer data.

## What this software does not do

- It does not execute remediations. Actions are recommendations only.
- It does not store operator secrets. `.env` is gitignored; `deploy/kubernetes/secret.example.yaml` is a template.

## Serving controls

- Payload size cap (`RESOLVELM_MAX_PAYLOAD_BYTES`, default 256 KiB).
- Generation timeout (`RESOLVELM_GENERATE_TIMEOUT_S`).
- Structured JSON logs with `request_id`.
- Prometheus metrics at `GET /metrics`.
- Invalid model JSON is not rewritten into a successful diagnosis.

## Training data

Do not paste real PII, credentials, or production dumps into the fixture generator. Loghub samples, if downloaded, remain public research logs.

## Supply chain

CI and Dependabot watch pip, npm (dashboard), and GitHub Actions. Pin Python to 3.11 via `uv`. Production images should run as a non-root user and pull models from an allowlisted registry — not implemented as a live cluster on this laptop.

## Dangerous outputs

DPO trains against `drop_production_db`-style rejected actions. The API still sets `needs_human_review` when evidence IDs are missing from the request context.
