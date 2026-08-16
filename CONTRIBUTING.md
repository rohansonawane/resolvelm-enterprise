# Contributing

## Setup

```bash
uv python pin 3.11
uv sync --extra dev --extra train
make test
```

Dashboard: Node 22+, `cd dashboard && npm install && npm run dev -- --port 3101`.

## Rules

- Do not invent metrics. Write numbers only from `artifacts/*.json` / `artifacts/*.csv`.
- Do not commit `.venv`, `node_modules`, checkpoints, raw datasets, `.env`, or MLflow volumes.
- Do not download 4B weights on the laptop profile.
- Keep incident variants inside one split (`family_id`).
- Prefer fixing the parser over loosening `IncidentAnalysis`.

## PR checklist

- [ ] `uv run ruff check src scripts tests`
- [ ] `uv run pytest tests -q`
- [ ] New metrics regenerated via `scripts/evaluate.py` if eval changed
- [ ] CUDA claims match whether CUDA actually ran

## Code layout

Package code lives in `src/resolvelm/`. Scripts in `scripts/` are thin CLIs. Tests mirror `tests/{unit,integration,e2e,data,model}`.
