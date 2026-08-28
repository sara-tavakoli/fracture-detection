# Contributing

Thanks for taking a look. This is a research codebase; contributions that
improve rigor, reproducibility, or documentation are especially welcome.

## Setup
```bash
make venv install
pre-commit install
make smoke        # sanity-check the full pipeline on synthetic data
```

## Before opening a PR
- `make lint type test` must pass (ruff, `ruff format --check`, mypy, pytest).
- New behaviour needs a test. Metric/statistics code needs a test with a known
  closed-form or reference value.
- Keep public APIs typed. `src/fracture` is mypy-clean and should stay that way.
- If you change training defaults, update `configs/` **and** `docs/MODEL_CARD.md`.
- Do not commit data, checkpoints, or `artifacts/` (see `.gitignore`).

## Scope guidance
- Anything touching intended-use, disclaimers, or the model card gets extra
  review — this project must not drift toward implying clinical readiness.
- Prefer additive config/experiment files over changing baseline numbers silently.

## Commit style
Conventional-ish: `area: short imperative summary`. Reference issues where relevant.
