# Contributing

AuditIQ is a small Python and Streamlit project. Keep changes focused, tested, and easy to review.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Pull Request Checklist

- Add or update tests for engine behavior.
- Run `pytest -q`.
- Update `README.md` when user-facing behavior changes.
- Update `CHANGELOG.md` for notable changes.
- Do not commit `.env`, local cache folders, generated reports, or private datasets.

## Code Style

- Prefer small functions with clear inputs and outputs.
- Keep check logic deterministic.
- Do not make Gemini or any external API required for the default flow.
- Avoid committing design handoff files or local assistant/tooling folders.
