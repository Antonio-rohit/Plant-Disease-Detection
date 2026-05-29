# Contributing

Thank you for taking the time to improve this project.

## Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run checks:

```bash
python -m compileall app.py src scripts tests
pytest
```

## Contribution Guidelines

- Keep model artifacts, datasets, generated reports, uploads, and cache files out of commits.
- Add tests for API, preprocessing, evaluation, or security changes.
- Do not commit secrets, API keys, local paths, or credentials.
- Use clear commit messages that describe the user-visible change.
- Document any model, dataset, or metric changes in the README or model card.

## Pull Request Checklist

- Tests pass locally.
- Documentation is updated.
- No large accidental files are included.
- No secrets or local machine paths are committed.
- Any ML metric changes are explained with the dataset split used.
