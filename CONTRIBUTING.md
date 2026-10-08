# Contributing

Thanks for your interest in improving **misinfo-toolkit**!

## Workflow

1. Open an [issue](https://github.com/napp3r/misinfo-toolkit/issues) describing the bug or idea.
2. Create a branch from `main`: `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>` or `ci/<short-name>`.
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/) prefixes
   (`feat:`, `fix:`, `docs:`, `test:`, `ci:`, `chore:`, `refactor:`).
4. Open a pull request that references the issue (`Closes #N`). CI must be green before merging.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"
pre-commit install          # ruff lint + format on every commit
```

## Checks

```bash
ruff check . && ruff format --check .   # lint / formatting
pytest --cov                            # tests + coverage (must stay >= 85 %)
```

## Releasing

1. Bump `version` in `pyproject.toml` and `src/misinfo_toolkit/__init__.py`, update `CHANGELOG.md`.
2. Merge to `main`, then tag: `git tag v0.2.0 && git push origin v0.2.0`.
3. The **Release** workflow tests, builds and publishes a GitHub Release with the wheel and sdist.
