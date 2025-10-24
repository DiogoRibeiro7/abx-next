# Contributing to abx-next

## Dev setup
```bash
poetry install
pre-commit install
```

## Quality gates
- Lint: `poetry run ruff check .`
- Types: `poetry run mypy src`
- Tests: `poetry run pytest -q`

## Branches
- `main`: protected, release branch
- `develop`: integration branch
- feature branches: `feat/<short-topic>`
- fix branches: `fix/<short-topic>`

## Commit style
- Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`)

## PR checklist
- [ ] Added/updated tests
- [ ] Docs/README updated if needed
- [ ] Passing CI
