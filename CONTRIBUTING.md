# Contributing to Dense-Evolution

Thanks for considering a contribution. This is a single-maintainer project — response times may vary, but every issue and PR gets read.

## Before you start

Dense-Evolution is released under the [Business Source License 1.1](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/license.md) — free for non-commercial use, with commercial-production terms defined in the license itself (converting to Apache 2.0 on the Change Date stated there). By submitting a contribution, you agree it's licensed under the same terms as the rest of the project.

## Reporting bugs

Open an [issue](https://github.com/tatopenn-cell/Dense-Evolution/issues) with:
- A minimal circuit/script that reproduces the problem
- What you expected vs. what actually happened
- `python -c "import dense_evolution; print(dense_evolution.__version__)"` output, plus your Python version and OS

If it's a correctness bug (wrong probabilities, wrong statevector, silently dropped gates), include a comparison against a known-correct reference if you have one — that's usually the fastest way to confirm and fix it.

## Reporting security issues

**Don't open a public issue for security vulnerabilities.** See [SECURITY.md](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/SECURITY.md).

## Development setup

```bash
git clone https://github.com/tatopenn-cell/Dense-Evolution.git
cd Dense-Evolution
pip install -e .[full]
```

### Pre-commit (recommended)

We use pre-commit to run black/isort/ruff/mypy locally. Install and enable hooks:

```bash
pip install pre-commit
pre-commit install
# run checks across the repo before opening a PR
pre-commit run --all-files
```

## Running tests

```bash
pytest tests/ -v
```

`tests/integration/test_interop.py` needs `qiskit`/`pennylane` installed (skips cleanly via `pytest.importorskip` if they aren't). CI (`.github/workflows/ci.yml`) runs the full suite on Python 3.10/3.11/3.12 on every push/PR to `main` — check it's green before asking for a review.

Tests are organized one file per source module, split into `tests/unit/` (a single `dense_evolution/` module in isolation, e.g. `tests/unit/test_simulator.py` ↔ `dense_evolution/backends/statevector.py`) and `tests/integration/` (multiple modules wired together, or a `tools/` app) — add new tests to the matching file rather than a catch-all, and pick the subdirectory matching what the new test actually exercises.

## Making changes

- Match the existing code style — no comments explaining *what* code does (names should do that); comments only for non-obvious *why* (a workaround, an invariant, a hard-won bug fix).
- Add tests for anything you fix or add — especially anything numerical (statevector/probability outputs). A test that would have caught the bug you just fixed is worth more than a description of the fix.
- Update `README.md`'s changelog section if your change is user-visible.
- Keep PRs focused — one fix or one feature per PR is easier to review than a bundle of unrelated changes.

## Submitting a PR

Push to a branch on your fork, open a PR against `main`, and describe *why* the change is needed, not just what it does — the "why" is what a reviewer (and future you) actually needs.
