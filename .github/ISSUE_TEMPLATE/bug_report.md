---
name: Bug report
about: Something behaves incorrectly (wrong results, crash, unexpected exception)
title: ""
labels: bug
assignees: ""
---

**Describe the bug**
What's wrong, and what you expected instead.

**Minimal reproduction**
A minimal circuit/script that reproduces it — the smaller the better.

```python
import dense_evolution as de
# ...
```

**If it's a correctness bug** (wrong probabilities/statevector, silently dropped gates, etc.), include a comparison against a known-correct reference if you have one (another simulator, a hand-computed result) — that's usually the fastest way to confirm and fix it.

**Environment**
- `dense-evolution` version: `python -c "import dense_evolution; print(dense_evolution.__version__)"`
- Python version:
- OS:
- Installed extras (`[jax]`, `[gpu]`, `[qiskit]`, `[pennylane]`, `[dashboard]`, `[full]`):
