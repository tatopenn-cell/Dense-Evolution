"""
Executes the actual Python code blocks published in docs/getting-started.md
and docs/examples.md, instead of a hand-maintained copy of them.

The docs snippets were written by copying from real, already-tested code
(experiments/matrix_healing_zne.py, test_mps.py, test_autodiff.py) -- this
file closes the other direction: if a future API change (renamed function,
changed signature, removed argument) breaks what's *published*, these tests
fail immediately instead of the breakage sitting undiscovered until someone
rereads the site by hand.

Each block is located by its section heading (not "first ```python fence in
the file"), since docs/getting-started.md also has a Colab-only block under
"## Install" that uses IPython magics (`!git clone`, `%cd`) and is not valid
standalone Python -- it's intentionally not selected here.
"""
import re
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).parent


def _extract_section_code(md_path, heading):
    text = (REPO_ROOT / md_path).read_text(encoding='utf-8')
    heading_pos = text.index(heading)
    remainder = text[heading_pos + len(heading):]
    match = re.search(r"```python\n(.*?)\n```", remainder, re.DOTALL)
    assert match is not None, f"no python fence found under {heading!r} in {md_path}"
    return match.group(1)


def _run_section(md_path, heading):
    code = _extract_section_code(md_path, heading)
    namespace = {'__name__': '__doc_example__'}
    exec(compile(code, f"{md_path}::{heading}", 'exec'), namespace)
    return namespace


class TestGettingStartedExamples:

    def test_quick_start(self):
        ns = _run_section('docs/getting-started.md', '## Quick start')
        probs = np.asarray(ns['probs'])
        sv = np.asarray(ns['sv'])
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(sv) == pytest.approx(1.0, abs=1e-6)

    def test_anti_oom_chunk(self):
        ns = _run_section('docs/getting-started.md', '## Anti-OOM for large circuits')
        sim = ns['sim']
        assert sim.n == 27
        assert sim.num_chunks >= 1

    def test_zero_noise_extrapolation(self):
        ns = _run_section('docs/getting-started.md', '## Zero-Noise Extrapolation')
        raw_fidelity = float(ns['raw_fidelity'])
        corrected_fidelity = float(ns['corrected_fidelity'])
        assert 0.0 <= raw_fidelity <= 1.0
        assert 0.0 <= corrected_fidelity <= 1.0
        # fixed rng seed=0 -- this is a real regression check, not just "ran
        # without raising": the correction must still improve fidelity.
        assert corrected_fidelity > raw_fidelity


class TestExamplesPage:

    def test_density_matrix_zne_healing(self):
        ns = _run_section('docs/examples.md', '## Density-matrix ZNE healing')
        raw_fidelity = float(ns['raw_fidelity'])
        corrected_fidelity = float(ns['corrected_fidelity'])
        assert 0.0 <= raw_fidelity <= 1.0
        assert 0.0 <= corrected_fidelity <= 1.0
        assert corrected_fidelity > raw_fidelity

    def test_mps_low_entanglement(self):
        ns = _run_section('docs/examples.md', '## MPS for low-entanglement circuits')
        prob = np.asarray(ns['prob'])
        n = ns['n']
        assert prob[0] == pytest.approx(0.5, abs=1e-6)
        assert prob[2 ** n - 1] == pytest.approx(0.5, abs=1e-6)
        assert prob.sum() == pytest.approx(1.0, abs=1e-6)

    def test_differentiable_vqe(self):
        ns = _run_section('docs/examples.md', '## Differentiable VQE')
        energy = float(ns['energy'])
        ground_state = float(np.min(np.diag(ns['h_matrix'])))
        # fixed rng seeds (theta=3, hamiltonian=7) -- 30 Adam epochs must land
        # meaningfully below the mid-spectrum, not just "somewhere".
        assert energy < ground_state / 2
