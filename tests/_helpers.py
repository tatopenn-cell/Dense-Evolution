"""Small assertion helpers shared across a few test files (not fixtures --
plain functions, imported explicitly where needed)."""
import numpy as np


def norm(sim):
    return float(np.linalg.norm(sim.get_statevector()))


def probs(sim):
    return sim.get_probabilities()
