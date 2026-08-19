"""Truncation of large numeric arrays (statevector, probabilities) to
their most significant entries -- a full statevector/probability array can
be thousands of entries for a 20+ qubit circuit; agents almost always care
about the dominant amplitudes, not the full dense array.

`top_k` is configurable per call (`RunCircuitInput.top_k`, threaded
through from dense_evolution_run_circuit) rather than a fixed 25 -- BUG
FIX: the pre-Phase-2 code hardcoded top_k=25 in the function signature
with no way for a caller to ask for more or fewer entries.

Measurement `counts` are deliberately NOT truncated here (a real, open
question from prog.txt's diagnosis, not silently dropped): `counts` is
already naturally bounded by `shots` (RunCircuitInput caps that at
1,000,000) and by the number of distinct basis states actually observed,
which for realistic qubit counts is far smaller than a full statevector.
Truncating it to a top-K + "other" shape would change
dense_evolution_run_circuit's response schema for every caller, not just
opt-in behavior -- left for a separate, deliberate decision rather than
folded into this phase's mechanical extraction.
"""


def _truncate_statevector(rows: list, top_k: int = 25) -> dict:
    sorted_rows = sorted(rows, key=lambda r: -r["abs"])
    return {
        "total_nonzero_amplitudes": len(rows),
        "shown": min(top_k, len(rows)),
        "top_amplitudes_by_magnitude": sorted_rows[:top_k],
    }


def _truncate_probabilities(probs: list, top_k: int = 25) -> dict:
    indexed = sorted(enumerate(probs), key=lambda t: -t[1])[:top_k]
    return {
        "total_basis_states": len(probs),
        "shown": min(top_k, len(probs)),
        "top_states_by_probability": [{"index": i, "probability": p} for i, p in indexed],
    }
