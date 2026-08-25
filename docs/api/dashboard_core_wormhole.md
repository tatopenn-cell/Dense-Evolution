# Dashboard Core — Wormhole (Traversable-Wormhole Teleportation)

Traversable-wormhole-inspired quantum teleportation (Gao-Jafferis-Wall
theory), via a binary sparse Sachdev-Ye-Kitaev (SYK) model — the real
protocol backing Composer's "traversable-wormhole-inspired teleportation"
panel and the MCP server's wormhole tools. Built on
[`dense_evolution.fermions`](fermions.md) (`majorana_pauli_terms`) and
[`dense_evolution.entropy`](entropy.md) (`mutual_information`), the
protocol's actual readout quantity.

```python
import math
from dashboard_core.wormhole import select_good_instance, run_wormhole_protocol, run_wormhole_protocol_trotter

# The paper's own instance-selection criterion: screen candidate seeds
# for the one whose commuting/anticommuting term-pair count best matches
# the reference K=10 instance (34 commuting / 11 anticommuting of 45 pairs).
seed = select_good_instance(n_majorana=8, k_terms=10, J=math.sqrt(2), n_candidates=200, target_commuting=34)
print(seed)  # 61

kwargs = dict(n_majorana=8, k_terms=10, J=math.sqrt(2), t0=0.3, t1=0.60, seed=seed, with_message=True)
print(run_wormhole_protocol(mu=12, **kwargs))    # 0.0133 -- weaker signal
print(run_wormhole_protocol(mu=-12, **kwargs))   # 0.0179 -- the "traversable" sign shows more

# The same protocol via a real Trotterized gate circuit (~6300 two-qubit gates)
# instead of exact matrix exponentiation, for comparison against real hardware:
print(run_wormhole_protocol_trotter(mu=-12, **kwargs))  # 0.0182 -- closely matches the exact backend
```

::: dashboard_core.wormhole

---

**Research log**: [Dense-Evolution-Discovery](https://github.com/tatopenn-cell/Dense-Evolution-Discovery)
runs this implementation through 20+ real, verified experiments (parameter
scans, generality checks, noise robustness, honest negative results) —
see its own [docs site](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/wormhole_syk_teleportation/)
for the full write-up. This page documents the shipped implementation;
that repo documents what's been discovered by running it.
