# Dashboard Core — Wormhole (Traversable-Wormhole Teleportation)

Traversable-wormhole-inspired quantum teleportation (Gao-Jafferis-Wall
theory), via a binary sparse Sachdev-Ye-Kitaev (SYK) model — the real
protocol backing Composer's "traversable-wormhole-inspired teleportation"
panel and the MCP server's wormhole tools. Built on
[`dense_evolution.fermions`](fermions.md) (`majorana_pauli_terms`) and
[`dense_evolution.entropy`](entropy.md) (`mutual_information`), the
protocol's actual readout quantity.

::: dashboard_core.wormhole

---

**Research log**: [Dense-Evolution-Discovery](https://github.com/tatopenn-cell/Dense-Evolution-Discovery)
runs this implementation through 20+ real, verified experiments (parameter
scans, generality checks, noise robustness, honest negative results) —
see its own [docs site](https://tatopenn-cell.github.io/Dense-Evolution-Discovery/wormhole_syk_teleportation/)
for the full write-up. This page documents the shipped implementation;
that repo documents what's been discovered by running it.
