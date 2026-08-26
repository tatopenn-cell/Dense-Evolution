# Registry (Hardware Detection)

`QuantumHardwareRegistry` reads the current machine's own RAM/GPU and suggests a safe
qubit limit — unrelated to noise, despite living in this module historically. See
[Noise](noise.md) for `NoiseModel`, `NoiseSpec`, and coherent adversarial noise.

::: dense_evolution.circuits.registry
