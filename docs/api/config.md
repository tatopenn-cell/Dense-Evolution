# Precision (`config`)

JAX computes in 32-bit floats unless it is told otherwise: about 7 correct
digits instead of about 16. A quantum simulation needs the 64-bit version
(`complex128` amplitudes), so Dense-Evolution switches JAX to 64 bits for you,
at the moment it is needed and never at `import` time.

## Step 1. It happens automatically

```python
import jax
import dense_evolution as de

before = jax.config.jax_enable_x64
sim = de.DenseSVSimulator(2)
sim.run_circuit_jit(de.QASMParser().parse('OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0]; cx q[0],q[1];').to_tuples())
(before, jax.config.jax_enable_x64, str(sim.get_statevector().dtype))
```

```
(False, True, 'complex128')
```

`import dense_evolution` alone leaves JAX as it was (`False`). Creating the
simulator calls `ensure_x64()`, which turns on JAX's 64-bit mode for the whole
process, and the Bell state comes back as `complex128`. The mitigation
functions, `arithmetic`, `postselect` and `spectral` do the same on entry.

## Step 2. Choosing the precision yourself

```python
import jax
import dense_evolution as de

de.set_precision(False)
sim = de.DenseSVSimulator(2)
jax.config.jax_enable_x64
```

```
False
```

`set_precision` is for the case where another JAX library in the same process
must stay in 32 bits. Once you call it, your choice sticks: `ensure_x64()` no
longer turns 64 bits back on, so the simulator above runs in single precision.
Call it before creating anything.

## Step 3. The same guard on your own functions

```python
import jax.numpy as jnp
from dense_evolution.config import with_x64

@with_x64
def norm(v):
    return jnp.linalg.norm(jnp.asarray(v))

str(norm([0.6, 0.8]).dtype)
```

```
float64
```

`with_x64` runs `ensure_x64()` before every call of the function it wraps.
Without it, the same `jnp.linalg.norm(jnp.asarray([0.6, 0.8]))` in a fresh
process returns `float32`: JAX silently truncates input it builds before
anything has enabled 64 bits.

---

## Details

`jax_enable_x64` is one flag for the whole Python process, shared by every
library that uses JAX. Earlier versions set it at import time in three
modules, which silently overrode a precision chosen by unrelated code running
in the same process; `config.py` is now the single place that sets it, lazily.

::: dense_evolution.config
