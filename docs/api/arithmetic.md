# Arithmetic (adders, comparators)

A register is a group of qubits read as one binary number. Quantum arithmetic adds,
subtracts and compares these numbers while keeping every superposition intact: an
adder applied to a register holding `|0> + |1>` returns `|c> + |c+1>`, both sums at
once. These operations are the building blocks of Shor's algorithm and of many
oracles.

Every arithmetic operation is reversible, so it simply moves each basis state to
another basis state. Dense-Evolution applies it exactly that way, as a permutation of
the statevector amplitudes.

## Step 1. Load two numbers into registers

```python
import numpy as np
import dense_evolution as de

qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
x q[0];
x q[1];
x q[3];
"""
circ = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(5)
sim.run_circuit_jit(circ)
sv0 = sim.get_statevector()
print(format(int(np.argmax(np.abs(sv0))), '05b'))
```

```
11010
```

Register `a` is qubits `[1, 0]` and register `b` is qubits `[4, 3, 2]`, each listed
least significant bit first. The `x` gates set `a = 3` (both qubits on) and `b = 2`
(only its middle bit, qubit 3, on). The printed bitstring is `q0 q1 q2 q3 q4`.

## Step 2. Add `a` into `b`

```python
sv1 = de.add_registers(sv0, 5, [1, 0], [4, 3, 2])
print(format(int(np.argmax(np.abs(sv1))), '05b'))
```

```
11101
```

`add_registers` maps `|a, b>` to `|a, a + b>`. Register `a` keeps `3`; register `b`
now holds `5` (qubits 2 and 4 on). `b` has one qubit more than `a`, so the sum can
never overflow.

## Step 3. Subtract it back

```python
sv2 = de.subtract_registers(sv1, 5, [1, 0], [4, 3, 2])
print(format(int(np.argmax(np.abs(sv2))), '05b'))
```

```
11010
```

`subtract_registers` maps `|a, b>` to `|a, b - a>` and is the exact inverse of
`add_registers`: the state returns to Step 1. If `b < a` the result wraps around and
the most significant qubit of `b` is `1`, which is how a subtraction tells which
number is larger.

## Step 4. Add a fixed number to a superposition

```python
qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[2];
"""
circ = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(3)
sim.run_circuit_jit(circ)
sv0 = sim.get_statevector()
sv1 = de.add_constant(sv0, 3, [2, 1, 0], 3)
print(np.round(np.asarray(sv1), 4))
```

```
[0.    +0.j 0.    +0.j 0.    +0.j 0.7071+0.j 0.7071+0.j 0.    +0.j
 0.    +0.j 0.    +0.j]
```

`h q[2]` puts the register `[2, 1, 0]` in `(|0> + |1>)/√2`. Adding the classical
number `3` gives `(|3> + |4>)/√2`: both values are shifted, and nothing is measured.
`c = 1` is an increment and `c = -1` a decrement.

## Step 5. Compare a register with a number

```python
qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[2];
x q[1];
"""
circ = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(4)
sim.run_circuit_jit(circ)
sv0 = sim.get_statevector()
sv1 = de.compare_constant(sv0, 4, [2, 1], 3, 3, '<')
print(np.round(np.asarray(sv1), 4))
```

```
[0.    +0.j 0.    +0.j 0.    +0.j 0.    +0.j 0.    +0.j 0.7071+0.j
 0.7071+0.j 0.    +0.j 0.    +0.j 0.    +0.j 0.    +0.j 0.    +0.j
 0.    +0.j 0.    +0.j 0.    +0.j 0.    +0.j]
```

Register `b` is qubits `[2, 1]` and holds `(|2> + |3>)/√2`. `compare_constant` flips the
output qubit 3 where `b < 3` is true: only the `|2>` branch, so the answer is now
entangled with the register (amplitudes at `0101` and `0110`). The registers are never
changed, only the output qubit. `op` can be `'<'`, `'>'`, `'<='`, `'>='`, `'=='` or
`'!='`; `compare_registers` does the same between two registers.

---

## Details

**Why a permutation.** A reversible function `f` on basis states has the unitary
`U|x> = |f(x)>` (Vedral, Barenco, Ekert, *Quantum networks for elementary arithmetic
operations*, quant-ph/9511018, Sect. II). Applying it as an index permutation is
exact and touches each amplitude once.

**Checked against the gate-level circuits.** The tests build the circuits from the
papers out of engine gates and compare them with these functions on every basis
state:

- `add_registers` against the ripple-carry adder with one ancilla (Cuccaro, Draper,
  Kutin, Moulton, quant-ph/0410184, MAJ/UMA construction), for 1, 2 and 3-bit numbers;
- `add_constant` against the transform adder in Fourier space with a classical
  addend (Draper, quant-ph/0008033, Sect. 5; Beauregard, quant-ph/0205095, Sect. 2.1);
- `compare_registers(..., '<')` against the comparator: complement `a`, compute only
  the high bit of the sum, undo (Cuccaro et al., Sect. 4.3);
- `compare_constant(..., '<')` against the most significant qubit after the reverse
  φADD(c) (Beauregard, Sect. 2.1, Fig. 4), for `c < 2^n` as in the paper.

**Register sizes.** With `len(b) == len(a) + 1` the sum is exact (Vedral et al.,
Sect. III.A); with `len(b) == len(a)` it is addition modulo `2^n` (Cuccaro et al.,
Sect. 4.1).

::: dense_evolution.circuits.arithmetic
