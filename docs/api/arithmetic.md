# Arithmetic (adders, comparators, modular)

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

## Step 6. Modular exponentiation, the core of Shor's algorithm

```python
qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
h q[0];
h q[1];
x q[5];
"""
circ = de.QASMParser().parse(qasm)
sim = de.DenseSVSimulator(6)
sim.run_circuit_jit(circ)
sv0 = sim.get_statevector()
sv1 = de.power_mod(sv0, 6, [1, 0], [5, 4, 3, 2], 7, 15)
p = np.abs(np.asarray(sv1)) ** 2
print([format(int(k), '06b') for k in np.flatnonzero(p > 1e-9)])
```

```
['000001', '010111', '100100', '111101']
```

Register `x` is qubits `[1, 0]`, put in an equal superposition of `0, 1, 2, 3` by the two
`h` gates. Register `y` is qubits `[5, 4, 3, 2]`, set to `1` by `x q[5]`.
`power_mod` maps `|x, y>` to `|x, y * 7^x mod 15>`, so each branch now pairs `x` with
`7^x mod 15`: `1, 7, 4, 13`. Reading the four bitstrings, `x` is the first two digits
and `y` the last four, least significant last. This is the state Shor's algorithm
measures to find the period of `7^x mod 15`.

The same module has the steps that build it: `add_registers_mod` and
`add_constant_mod` (`(a + b) mod N`), `multiply_add_mod` (`b + a*x mod N`) and
`multiply_mod` (`x -> a*x mod N` in place, for `a` coprime with `N`; with
`N = 2^n` and odd `a` it is plain multiplication). The modular functions accept
`controls`, a list of qubits that must all be `1` for the operation to act.

## Step 7. The same additions as gate circuits

```python
import numpy as np
import dense_evolution as de
from dense_evolution.circuits.arithmetic_qasm import cuccaro_adder_qasm

circ = de.QASMParser().parse(cuccaro_adder_qasm(2))
sim = de.DenseSVSimulator(circ.n_qubits)
sim.set_initial_state(np.eye(2 ** circ.n_qubits)[0b110100])
sim.run_circuit_jit(circ.to_tuples())
print(format(int(np.argmax(np.abs(sim.get_statevector()))), '06b'))
```

```
111010
```

The functions above move amplitudes directly, which is exact but is not a circuit you
could run on a device or put through a noise model. `cuccaro_adder_qasm(n)` returns
the ripple-carry adder of Cuccaro et al. as OPENQASM 2.0: registers `a[2]`, `b[3]`
and one ancilla `x[1]`, each least significant bit first. The input `110100` is
`a = 3`, `b = 2`; the output `111010` is `a = 3`, `b = 5`, ancilla back to `0`. The
circuit uses 4 Toffoli and 9 CNOT gates (`2n` and `4n + 1`).

The same module has `draper_adder_qasm(n)` (addition in Fourier space, no ancilla),
`constant_adder_qasm(c, n)` (adds a fixed number) and
`modular_constant_adder_qasm(c, N, n)` (adds a fixed number modulo `N`). Each one gives
the same state as the matching function above: `add_registers`, `add_constant`,
`add_constant_mod`. `cmult_mod_qasm(a, N, n)` and `controlled_ua_qasm(a, N, n)` go one
level up, to the controlled modular multiplication of Shor's algorithm, and match
`multiply_add_mod` and `multiply_mod`.

## Step 8. Shor's algorithm on 11 qubits

```python
from fractions import Fraction
from math import gcd
from dense_evolution.circuits.shor import shor_order_finding

y, m = shor_order_finding(7, 15, rng=2)
r = Fraction(y, 2 ** m).limit_denominator(15).denominator
y, r, gcd(7 ** (r // 2) - 1, 15), gcd(7 ** (r // 2) + 1, 15)
```

```
(192, 4, 3, 5)
```

Shor's algorithm factors `N` by finding the order `r` of a number `a`: the smallest `r`
with `a^r mod N = 1`. `shor_order_finding` runs the quantum part with the gate circuits
of Step 7 and returns one measured integer `y` out of `2^m` (`m = 8` for `N = 15`).
`y / 2^m = 192 / 256 = 3/4`, so the order of `7` modulo `15` is `4`, and
`gcd(7^2 - 1, 15)` and `gcd(7^2 + 1, 15)` give the factors `3` and `5`. Each run is one
measurement, so another seed can return `0` or `1/2`, which only gives a divisor of `r`;
in practice the run is repeated.

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
- `add_constant_mod` against the modular adder built from adders, a subtraction of
  `N`, the overflow qubit and its reset (Beauregard, Sect. 2.2, Fig. 5);
- `multiply_add_mod` against `n` controlled modular additions of `2^i*a mod N`
  (Vedral et al., Sect. III.C; Beauregard, Sect. 2.3, Eq. 2);
- `multiply_mod` against the controlled-`U_a` circuit: multiply-add, controlled swap,
  inverse multiply-add by `a^-1 mod N` (Beauregard, Sect. 2.3, Fig. 7);
- `power_mod` against the chain of controlled multiplications by `a^(2^i) mod N`
  (Vedral et al., Sect. III.D).

**Modular domain.** The modular functions act on values below `N`, as in the papers.
Basis states holding a value `>= N` are left unchanged, so every function stays a
permutation and therefore unitary.

**Register sizes.** With `len(b) == len(a) + 1` the sum is exact (Vedral et al.,
Sect. III.A); with `len(b) == len(a)` it is addition modulo `2^n` (Cuccaro et al.,
Sect. 4.1).

**Gate circuits in `arithmetic_qasm`.** The OPENQASM 2.0 generators follow the
figures of the papers: MAJ and UMA as user-defined `gate`s (Cuccaro et al., Fig. 1-2,
the two-CNOT UMA), the adder of Fig. 3-4; the QFT adder without swaps (Draper);
φADD(c) and φADD(c)MOD(N) (Beauregard, Fig. 3 and 5), here without the two controls.
CMULT(a)MOD(N) is `n` doubly controlled φADD(2^i a mod N)MOD(N) (Fig. 6), the
controlled-`U_a` is CMULT(a), a controlled swap and the inverse of CMULT(a^-1) (Fig. 7):
`2n + 3` qubits in total. `shor_order_finding` uses one control qubit, measured and reset
after each controlled-`U_{a^(2^k)}`, with the inverse QFT done semiclassically (Fig. 8);
the measurement is sampled in Python between circuit runs, since `QASMParser` skips
`measure`, `reset` and `if`, so one QASM program cannot feed a measurement back into later gates.
`tests/unit/test_arithmetic_qasm.py` compares each one with the permutation functions on
random states over every basis state of its domain. The modular adder needs
`0 <= b < N` and `0 <= c < N`, as in the paper; on `b >= N` it is outside its contract.

::: dense_evolution.circuits.arithmetic

::: dense_evolution.circuits.arithmetic_qasm

::: dense_evolution.circuits.shor
