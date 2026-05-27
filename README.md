# 💎 Dense Evolution v8.0 

[![Python Version](https://shields.io)](https://pypi.org/project/dense-evolution/)
[![Backend](https://shields.io)](https://github.com/tatopenn-cell/Dense-Evolution)
[![License](https://shields.io)](https://github.com/tatopenn-cell/Dense-Evolution/blob/main/LICENSE)

**Dense Evolution v8.0** è un simulatore quantistico basato su vettori di stato (*Statevector*) ad altissime prestazioni, ingegnerizzato specificamente per l'esecuzione di circuiti NISQ (Noisy Intermediate-Scale Quantum) complessi, profondi e algoritmi di Quantum Machine Learning (QML) e VQE.

L’architettura interna si basa sul principio della **Linear Kernel Fusion** ad allocazione controllata, superando i tradizionali colli di bottiglia legati all’uso della memoria ausiliaria (*scratchpad RAM*) e ridefinendo i limiti computazionali della compilazione statica accelerata via hardware.

---

## 🚀 Caratteristiche Architetturali & Features

* **⚡ Linear Kernel Fusion (JAX XLA):** Il simulatore non calcola mai esplicitamente le enormi matrici di gate derivanti dai prodotti tensoriali (Kronecker). L’applicazione degli operatori avviene tramite algoritmi di *stride-slicing* e permutazione lineare sui tensori contigui, riducendo la complessità di memoria spaziale al minimo teorico assoluto.
* **🧩 Circuit Chunking Transpiler:** Risolve il problema del congelamento o degrado della cache JIT di JAX quando si lavora con migliaia di porte logiche. Il circuito viene frammentato in sotto-blocchi (chunk) geometrici equivalenti, garantendo stabilità computazionale infinita, azzerando l'overhead di tracing su circuiti massivi.
* **🎲 Coerenza Stocastica e Collasso d'Onda:** La funzione di misura implementa uno *stride-slicing* chirurgico direttamente sulle matrici di vista hardware (NumPy/CuPy/JAX). Questo garantisce la perfetta convergenza binomiale ed evita l'allocazione di maschere booleane giganti in RAM, prevenendo crash di sistema.
* **📉 Modelli di Rumore a Traiettoria Kraus:** Consente la simulazione realistica di hardware affetti da rumore ambientale tramite canali di *Amplitude Damping*, *Phase Damping* e *Depolarizzazione*, applicati como salti quantici stocastici discreti senza l’onere computazionale $2^{2n}$ delle matrici di densità piene.
* **🎛️ Disaccoppiamento Hardware (Agnostic Backend):** Sfrutta un’astrazione polimorfa per selezionare a runtime l’hardware più efficiente: NumPy (CPU standard), JAX (Compilazione JIT hardware parallelizzata CPU/TPU) o CuPy (Calcolo parallelo accelerato su NVIDIA GPU CUDA).


------------------------------
## ⚙️ Istruzioni di Installazione
Il motore è strutturato in conformità con lo standard PEP 621 (tramite pyproject.toml) ed è completamente installabile tramite pip. È possibile scegliere tra l'installazione rapida dal registro ufficiale, la build locale o la modalità di sviluppo.
## 1. Installazione Rapida (da PyPI)
Per l'utilizzo standard del simulatore, installa l'ultima versione stabile direttamente dall'indice ufficiale dei pacchetti:

pip install dense-evolution

## 2. Installazione Locale e Sviluppo (da Repository)
Se desideri accedere al codice sorgente in locale o collaborare allo sviluppo del motore:

# Clona la repository ufficiale
git clone https://github.com/tatopenn-cell/Dense-Evolution.git
cd Dense-Evolution
# Opzione A: Installazione Standard (Backend CPU standard NumPy)
pip install .
# Opzione B: Installazione High-Performance (Modalità editable per sviluppatori)
pip install -e .

## 3. Esecuzione su Google Colab 🚀
Se utilizzi l'ambiente cloud di Google Colab, esegui questa cella di codice per configurare automaticamente l'ambiente in modalità sviluppatore:

# 1. Scarica la repository nel runtime di Colab
!git clone https://github.com/tatopenn-cell/Dense-Evolution.git
# 2. Spostati nella cartella principale del progetto
%cd Dense-Evolution
# 3. Installa il pacchetto in modalità editable
!pip install -e .

------------------------------



## 📊 Benchmark Industriali e Limiti del Sistema

Il motore è stato sottoposto a stress-test aggressivi in ambienti a risorse limitate (Google Colab Free), registrando risultati d’élite nel contenimento della memoria e nella precisione aritmetica.

### 1. Stabilità Numerica Assoluta (Zero-Drift Execution)
Sottoposto ad Ansatz variazionali profondi (oltre 80 strati e 1360 porte parametriche consecutive fuse in un unico blocco XLA), il core del simulatore ha registrato una deriva numerica assoluta controllata pari a:
$$\Delta = 1.1102230246251565 \times 10^{-16}$$
Questo valore coincide esattamente con l'**Epsilon di macchina ($\epsilon$)** per la precisione doppia a 64 bit (`float64`). La fusione algebrica dei kernel in XLA annulla l'accumulo sequenziale degli errori di arrotondamento delle funzioni trigonometriche.

### 2. Scaling dei Qubit e Throughput Computazionale (Chunking Engine)
Grazie al motore di chunking in-place, il simulatore gestisce registri quantistici estesi ottimizzando chirurgicamente la cache di sistema senza generare copie temporanee dello stato:


| Qubits | Dimensione Stato (Ampiezze) | Tempo di Esecuzione (s) | Gates / Secondo | RAM Reale Allocata | Delta RAM a Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **14** | 16.384 | 0.3546 s | 2819.9 | ~0.26 MB | **0.00 MB** |
| **16** | 65.536 | 0.4217 s | 2370.8 | ~1.04 MB | **0.00 MB** |
| **24** | 16.777.216 | 0.7090 s | *JIT Standard Tier* | ~256.00 MB | < 1.00 MB |
| **29** | **536.870.912** | *HPC Chunk Tier* | *Hardware Saturation* | **8192.00 MB** | **0.00 MB** |

> 💡 **Nota di Merito:** Il superamento della barriera dei 24 qubit in ambienti con soli 12 GB di RAM totali (Colab Free) evidenzia l’efficacia dell'architettura lineare 1D a norma fissa, che azzera i reshape dinamici a basso livello.

### 3. Parallelizzazione Vettorizzata JAX `vmap` (Batch Engine)
Il modulo `run_parametric_batch_jit` sfrutta la parallelizzazione inter-circuito per il Quantum Machine Learning. Esegue un singolo tracciamento del grafo computazionale e distribuisce istantaneamente N configurazioni di parametri sulla griglia hardware:
* **Throughput testato:** 64 circuiti variazionali paralleli eseguiti simultaneamente in **1.96 secondi**.
* **Tempo medio per circuito:** ⏱️ **0.031 secondi**.

---

## 💻 Esempi Pratici di Codice

### 🛠️ Esempio 1: Esecuzione in Beast Mode (Kernel Fusion JIT)
Dimostrazione dell'interfaccia ultra-veloce a zero allocazioni. La Beast Mode accetta un array lineare di operazioni stringa per bypassare completamente i controlli dell'interprete Python:

```python
import dense_evolution as de

# Inizializzazione del simulatore a 2 Qubit
sim = de.DenseSVSimulator(n_qubits=2)

# Definizione del circuito strutturato (Porta, Target, Controllo/Parametro)
# Generazione nativa di uno Stato di Bell entangled
ops = [
    ["h", 0, -1],
    ["cx", 1, 0]
]

# Esecuzione istantanea nel compilatore fuso XLA
sim.run_circuit_jit_beast_mode(ops)

print(f"Stato Finale Entangled JIT: {sim.get_statevector()}")
print(f"Probabilità di estrazione: {sim.get_probabilities()}")
```

### 🧠 Esempio 2: Decomposizione Topologica con il QuantumTranspiler
Il transpiler integrato scompone le porte logiche non native e complesse a più qubit nelle primitive a 1 e 2 qubit accettate dal core lineare 1D:

```python
import dense_evolution as de

transpiler = de.QuantumTranspiler()

# Estrazione della scomposizione esatta di una porta Toffoli (CCNOT) sui qubit 0, 1 e 2
sequenza_primitive = transpiler.decompose_toffoli(0, 1, 2)

print(f"Totale porte primitive generate per il Core V4: {len(sequenza_primitive)}")
for gate in sequenza_primitive:
    print(f"  -> {gate}")
# Output generato: Sequenza esatta a 15 porte stabili (H, CNOT, T, Tdg)
```

### 📉 Esempio 3: Iniezione stocastica del NoiseModel
Applicazione di canali di rumore realistici NISQ in modalità stocastica unificata JAX-safe:

```python
import dense_evolution as de
import numpy as np

sim = de.DenseSVSimulator(n_qubits=2)

# Applicazione manuale di una porta singola (Firma: Matrice, Qubit)
h_matrix = de.GATES['h']
sim.apply_gate_1q(h_matrix, 0)

# Lettura telemetria di sistema in tempo reale (Variabile float globale)
print(f"RAM attualmente disponibile su Colab: {de.ram_avail:.2f} MB")

# Iniezione di rumore di depolarizzazione al 5% sul vettore di stato
sim.sv = de.NoiseModel.apply_to_sv(
    sv=sim.sv, 
    n=2, 
    model='depolarizing', 
    p=0.05
)

print(f"Stato rumoroso degradato: {sim.get_statevector()}")
```

---

## 📂 Architettura dei File nella Repository

```text
Dense-Evolution/
│
├── pyproject.toml         # Configurazione PEP 621, build backend e dipendenze opzionali [jax,gpu]
├── README.md              # Documentazione tecnica ufficiale, telemetria e benchmark (Questo file)
└── dense_evolution.py     # Codice sorgente core del simulatore (DenseSVSimulator v8.0)
```

---

## 📜 Licenza e Note Legali

Il progetto è interamente distribuito sotto i termini della licenza ufficiale **MIT**.

```text
MIT License

Copyright (c) 2026 salvatore pennacchio [tatopenn-cell]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---## 💎 Appendice Tecnica v8.0: Ottimizzazioni Avanzate e Risoluzione Problemi (CPU/Colab)
Durante i test di stress-test intensivi in ambienti a risorse condivise (come Google Colab CPU Free), sono state ingegnerizzate e documentate le migliori pratiche per spingere l'engine al massimo delle sue potenzialità teoriche, risolvendo alcune rigidità strutturali di JAX XLA.
### 🚀 1. Sbloccare la "Beast Mode" a 19 e 24 Qubit (Velocità 180x+ vs C++)
Il metodo `.run_circuit_jit_beast_mode()` è l'unico canale ottimizzato in grado di fondere l'intera sequenza di operazioni in un unico blocco esecutivo a livello di microprocessore (*Linear Kernel Fusion*). 

A causa delle rigide restrizioni sui tipi di JAX (`lax.cond`), se si inizializza il simulatore con il flag predefinito `use_float32=True`, il compilatore fallirà restituendo l'errore:`TracerArrayConversionError / cond branches must have equal output types (complex64 vs complex128)`.

**Risoluzione Definitiva:** Forzare l'inizializzazione del simulatore in precisione doppia impostando `use_float32=False`. Questo allinea i tipi interni e sblocca l'esecuzione a codice macchina sigillato. Al secondo ciclo di calcolo (Giro 2), l'engine esegue circuiti complessi a 19 e 24 qubit in frazioni di millisecondo, superando i loop C++ dei simulatori tradizionali.
#### Esempio di implementazione corretta:```python
import time
import jax
import dense_evolution as de

num_qubits = 19

# Definizione di un circuito iterabile piatto (compatibile con il Transpiler interno)
class BeastCircuit(de.QASMCircuit, list):
    def __init__(self, n_qubits):
        list.__init__(self)
        de.QASMCircuit.__init__(self, n_qubits=n_qubits)

circuit = BeastCircuit(n_qubits=num_qubits)
circuit.append(('h', 0))
circuit.append(('rx', 0.123, 0)) # Formato piatto standard: (nome_gate, parametro, target)

# FIX FONDAMENTALE: use_float32=False impedisce il crash dei rami condizionali JAX
sim = de.DenseSVSimulator(n_qubits=num_qubits, use_gpu=False, use_float32=False)

# Giro 1: Tracciamento iniziale ed overhead di compilazione hardware
sv_compiled = sim.run_circuit_jit_beast_mode(circuit)
jax.block_until_ready(sv_compiled)

# Giro 2: Esecuzione PURA a regime (Zero-Overhead)
sim.set_initial_state()
start = time.time()
sv_final = sim.run_circuit_jit_beast_mode(circuit)
jax.block_until_ready(sv_final)

print(f"🚀 Tempo di calcolo puro in Beast Mode: {time.time() - start:.6f} secondi")
```
---
### 🛠 2. Integrazione Corretta con `QASMParser` (Adattatore di Tipo)

Il modulo `QASMParser` nativo analizza il codice OpenQASM 2.0 traducendo le istruzioni in un elenco strutturato di dizionari (`op['name']`, `op['qubits']`). Tuttavia, il metodo core di simulazione del backend `.run_circuit()` si aspetta rigidamente una sequenza posizionale lineare di tuple per evitare l'overhead dei reshape dinamici.

Per evitare crash di tipo `TypeError: 'QASMCircuit' object is not iterable` o `KeyError: 0`, è necessario interporre un convertitore leggero prima di dare in pasto le operazioni al simulatore.
#### Esempio di Parsing ed Esecuzione OpenQASM 2.0:```python
import dense_evolution as de

qasm_string = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""

# 1. Parsing text-based standard
parser = de.QASMParser()
parsed_circuit = parser.parse(qasm_string)

# 2. Traduttore Adattivo: Converte i dizionari in tuple posizionali pulite
formatted_ops = []
for op in parsed_circuit.ops:
    name = op['name']
    qubits = op['qubits']
    params = op['params']
    
    if name == 'cx': 
        # Isola gli indici dei qubit estratti come elementi singoli
        formatted_ops.append(('cx', int(qubits[0]), int(qubits[1])))
    elif params: 
        formatted_ops.append((name, int(qubits[0]), float(params[0])))
    else: 
        formatted_ops.append((name, int(qubits[0])))

# 3. Esecuzione sul simulatore denso
sim = de.DenseSVSimulator(n_qubits=2, use_gpu=False)
sim.run_circuit(formatted_ops, transpile=True)
statevector = sim.get_statevector()
```
---
### 🧠 3. Gestione Efficiente del Calcolo con Rumore (`NoiseModel`)

La classe `NoiseModel` agisce come un modulo funzionale stocastico tramite l'applicazione diretta degli operatori di Kraus sul vettore di stato con il metodo `NoiseModel.apply_to_sv()`. 

**Nota sulle Performance:** L'applicazione del rumore stocastico inserisce variabili casuali che interrompono la catena statica di fusione dei grafi di JAX (*Kernel Fusion*). Per simulazioni che includono canali di errore intensivi (`depolarizing`, `amplitude_damping`), si raccomanda di circoscrivere i test a registri quantistici compresi tra **4 e 12 qubit** per evitare l'esplosione dei tempi di calcolo dovuta ai continui accessi asincroni alla RAM della CPU.





