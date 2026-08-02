# Come si calcola una molecola su un computer quantistico

Documento di riferimento (base di conoscenza / RAG) sul processo reale che
porta da una molecola (atomi + geometria) a un circuito quantistico
eseguibile che ne stima l'energia di stato fondamentale. Scritto per
tenere corretta la terminologia usata nel Composer (`dashboard_core/`) e
per colmare un buco reale nell'implementazione attuale, individuato
studiando le fonti sotto: il Composer oggi usa solo un ansatz
hardware-efficient generico, non l'ansatz UCCSD chimicamente motivato che
la letteratura usa come riferimento standard.

## 1. Hartree-Fock: il punto di partenza classico

Prima di toccare un qubit, si risolve il problema elettronico della
molecola con un metodo classico di campo medio: Hartree-Fock (HF).
Restituisce un insieme di **orbitali molecolari** (spaziali) occupati e
virtuali, e un'energia di riferimento (l'energia HF) che è un limite
superiore reale — ma non esatto — dell'energia vera (l'HF trascura la
correlazione elettronica).

Nel Composer questo è il metodo `method="dhf"` di PennyLane
(Differentiable Hartree-Fock, nativo, senza bisogno di PySCF).

## 2. Seconda quantizzazione e mappatura fermione→qubit

Ogni orbitale spaziale diventa **due qubit** (spin su/giù). L'Hamiltoniana
elettronica, scritta in operatori di creazione/distruzione fermionici
(seconda quantizzazione), viene trasformata in una somma di stringhe di
Pauli tramite una mappatura reale:

- **Jordan-Wigner**: mappatura diretta, un qubit per spin-orbitale,
  stringhe di Pauli più lunghe.
- **Bravyi-Kitaev**: mappatura ad albero, stringhe di Pauli mediamente
  più corte, stesso numero di qubit.

**Punto chiave, verificato nel codice**: lo spettro (quindi l'energia di
stato fondamentale) è identico con entrambe le mappature — cambia solo
la rappresentazione a qubit dell'operatore, non la fisica. Per questo nel
Composer la scelta della mappatura non altera mai il numero riportato nel
pannello Hamiltonian.

## 3. Active space (frozen-core): perché H2O usa 12 qubit e non 14

Ogni orbitale spaziale = 2 qubit, quindi il numero di qubit cresce in
fretta. Una molecola come l'acqua in base minima STO-3G ha 7 orbitali =
14 qubit. L'**active space approximation** congela gli orbitali di core
(fortemente legati, poco rilevanti per la chimica di valenza — per H2O,
l'1s dell'ossigeno) e riduce il problema a un sottoinsieme di elettroni e
orbitali attivi. È un'approssimazione reale e standard in chimica
computazionale, non un trucco: per questo nel Composer H2O usa
`active_electrons=8, active_orbitals=6` → 12 qubit, un numero onesto,
diverso dall'energia esatta a piena base ma fisicamente motivato.

## 4. Lo stato di riferimento di Hartree-Fock sul circuito

Il circuito quantistico parte sempre dallo **stato di occupazione di
Hartree-Fock**: un qubit `|1⟩` (porta X) per ogni spin-orbitale occupato
secondo HF, `|0⟩` per i virtuali. Esempio H2 (4 qubit, 2 elettroni):
occupazione `1100` — i primi due spin-orbitali occupati, gli ultimi due
vuoti. Questo è esattamente ciò che `dashboard_core/vqe.py` già genera
correttamente (verificato: l'energia di questo solo stato, senza nessuna
rotazione aggiuntiva, riproduce l'energia Hartree-Fock esatta).

## 5. VQE: il loop di ottimizzazione

Il Variational Quantum Eigensolver è un algoritmo ibrido:

1. Un circuito parametrico (**ansatz**) prepara uno stato di prova
   |Ψ(θ)⟩ a partire dallo stato di Hartree-Fock.
2. Si misura ⟨Ψ(θ)|H|Ψ(θ)⟩ (l'energia del Hamiltoniano nello stato di
   prova) — nel Composer, calcolata esattamente dal vero statevector,
   non stimata a shot.
3. Un ottimizzatore classico (Adam, con gradiente reale via
   differenziazione adjoint) aggiorna θ per abbassare l'energia.
4. Si ripete fino a convergenza. Per il principio variazionale,
   l'energia trovata è sempre ≥ l'energia esatta — l'errore residuo
   misura quanto l'ansatz scelto riesce ad avvicinarsi.

## 6. Le due famiglie di ansatz — e il buco da colmare

**Ansatz hardware-efficient** (quello attualmente implementato in
`dashboard_core/vqe.py`): layer generici di rotazioni RY + scala di CNOT,
nessun legame diretto con la struttura fermionica della molecola. Vantaggi:
pochi gate, poca profondità, facile da adattare a qualunque numero di
qubit. Svantaggio: non "sa" nulla della chimica specifica — è un template
NISQ generico, non derivato dagli operatori di eccitazione fermionici
della molecola.

**Ansatz UCCSD (Unitary Coupled-Cluster Singles and Doubles)**: il
riferimento standard in letteratura per VQE chimico. Costruito
applicando allo stato di Hartree-Fock gli operatori di **eccitazione
singola e doppia** (un elettrone che salta da un orbitale occupato a uno
virtuale, o una coppia che salta insieme), generati da
`qml.qchem.excitations(electrons, qubits)` — quindi *derivati
direttamente dalla struttura fermionica reale della molecola*, non da un
template generico. Ogni eccitazione diventa un gate reale
(`FermionicSingleExcitation` / `FermionicDoubleExcitation`, equivalenti a
rotazioni di Givens), che si decompone in CNOT + RY standard — eseguibile
su qualunque simulatore/hardware reale, incluso `dense_evolution`.

Esempio minimo verificato in letteratura (H2, 4 qubit, singola doppia
eccitazione |1100⟩→|0011⟩): stato di prova
|Ψ(θ)⟩ = cos(θ/2)|1100⟩ − sin(θ/2)|0011⟩, energia convergente
−1.13726250 Hartree in 13 iterazioni — coerente (stessa Hamiltoniana,
stesso ordine di grandezza) con i numeri esatti già verificati nel
Composer per H2 (−1.1372701749 Hartree).

**Questo è il pezzo mancante nell'implementazione attuale**: il Composer
oggi genera solo l'ansatz hardware-efficient. Un ansatz UCCSD reale
(eccitazioni derivate da `qml.qchem.excitations` + `qml.UCCSD`, poi
decomposto ed esportato come OpenQASM per l'esecuzione su
`dense_evolution`, esattamente come già fatto per l'ansatz
hardware-efficient) è il prossimo passo naturale, ed è quello che risponde
correttamente a "usa i fermioni per creare il circuito".

## 7. Come tutto questo entra in un simulatore reale

PennyLane costruisce e ottimizza il circuito (con gradienti reali) usando
il proprio dispositivo (`lightning.qubit`, differenziazione adjoint — 60x
più veloce del backprop di default misurato su LiH a 12 qubit in questo
progetto). Una volta convergenti i parametri, il circuito — sequenza
concreta di X/RY/CNOT (hardware-efficient) o di rotazioni di Givens
decomposte in CNOT+RY (UCCSD) — viene tradotto in OpenQASM 2.0 reale ed
eseguito sul motore proprio del progetto
(`dense_evolution.DenseSVSimulator` / `MPSSimulator`), con Qiskit che
disegna il circuito e mostra statevector/probabilità/Q-sphere. Lo stesso
motore, lo stesso circuito, nessuna doppia implementazione della fisica.

## Fonti

- [A brief overview of VQE — PennyLane Demos](https://pennylane.ai/demos/tutorial_vqe/)
- [qml.UCCSD — PennyLane documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.UCCSD.html)
- [qml.FermionicSingleExcitation — PennyLane documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.FermionicSingleExcitation.html)
- [qml.FermionicDoubleExcitation — PennyLane documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.FermionicDoubleExcitation.html)
- [qml.qchem.excitations — PennyLane documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.qchem.excitations.html)
- [Variational Quantum Algorithms for Chemical Simulation and Drug Discovery (arXiv:2211.07854)](https://arxiv.org/pdf/2211.07854)
