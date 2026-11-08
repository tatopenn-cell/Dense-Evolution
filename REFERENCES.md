# References

The papers that the code and the documentation of Dense-Evolution cite. The list of arXiv papers was
written from the `arXiv:` identifiers found in `dense_evolution/`, `tools/`, `docs/`, `research/` and
`README.md`; the titles, authors and years were read from the arXiv API. Each entry says where it is cited.

## Papers on arXiv

| arXiv | Authors | Title | Year | Cited in |
|---|---|---|---|---|
| [0904.3556](https://arxiv.org/abs/0904.3556) | Thomas M. Stace, Sean D. Barrett, Andrew C. Doherty | Thresholds for topological codes in the presence of loss | 2009 | `dense_evolution/physics/qec.py` |
| [1008.1607](https://arxiv.org/abs/1008.1607) | Takashi Tsuchimochi, Gustavo E. Scuseria | ROHF Theory Made Simple | 2010 | `dense_evolution/native_hf/scf.py`, `docs/api/native_hf.md` |
| [1106.5458](https://arxiv.org/abs/1106.5458) | John A. Smolin, Jay M. Gambetta, Graeme Smith | Maximum Likelihood, Minimum Effort | 2011 | `dense_evolution/mitigation/zne.py`, `docs/api/mitigation.md`, `docs/examples.md` |
| [1205.6812](https://arxiv.org/abs/1205.6812) | Morag Scrimgeour, Tamara Davis, Chris Blake et al. | The WiggleZ Dark Energy Survey: the transition to large-scale cosmic homogeneity | 2012 | `dense_evolution/physics/qec.py` |
| [1306.3142](https://arxiv.org/abs/1306.3142) | Martin Müller-Lennert, Frédéric Dupuis, Oleg Szehr et al. | On quantum Renyi entropies: a new generalization and some properties | 2013 | `dense_evolution/mitigation/renyi.py` |
| [1411.4028](https://arxiv.org/abs/1411.4028) | Edward Farhi, Jeffrey Goldstone, Sam Gutmann | A Quantum Approximate Optimization Algorithm | 2014 | `dense_evolution/backends/variational_numpy.py`, `docs/assets/composer/py/variational_numpy.py` |
| [1703.01517](https://arxiv.org/abs/1703.01517) | Nicolas Delfosse, Gilles Zémor | Linear-Time Maximum Likelihood Decoding of Surface Codes over the Quantum Erasure Channel | 2017 | `dense_evolution/physics/qec.py` |
| [1704.05018](https://arxiv.org/abs/1704.05018) | Abhinav Kandala, Antonio Mezzacapo, Kristan Temme et al. | Hardware-efficient Variational Quantum Eigensolver for Small Molecules and Quantum Magnets | 2017 | `dense_evolution/backends/variational_numpy.py`, `docs/assets/composer/py/variational_numpy.py` |
| [1707.03429](https://arxiv.org/abs/1707.03429) | Andrew W. Cross, Lev S. Bishop, John A. Smolin et al. | Open Quantum Assembly Language | 2017 | `dense_evolution/backends/qasm_subset.py`, `docs/assets/composer/py/qasm_subset.py` |
| [1708.00798](https://arxiv.org/abs/1708.00798) | Jérémy Ribeiro, Gláucia Murta, Stephanie Wehner | Fully device independent Conference Key Agreement | 2017 | `dense_evolution/protocols/di_qkd_ghz.py`, `dense_evolution/protocols/dicka_protocol2.py`, `docs/api/protocols.md`, `research/local_site/app/server.py` and 3 more |
| [1709.06218](https://arxiv.org/abs/1709.06218) | Nicolas Delfosse, Naomi H. Nickerson | Almost-linear time decoding algorithm for topological codes | 2017 | `dense_evolution/physics/qec.py` |
| [1712.02360](https://arxiv.org/abs/1712.02360) | S. T. Spitz, B. Tarasinski, C. W. J. Beenakker et al. | Adaptive weight estimator for quantum error correction | 2017 | `dense_evolution/physics/qec.py` |
| [1801.01037](https://arxiv.org/abs/1801.01037) | Ryan LaRose | Distributed Memory Techniques for Classical Simulation of Quantum Circuits | 2018 | `docs/api/chunk.md` |
| [1803.00745](https://arxiv.org/abs/1803.00745) | Kosuke Mitarai, Makoto Negoro, Masahiro Kitagawa et al. | Quantum Circuit Learning | 2018 | `dense_evolution/backends/variational_numpy.py`, `docs/assets/composer/py/variational_numpy.py` |
| [1901.04085](https://arxiv.org/abs/1901.04085) | Rodrigo Nogueira, Kyunghyun Cho | Passage Re-ranking with BERT | 2019 | `docs/api/ia_utils_rag.md`, `tools/ia_utils/rag.py` |
| [1910.09534](https://arxiv.org/abs/1910.09534) | Edwin Pednault, John A. Gunnels, Giacomo Nannicini et al. | Leveraging Secondary Storage to Simulate Deep 54-qubit Sycamore Circuits | 2019 | `dense_evolution/backends/chunk/core.py`, `dense_evolution/backends/chunk/disk_overflow.py`, `docs/api/chunk.md` |
| [1912.12029](https://arxiv.org/abs/1912.12029) | Susi Lehtola, Frank Blockhuys, Christian Van Alsenoy | An overview of self-consistent field calculations within finite basis sets | 2019 | `docs/api/native_hf.md` |
| [2004.04906](https://arxiv.org/abs/2004.04906) | Vladimir Karpukhin, Barlas Oğuz, Sewon Min et al. | Dense Passage Retrieval for Open-Domain Question Answering | 2020 | `docs/api/ia_utils_rag.md`, `tools/ia_utils/rag.py` |
| [2005.10921](https://arxiv.org/abs/2005.10921) | Tudor Giurgica-Tiron, Yousef Hindy, Ryan LaRose et al. | Digital zero noise extrapolation for quantum error mitigation | 2020 | `dense_evolution/mitigation/zne.py`, `docs/api/mitigation.md` |
| [2011.04366](https://arxiv.org/abs/2011.04366) | Muhammad Firmansyah Kasim | Derivatives of partial eigendecomposition of a real symmetric matrix for degenerate cases | 2020 | `dense_evolution/mitigation/zne.py`, `dense_evolution/physics/spectral.py`, `docs/api/spectral.md` |
| [2103.12097](https://arxiv.org/abs/2103.12097) | Daniel P. Arovas, Erez Berg, Steven Kivelson et al. | The Hubbard Model | 2021 | `dense_evolution/physics/fermions.py`, `docs/api/fermions.md` |
| [2104.05219](https://arxiv.org/abs/2104.05219) | Matt McEwen, Lara Faoro, Kunal Arya et al. | Resolving catastrophic error bursts from cosmic rays in large arrays of superconducting qubits | 2021 | `dense_evolution/circuits/trotter.py`, `dense_evolution/noise/cosmic_ray.py`, `dense_evolution/noise/density_matrix_channels.py`, `dense_evolution/physics/qec.py` and 8 more |
| [2106.12587](https://arxiv.org/abs/2106.12587) | Lorenzo Leone, Salvatore F. E. Oliviero, Alioscia Hamma | Stabilizer Rényi entropy | 2021 | `dense_evolution/mitigation/stabilizer_renyi_entropy.py`, `docs/api/mitigation.md` |
| [2111.09967](https://arxiv.org/abs/2111.09967) | Juan Miguel Arrazola, Soran Jahangiri, Alain Delgado et al. | Differentiable quantum computational chemistry with PennyLane | 2021 | `dense_evolution/native_hf/__init__.py`, `docs/api/native_hf.md` |
| [2203.13300](https://arxiv.org/abs/2203.13300) | Piotr Migdał, Klementyna Jankiewicz, Paweł Grabarz et al. | Visualizing quantum mechanics in an interactive simulation -- Virtual Lab by Quantum Flytrap | 2022 | `tools/dashboard/sota_views.py` |
| [2211.07854](https://arxiv.org/abs/2211.07854) | Hasan Mustafa, Sai Nandan Morapakula, Prateek Jain et al. | Variational Quantum Algorithms for Chemical Simulation and Drug Discovery | 2022 | `research/quantum_chemistry_vqe_pipeline.md` |
| [2306.09292](https://arxiv.org/abs/2306.09292) | Kaifeng Bu, Weichen Gu, Arthur Jaffe | Stabilizer Testing and Magic Entropy via Quantum Fourier Analysis | 2023 | `dense_evolution/mitigation/magic_entropy.py` |
| [2307.05203](https://arxiv.org/abs/2307.05203) | Ritajit Majumdar, Pedro Rivero, Friederike Metz et al. | Best practices for quantum error mitigation with digital zero-noise extrapolation | 2023 | `dense_evolution/mitigation/zne.py`, `docs/api/mitigation.md` |
| [2401.14385](https://arxiv.org/abs/2401.14385) | Kaifeng Bu, Weichen Gu, Arthur Jaffe | Quantum Ruzsa Divergence to Quantify Magic | 2024 | `dense_evolution/mitigation/magic_entropy.py` |
| [2405.02278](https://arxiv.org/abs/2405.02278) | James Mills, Rawad Mezher | Mitigating photon loss in linear optical quantum circuits | 2024 | `dense_evolution/mitigation/zne.py` |
| [2408.00829](https://arxiv.org/abs/2408.00829) | Shouzhen Gu, Yotam Vaknin, Alex Retzker et al. | Optimizing quantum error correction protocols with erasure qubits | 2024 | `docs/api/qec.md` |
| [2411.13509](https://arxiv.org/abs/2411.13509) | Kao-Yueh Kuo, Yingkai Ouyang | Degenerate quantum erasure decoding | 2024 | `dense_evolution/physics/qec.py` |
| [2501.00331](https://arxiv.org/abs/2501.00331) | Yasunari Suzuki, Takanori Sugiyama, Tomochika Arai et al. | Q3DE: A fault-tolerant quantum computer architecture for multi-bit burst errors by cosmic rays | 2024 | `tools/dashboard/app.py` |
| [2510.00895](https://arxiv.org/abs/2510.00895) | Michael J. McGuffin, Jean-Marc Robert | Visualizing Quantum Circuits: State Vector Difference Highlighting and the Half-Matrix | 2025 | `tools/dashboard/sota_views.py` |
| [2604.10090](https://arxiv.org/abs/2604.10090) | Moongul Byun, Hyeonsoo Lee, Keun-Young Kim | Quantum simulation of traversable-wormhole-inspired quantum teleportation in a chaotic binary sparse SYK model | 2026 | `dense_evolution/circuits/trotter.py`, `dense_evolution/physics/fermions.py`, `docs/api/entropy.md`, `docs/api/fermions.md` and 9 more |
| [2604.24475](https://arxiv.org/abs/2604.24475) | Andriy Miranskyy, Adam Sorrenti, Jasmine Thind et al. | Improving Zero-Noise Extrapolation via Physically Bounded Models | 2026 | `dense_evolution/mitigation/zne.py`, `docs/api/mitigation.md` |
| [2607.25746](https://arxiv.org/abs/2607.25746) | Timon Scheiber | Classically Augmented Zero-Noise Extrapolation | 2026 | `dense_evolution/mitigation/zne.py`, `dense_evolution/noise/density_matrix_channels.py`, `docs/api/mitigation.md` |
| [2607.27465](https://arxiv.org/abs/2607.27465) | Mengqi He, Jing Zhang | IGME: Efficient Chained Method Ensemble for Transferable Semantic Segmentation Attacks | 2026 | `docs/api/ia_utils_adversarial_vector_attack.md`, `tools/ia_utils/adversarial_vector_attack.py` |
| [2608.16716](https://arxiv.org/abs/2608.16716) | Leonardo Massai, Bence Hetényi, Eoin G. Kelly et al. | Engineering two-qubit gates via anisotropic exchange in germanium spin qubits | 2026 | `dense_evolution/circuits/trotter.py`, `dense_evolution/noise/density_matrix_channels.py`, `docs/api/diagram.md`, `docs/api/mitigation.md` and 1 more |
| [2608.20676](https://arxiv.org/abs/2608.20676) | Eric Huang, Daiwei Zhu, Matteo Ippoliti et al. | Continuous-angle logical rotations in the Steane code | 2026 | `dense_evolution/physics/qec.py` |
| [2609.06491](https://arxiv.org/abs/2609.06491) | Christoph Hümmer, Joachim Sicking, Fabian Hüger et al. | Diffuse2Seg: Diffusion Models Can Segment Anything Without Supervision | 2026 | `dense_evolution/qmmm/propagation.py`, `docs/api/qmmm.md` |
| [cond-mat/0108426](https://arxiv.org/abs/cond-mat/0108426) | D. Balamurugan, R. Prasad | Effect of hydrogen on ground state structures of small silicon clusters | 2001 | `docs/api/dashboard_core_hamiltonians.md`, `tools/dashboard/core/hamiltonians.py` |
| [quant-ph/0301063](https://arxiv.org/abs/quant-ph/0301063) | Guifre Vidal | Efficient classical simulation of slightly entangled quantum computations | 2003 | `dense_evolution/backends/mps_numpy.py`, `docs/assets/composer/py/mps_numpy.py` |

## Journal papers and books without an arXiv identifier

Each is cited by name in the docstring of the file named.

- Obara, S., Saika, A. *J. Chem. Phys.* **84**, 3963 (1986): the Obara-Saika recursion for the Gaussian integrals (`dense_evolution/native_hf/`).
- Temme, K., Bravyi, S., Gambetta, J. M. *Phys. Rev. Lett.* **119**, 180509 (2017): Richardson extrapolation and its sample allocation, Eq. 3 (`dense_evolution/mitigation/zne.py`).
- Pople, J. A., Krishnan, R., Schlegel, H. B., Binkley, J. S. *Int. J. Quantum Chem. Symp.* **13**, 225 (1979): the analytic Hartree-Fock gradient, Eqs. 21-22 (`dense_evolution/native_hf/scf.py`, `libcint_gradient.py`).
- Saunders, V. R., Hillier, I. H. *Int. J. Quantum Chem.* **7**, 699 (1973): level shifting of the Fock matrix (`dense_evolution/native_hf/scf.py`).
- Grassl, M., Beth, T., Pellizzari, T. *Phys. Rev. A* **56**, 33 (1997): codes for the quantum erasure channel (`dense_evolution/physics/qec.py`).
- Szabo, A., Ostlund, N. S. *Modern Quantum Chemistry* (book): the Hartree-Fock equations and the oscillation of the SCF iteration (`dense_evolution/native_hf/overlap.py`, `scf.py`).

## How to cite Dense-Evolution itself

See `CITATION.cff` and the DOI badge in `README.md`.
