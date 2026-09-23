# Cryptography Protocols

Quantum cryptography protocols (crypto-q, promoted from [Dense-Evolution-Discovery issue #189](https://github.com/tatopenn-cell/Dense-Evolution-Discovery/issues/189)), built entirely from existing simulator primitives — no new quantum channel was added to the core to support this section.

## BB84

The reference pattern every protocol here follows: `prepare -> channel -> measure -> sift -> QBER`. Validated at `N=5000` rounds, 5 independent seeds: `QBER=0` on a perfect channel, `QBER=2p/3` under depolarizing noise at rate `p`, `QBER=0.25` under an intercept-resend attack — all within ±1σ of theory (worst case ±0.87σ).

## Three-party device-independent conference key agreement (GHZ)

Following Ribeiro, Murta & Wehner 2018 ([arXiv:1708.00798](https://arxiv.org/abs/1708.00798)): a GHZ(3) state and the paper's own "Parity-CHSH" inequality (not Mermin's — verified directly against the paper text). `parity_chsh_win_rate` reaches the quantum maximum `P_win = 0.85355...` to machine precision on an ideal channel, comfortably clearing the classical bound of `0.75`. Two real bugs were found and fixed while implementing this (a fixed test question read from the wrong value; a depolarizing shrink-factor convention mismatch) — see `di_qkd_ghz`'s own module docstring for both, since a number without that history is a number without meaning.

## Multi-round DICKA structure

`dicka_protocol2.run_protocol` runs the full multi-round structure (Appendix Protocol 2 of the same paper) around the GHZ primitives above: round selection, parameter estimation, and the abort decision. It deliberately does **not** report a secure key length — Theorem 4's exact value depends on a numerical optimization (Lemma 3) the source paper never reduces to closed form, and inventing one here would not be a real number. What it reports is everything the protocol actually specifies as a physical procedure, honestly bounded at that.

::: dense_evolution.protocols.bb84

::: dense_evolution.protocols.di_qkd_ghz

::: dense_evolution.protocols.dicka_protocol2
