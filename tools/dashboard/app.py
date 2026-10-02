"""
Dense Evolution - Interactive Dashboard (entrypoint)
------------------------------------------------------------
Sidebar organized into grouped sections instead of a single flat row of
tabs. Base sections (Build / Results / Chemistry / Noise / System) are
always visible; an "Advanced mode" toggle in the sidebar reveals the rest
(Dynamics / Wormhole / QEC / Magic & Divergences / Condensed Matter /
Large Systems / Import Circuit / All functions): a first-time visitor sees
5 sections, not 13.

Every number shown here comes from a run of dense_evolution's engine
(dashboard_core + dense_evolution's public API). A free-text "paste
Qiskit/PennyLane code" importer would need to exec() arbitrary Python, so
the Import Circuit section is limited to a fixed Qiskit fake backend.

Run with:
    pip install streamlit
    streamlit run app.py
"""

import matplotlib
matplotlib.use('Agg')

import json

import numpy as np
import dense_evolution
import streamlit as st

import dashboard_core as dc
import sota_views as sv
import engine_explorer as ex
import circuit_editor as ce
dc.enable_dashboard_precision()

from dense_evolution.mitigation.magic_entropy import magic_entropy
from dense_evolution.mitigation.stabilizer_renyi_entropy import stabilizer_renyi_entropy
from dense_evolution.native_hf.differentiable import build_energy_fn

ANGSTROM_TO_BOHR = 1.8897259886
QEC_DEFAULT_STABILIZERS = "XZZXI\nIXZZX\nXIXZZ\nZXIXZ"


@st.cache_data(show_spinner="Building the molecule catalog (first time only)...")
def _cached_all_molecules():
    return dc.get_all_molecules()


st.set_page_config(
    page_title=f"Dense Evolution v{dense_evolution.__version__} - Dashboard",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️ Dense Evolution — Dashboard")

# ── Guided tour: step overlay, only on the first run of the session ─────
_TOUR_STEPS = [
    (
        "👋 Welcome to Dense Evolution",
        "The sections are on the left: **Build**, **Results**, **Chemistry**, "
        "**Noise**, **System**. Every circuit you run goes through the "
        "library's simulation engine.",
    ),
    (
        "🔀 Simple / Advanced mode",
        "The **\"Advanced mode\"** toggle at the top of the sidebar shows or "
        "hides the specialist sections (Wormhole, QEC, Condensed Matter, ...). "
        "Start in Simple mode: the 5 base sections are enough for a first circuit.",
    ),
    (
        "✍️ The QASM editor",
        "In **Build** you write or paste a circuit in OpenQASM 2.0, the same "
        "standard format Qiskit uses, or drag gates in the Graphical Editor "
        "and load them into the editor.",
    ),
    (
        "▶ Run",
        "The **▶ Run** button in the sidebar runs the circuit on the "
        "statevector simulator (with the shots and seed you choose) and keeps "
        "the result for every other section.",
    ),
    (
        "📊 Results",
        "Open **Results** to see the statevector, probabilities, Q-sphere, "
        "Bloch spheres and entropy of the circuit you just ran.",
    ),
]


@st.dialog("Guided tour")
def _show_tour_dialog():
    step = st.session_state["tour_step"]
    title, body = _TOUR_STEPS[step]
    st.subheader(title)
    st.write(body)
    st.caption(f"Step {step + 1}/{len(_TOUR_STEPS)}")
    col_skip, col_next = st.columns([1, 1])
    if col_skip.button("Skip", width="stretch"):
        st.session_state["tour_dismissed"] = True
        st.rerun()
    next_label = "Done" if step == len(_TOUR_STEPS) - 1 else "Next →"
    if col_next.button(next_label, type="primary", width="stretch"):
        if step == len(_TOUR_STEPS) - 1:
            st.session_state["tour_dismissed"] = True
        else:
            st.session_state["tour_step"] = step + 1
        st.rerun()


if "tour_dismissed" not in st.session_state:
    st.session_state["tour_dismissed"] = False
    st.session_state["tour_step"] = 0
if not st.session_state["tour_dismissed"]:
    _show_tour_dialog()

# ── Non-empty initial state: the Bell state already runs on first load ──
if "result" not in st.session_state:
    try:
        st.session_state["result"] = dc.run_circuit_from_qasm(
            dc.QASM_LIBRARY["Bell state (2 qubit)"], n_shots=1000, seed=42,
        )
        st.session_state["error"] = None
    except Exception as _exc:
        st.session_state["result"] = None
        st.session_state["error"] = str(_exc)

_BASE_SECTIONS = ["Build", "Results", "Chemistry", "Noise", "System"]
_ADVANCED_SECTIONS = [
    "Dynamics", "Wormhole", "QEC", "Magic & Divergences", "Condensed Matter",
    "Large Systems", "Import Circuit", "All functions",
]

def _load_qasm(qasm, go_to_build=False):
    st.session_state["preset_select"] = "Custom"
    st.session_state["qasm_text__Custom"] = qasm
    if go_to_build:
        st.session_state["nav_section"] = "Build"


# ── Health bar, always visible at the top of the sidebar ────────────────
with st.sidebar:
    limits = dc.max_safe_dense_qubits()
    st.caption(
        f"🖥️ {limits['available_mb']:,.0f} / {limits['total_mb']:,.0f} MB free · "
        f"up to **{limits['max_qubits_dense']} qubits** dense on this machine"
    )
    st.divider()
    advanced_mode = st.toggle("Advanced mode", key="advanced_mode")
    all_sections = _BASE_SECTIONS + (_ADVANCED_SECTIONS if advanced_mode else [])
    if st.session_state.get("nav_section") not in all_sections:
        st.session_state["nav_section"] = "Build"
    section = st.radio("Section", all_sections, key="nav_section")
    st.divider()

    if section == "Build":
        st.header("Circuit")
        preset_name = st.selectbox(
            "Preset", ["Custom"] + list(dc.QASM_LIBRARY.keys()), key="preset_select",
        )
        default_qasm = dc.QASM_LIBRARY.get(preset_name, dc.QASM_LIBRARY["Bell state (2 qubit)"])
        qasm_text = st.text_area(
            "OpenQASM 2.0", value=default_qasm, height=220, key=f"qasm_text__{preset_name}",
        )
        n_shots = st.number_input("Shots", min_value=1, max_value=100_000, value=1000, step=100)
        seed = st.number_input("Seed", min_value=0, max_value=2 ** 31 - 1, value=42, step=1)
        with st.expander("Noise & backend"):
            run_noise_model = st.selectbox(
                "Noise model", ["ideal", "depolarizing", "bitflip", "phaseflip",
                                "amplitude_damping", "combined"],
                key="run_noise_model",
            )
            run_noise_p = st.slider(
                "Noise strength (p)", 0.0, 1.0, 0.0, key="run_noise_p",
                disabled=(run_noise_model == "ideal"),
            )
            run_backend = st.selectbox(
                "Backend", ["dense", "mps"], key="run_backend",
                help="mps (Matrix Product State) uses less memory on weakly entangled "
                     "circuits, up to 24 qubits; beyond that, use Large Systems.",
            )
        run_clicked = st.button("▶ Run", type="primary", width="stretch")
    else:
        run_clicked = False

if run_clicked:
    try:
        st.session_state["result"] = dc.run_circuit_from_qasm(
            qasm_text, n_shots=int(n_shots), seed=int(seed),
            noise_model=run_noise_model, noise_p=float(run_noise_p), backend=run_backend,
        )
        st.session_state["error"] = None
    except Exception as exc:
        st.session_state["result"] = None
        st.session_state["error"] = str(exc)

result = st.session_state.get("result")
error = st.session_state.get("error")


# ── BUILD ────────────────────────────────────────────────────────────────
if section == "Build":
    st.caption(
        "Current state: " + (
            f"{result.n_qubits}-qubit circuit already run — open Results, "
            "or add noise in Noise." if result is not None and not error
            else "press ▶ Run in the sidebar to run a new circuit."
        )
    )
    tab_builder, tab_circuit = st.tabs(["Graphical Editor", "Circuit"])

    with tab_builder:
        st.caption(
            "Drag gates onto the grid to build a circuit by hand: a control ● and a "
            "target in the same column form a 2-qubit gate, two × in the same column "
            "form a SWAP."
        )
        if "n_qubits_builder" not in st.session_state:
            try:
                st.session_state["n_qubits_builder"] = min(8, json.loads(st.query_params.get("circuit", "")).get("n", 3))
            except (ValueError, AttributeError):
                st.session_state["n_qubits_builder"] = 3
        n_qubits_builder = st.number_input(
            "Qubits", min_value=1, max_value=8, step=1, key="n_qubits_builder",
        )
        builder_ops = ce.circuit_editor(int(n_qubits_builder))

        st.button("→ Load into the Circuit Editor", on_click=_load_qasm,
                  args=(dc.gate_tuples_to_qasm(dc.ops_to_native_tuples(int(n_qubits_builder), builder_ops),
                                               int(n_qubits_builder)) if builder_ops else None,),
                  disabled=not builder_ops, help="Place at least one gate on the grid first." if not builder_ops else None)

    with tab_circuit:
        if result is None:
            st.info("Press ▶ Run in the sidebar to run the circuit.")
        elif error:
            st.error(f"Circuit run failed: {error}")
        else:
            st.pyplot(dc.draw_circuit_figure(result.ops, result.n_qubits))

    if error:
        st.error(f"Circuit run failed: {error}")


# ── RESULTS ──────────────────────────────────────────────────────────────
elif section == "Results":
    if result is None:
        st.info("No circuit run yet — open Build and press ▶ Run.")
    elif error:
        st.error(f"Circuit run failed: {error}")
    else:
        st.caption(
            f"Current state: {result.n_qubits}-qubit circuit — you can add noise "
            "(Noise) or build a new one (Build)."
        )
        if result.fidelity_vs_ideal is not None or result.backend == "mps":
            cols = st.columns(4)
            i = 0
            if result.fidelity_vs_ideal is not None:
                cols[i].metric("Fidelity vs. ideal", f"{result.fidelity_vs_ideal:.4f}")
                i += 1
            if result.backend == "mps":
                cols[i].metric("Backend", "MPS")
                cols[i + 1].metric("Max bond used", result.mps_max_bond_used)
                cols[i + 2].metric("MPS memory (MB)", f"{result.mps_memory_mb:.2f}")
            if result.fidelity_vs_ideal is not None:
                st.caption("Fidelity = how far the noisy run is from the ideal circuit (1.0 = identical).")
        tab_sv, tab_prob, tab_qsphere, tab_bloch, tab_entropy, tab_steps, tab_pairs, tab_qubits = st.tabs(
            ["Statevector", "Probabilities", "Q-sphere", "Bloch per qubit", "Entropy & mutual information",
             "Step by step", "Qubit pairs", "Qubit summary"]
        )
        with tab_steps:
            if result.n_qubits > 10 or not result.ops:
                st.info("Available for circuits up to 10 qubits with at least one gate.")
            else:
                states = sv.step_states(result.ops, result.n_qubits)
                k = st.slider("Gate", 1, len(result.ops), 1, key="sota_step")
                st.caption(f"Gate {k}: {result.ops[k - 1]} — how each amplitude changes from the previous gate.")
                st.dataframe(sv.difference_rows(states[k - 1], states[k], result.n_qubits), width="stretch")
        with tab_pairs:
            if result.n_qubits < 2 or result.n_qubits > 10:
                st.info("Available from 2 to 10 qubits.")
            else:
                st.pyplot(sv.half_matrix_figure(result.statevector, result.n_qubits))
        with tab_qubits:
            st.dataframe(sv.qubit_table(result.statevector, result.n_qubits), width="stretch")
            st.code(sv.ket_string(result.statevector, result.n_qubits), language=None)
        with tab_sv:
            st.caption(f"{result.n_qubits} qubits — {len(result.statevector)} amplitudes (Qiskit convention)")
            rows = [
                {
                    "state": format(i, f"0{result.n_qubits}b"),
                    "amplitude (re)": float(amp.real),
                    "amplitude (im)": float(amp.imag),
                    "|amplitude|": abs(amp),
                    "phase (rad)": float(np.angle(amp)),
                }
                for i, amp in enumerate(result.statevector)
                if abs(amp) > 1e-10
            ]
            st.dataframe(rows, width="stretch")
        with tab_prob:
            st.caption("Shots sampled from the computed statevector")
            color_by_phase = st.checkbox(
                "Color by phase", key="prob_color_by_phase",
                help="Colors each bar by the complex phase of that state's amplitude "
                     "(cyclic colormap); the bar height stays the shot count.",
            )
            st.pyplot(dc.histogram_figure(
                result.counts, statevector=result.statevector if color_by_phase else None,
            ))
        with tab_qsphere:
            st.pyplot(dc.qsphere_figure(result.statevector))
        with tab_bloch:
            st.caption("One Bloch sphere per qubit, from its reduced density matrix.")
            st.pyplot(dc.bloch_multivector_figure(result.statevector))
        with tab_entropy:
            st.caption(
                "An entangled qubit has <Z>=0 even when its partner has been operated on: "
                "mutual information sees correlations a single expectation value cannot "
                "(no-signaling theorem). Qubit numbering as in Results (Qiskit convention)."
            )
            all_qubits = list(range(result.n_qubits))
            qa_text = st.text_input("Subsystem A (comma-separated indices)", value="0", key="entropy_a")
            qb_text = st.text_input("Subsystem B (comma-separated indices)", value="1" if result.n_qubits > 1 else "", key="entropy_b")
            if st.button("Compute entropy and mutual information"):
                try:
                    qa = [int(x) for x in qa_text.split(",") if x.strip() != ""]
                    qb = [int(x) for x in qb_text.split(",") if x.strip() != ""]
                    if set(qa) & set(qb):
                        raise ValueError("the two subsystems must be disjoint")
                    # Same Qiskit-little-endian -> native-MSB-first conversion
                    # as Magic & Divergences: flip each index before calling
                    # partial_trace/mutual_information, which use dense_evolution's
                    # own convention, not Qiskit's.
                    n = result.n_qubits
                    native_a = [n - 1 - q for q in qa]
                    native_b = [n - 1 - q for q in qb]
                    rho_a = dense_evolution.partial_trace(result.statevector, n, native_a)
                    rho_b = dense_evolution.partial_trace(result.statevector, n, native_b)
                    s_a = dense_evolution.von_neumann_entropy(rho_a)
                    s_b = dense_evolution.von_neumann_entropy(rho_b)
                    i_ab = dense_evolution.mutual_information(result.statevector, n, native_a, native_b)
                    st.session_state["entropy_result"] = (s_a, s_b, i_ab)
                    st.session_state["entropy_error"] = None
                except Exception as exc:
                    st.session_state["entropy_result"] = None
                    st.session_state["entropy_error"] = str(exc)

            entropy_error = st.session_state.get("entropy_error")
            entropy_result = st.session_state.get("entropy_result")
            if entropy_error:
                st.error(f"Error: {entropy_error}")
            elif entropy_result is not None:
                s_a, s_b, i_ab = entropy_result
                col1, col2, col3 = st.columns(3)
                col1.metric("S(A) (nat)", f"{s_a:.4f}")
                col2.metric("S(B) (nat)", f"{s_b:.4f}")
                col3.metric("I(A:B) (nat)", f"{i_ab:.4f}")


# ── CHEMISTRY ────────────────────────────────────────────────────────────
elif section == "Chemistry":
    st.header("Molecules & Hamiltonians")
    molecules = _cached_all_molecules()
    mol_name = st.selectbox("Molecule", list(molecules.keys()), key="mol_select")
    mol = molecules[mol_name]
    st.caption(f"This molecule uses **{mol['n_qubits']} qubits**.")

    if st.button("Build Hamiltonian"):
        try:
            H_dense, n_qubits = dc.build_molecular_hamiltonian(
                mol["symbols"], mol["geometry"], mol["charge"],
            )
            st.session_state["ham_result"] = (H_dense, n_qubits)
            st.session_state["ham_error"] = None
        except Exception as exc:
            st.session_state["ham_result"] = None
            st.session_state["ham_error"] = str(exc)

    ham_error = st.session_state.get("ham_error")
    ham_result = st.session_state.get("ham_result")
    if ham_error:
        st.error(f"Error: {ham_error}")
    elif ham_result is not None:
        H_dense, n_qubits = ham_result
        exact_e = dc.ground_state_energy(H_dense)
        st.metric("Exact reference energy (Hartree)", f"{exact_e:.6f}")

    st.divider()
    st.header("VQE")
    ansatz_type = st.selectbox("Ansatz", ["hardware_efficient", "uccsd"], key="vqe_ansatz")
    n_layers = st.slider("n_layers (hardware_efficient only)", 1, 12, 4, key="vqe_n_layers")
    maxiter = st.slider("Iterations (Adam)", 0, 500, 150, key="vqe_maxiter")
    with st.expander("Advanced"):
        step_size = st.number_input("step_size", value=0.1, format="%.4f", key="vqe_step_size")
        seed_vqe = st.number_input("seed", value=0, step=1, key="vqe_seed")

    if st.button("Run VQE", type="primary"):
        try:
            vqe_result = dc.run_vqe(
                mol["symbols"], mol["geometry"], mol["charge"],
                ansatz_type=ansatz_type, n_layers=int(n_layers), maxiter=int(maxiter),
                step_size=float(step_size), seed=int(seed_vqe),
            )
            st.session_state["vqe_result"] = vqe_result
            st.session_state["vqe_error"] = None
        except Exception as exc:
            st.session_state["vqe_result"] = None
            st.session_state["vqe_error"] = str(exc)

    vqe_error = st.session_state.get("vqe_error")
    vqe_result = st.session_state.get("vqe_result")
    if vqe_error:
        st.error(f"VQE error: {vqe_error}")
    elif vqe_result is not None:
        col1, col2 = st.columns(2)
        col1.metric("VQE energy (Hartree)", f"{vqe_result['vqe_energy_hartree']:.6f}")
        if vqe_result["exact_energy_hartree"] is not None:
            col2.metric("Exact energy (Hartree)", f"{vqe_result['exact_energy_hartree']:.6f}")
        st.line_chart(vqe_result["energy_history"])
        st.caption(
            "The energy converges toward the molecule's ground state: the smaller the gap "
            "to the exact energy, the better the circuit approximates it."
        )
        # Heuristic flattening check, not a formal Barren Plateau detector:
        # BP is a training-time property (gradient variance vanishing with
        # system size); this only looks at whether energy_history's own
        # tail stopped moving, using data run_vqe already returns -- a
        # proxy for "the optimizer may be stuck". Skipped once the energy is
        # within chemical accuracy (1.6 mHa) of the exact value.
        history = np.asarray(vqe_result["energy_history"], dtype=np.float64)
        exact = vqe_result["exact_energy_hartree"]
        reached_exact = exact is not None and abs(vqe_result["vqe_energy_hartree"] - exact) < 1.6e-3
        if len(history) >= 10 and not reached_exact:
            tail = history[-max(5, len(history) // 5):]
            spread = float(np.ptp(history)) or 1.0
            tail_std = float(np.std(np.diff(tail)))
            if tail_std < 1e-4 * spread:
                st.warning(
                    "The curve flattened over the last iterations (almost no energy change): "
                    "possible plateau or local minimum, not necessarily the global minimum. "
                    "Try fewer n_layers or the UCCSD ansatz."
                )
        st.button("→ Load the VQE circuit into the QASM Editor", on_click=_load_qasm,
                  args=(vqe_result["qasm"], True))

        with st.expander("2D energy landscape"):
            if vqe_result["ansatz_type"] != "hardware_efficient" or vqe_result["n_params"] < 2:
                st.info(
                    "Available only for the hardware_efficient ansatz with at least 2 "
                    "parameters: UCCSD maps weights to angles affinely (not one parameter "
                    "per rotation), so 'parameter i' does not correspond to a single angle."
                )
            else:
                st.caption(
                    "The circuit is re-run on a fixed grid around the minimum found by VQE, "
                    "keeping every other parameter fixed: each point is an energy computed "
                    "from scratch, not an interpolation."
                )
                n_p = vqe_result["n_params"]
                col_pi, col_pj = st.columns(2)
                land_i = col_pi.number_input(
                    "Parameter A (index)", min_value=0, max_value=n_p - 1, value=0, key="land_param_i",
                )
                land_j = col_pj.number_input(
                    "Parameter B (index)", min_value=0, max_value=n_p - 1,
                    value=min(1, n_p - 1), key="land_param_j",
                )
                land_res = st.slider("Grid resolution", 5, 41, 21, step=2, key="land_resolution")
                if land_i == land_j:
                    st.warning("Pick two different indices.")
                elif st.button("Compute energy landscape"):
                    base_params = np.asarray(vqe_result["params"], dtype=np.float64)
                    center_i, center_j = base_params[int(land_i)], base_params[int(land_j)]
                    values_i = np.linspace(center_i - np.pi, center_i + np.pi, int(land_res))
                    values_j = np.linspace(center_j - np.pi, center_j + np.pi, int(land_res))
                    try:
                        energies = dc.scan_hardware_efficient_energy_landscape(
                            mol["symbols"], mol["geometry"], mol["charge"], vqe_result["n_layers"],
                            vqe_result["hf_occupation"], base_params, int(land_i), int(land_j),
                            values_i, values_j,
                        )
                        st.session_state["landscape_result"] = (
                            values_i, values_j, energies, int(land_i), int(land_j),
                            center_i, center_j, vqe_result["vqe_energy_hartree"],
                        )
                        st.session_state["landscape_error"] = None
                    except Exception as exc:
                        st.session_state["landscape_result"] = None
                        st.session_state["landscape_error"] = str(exc)

                land_error = st.session_state.get("landscape_error")
                land_data = st.session_state.get("landscape_result")
                if land_error:
                    st.error(f"Error: {land_error}")
                elif land_data is not None:
                    st.pyplot(dc.energy_landscape_figure(*land_data))

    if advanced_mode:
        st.divider()
        with st.expander("Optimize geometry (analytic dE/dR, native_hf)"):
            st.caption(
                "Relaxes the nuclear positions toward the energy minimum with the analytic "
                "gradient dE/dR (native_hf). Different from run_md_trajectory (Dynamics), "
                "which moves the nuclei with Newtonian dynamics at a fixed electronic state."
            )
            n_geom_steps = st.slider("Descent steps", 1, 50, 15, key="geom_opt_steps")
            geom_lr = st.number_input("Step (learning rate, Bohr)", value=0.05, format="%.3f", key="geom_opt_lr")
            if st.button("Optimize geometry"):
                try:
                    import jax
                    from basis_set_exchange.lut import element_Z_from_sym

                    atomic_numbers = [element_Z_from_sym(s) for s in mol["symbols"]]
                    n_electrons = sum(atomic_numbers) - mol["charge"]
                    geometry_bohr = np.asarray(mol["geometry"], dtype=np.float64) * ANGSTROM_TO_BOHR
                    energy_fn = build_energy_fn(
                        atomic_numbers, [float(z) for z in atomic_numbers], n_electrons,
                        "sto-3g", geometry_bohr,
                    )
                    grad_fn = jax.grad(energy_fn)
                    geom = geometry_bohr.copy()
                    history = [float(energy_fn(geom))]
                    for _ in range(int(n_geom_steps)):
                        g = np.asarray(grad_fn(geom))
                        geom = geom - float(geom_lr) * g
                        history.append(float(energy_fn(geom)))
                    st.session_state["geom_opt_result"] = {
                        "history": history,
                        "geometry_angstrom": (geom / ANGSTROM_TO_BOHR).tolist(),
                    }
                    st.session_state["geom_opt_error"] = None
                except Exception as exc:
                    st.session_state["geom_opt_result"] = None
                    st.session_state["geom_opt_error"] = str(exc)

            geom_error = st.session_state.get("geom_opt_error")
            geom_res = st.session_state.get("geom_opt_result")
            if geom_error:
                st.error(f"Error: {geom_error}")
            elif geom_res is not None:
                st.line_chart(geom_res["history"])
                st.caption(
                    f"Energy: {geom_res['history'][0]:.6f} -> {geom_res['history'][-1]:.6f} Hartree "
                    "(the first JAX compilation can take a few minutes)."
                )


# ── NOISE ────────────────────────────────────────────────────────────────
elif section == "Noise":
    st.header("ZNE mitigation")
    if result is None:
        st.info("Needs a circuit that has already run: open Build, press ▶ Run, then come back.")
    else:
        pauli_string = st.text_input(
            "Pauli observable (one character per qubit, e.g. 'Z' or 'ZZ')",
            value="Z" * result.n_qubits, key="zne_pauli",
        )
        noise_model = st.selectbox(
            "Noise model", ["depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined"],
            key="zne_noise_model",
        )
        noise_p = st.slider("Noise strength (p)", 0.0, 0.5, 0.05, key="zne_noise_p")
        extrapolation_method = st.selectbox(
            "Extrapolation method", ["richardson", "polynomial"], key="zne_method",
        )
        n_trials = st.slider("n_trials (Monte Carlo average per point)", 20, 500, 200, key="zne_n_trials")

        if st.button("Apply ZNE mitigation", type="primary"):
            qasm_for_zne = dc.gate_tuples_to_qasm(result.ops, result.n_qubits)
            try:
                zne_result = dc.run_zne_mitigation(
                    qasm_for_zne, pauli_string=pauli_string, noise_model=noise_model,
                    noise_p=float(noise_p), extrapolation_method=extrapolation_method,
                    n_trials=int(n_trials),
                )
                st.session_state["zne_result"] = zne_result
                st.session_state["zne_error"] = None
            except Exception as exc:
                st.session_state["zne_result"] = None
                st.session_state["zne_error"] = str(exc)

        zne_error = st.session_state.get("zne_error")
        zne_result = st.session_state.get("zne_result")
        if zne_error:
            st.error(f"Error: {zne_error}")
        elif zne_result is not None:
            col1, col2 = st.columns(2)
            col1.metric("Ideal value", f"{zne_result.ideal_expectation:.4f}")
            col2.metric("Extrapolated to zero noise", f"{zne_result.zne_extrapolated:.4f}")
            st.pyplot(dc.zne_bar_figure(
                zne_result.noise_factors, zne_result.noisy_expectations, zne_result.noisy_sems,
                zne_result.zne_extrapolated, zne_result.ideal_expectation,
            ))
            st.caption(
                "Each point is the value measured at that noise scale (1x, 2x, 3x...), with "
                "a ±1 SEM error bar (standard error of the mean over the n_trials Monte Carlo "
                "samples, not the spread of single trials); the extrapolation estimates the "
                "zero-noise value."
            )

    st.divider()
    st.header("Channels & physical noise")
    tab_cosmic, tab_osc, tab_dm, tab_heal = st.tabs(
        ["Cosmic-ray burst", "Oscillating noise", "Density-matrix channel", "Heal vector"]
    )
    with tab_cosmic:
        st.caption("Reproduces an event measured on a 26-qubit chip (arXiv:2104.05219).")
        baseline_gamma = st.number_input("baseline_gamma", value=1e-4, format="%.6f", key="cosmic_gamma")
        if st.button("Simulate burst"):
            cosmic_result = dc.run_cosmic_ray_burst(float(baseline_gamma))
            st.bar_chart({str(t): p for t, p in zip(cosmic_result.times_us, cosmic_result.decay_probabilities)})
            st.caption(f"Peak/baseline ratio: {cosmic_result.peak_ratio:.2f}x")
    with tab_osc:
        base_p = st.slider("base_p", 0.0, 0.5, 0.05, key="osc_base_p")
        freq = st.slider("freq", 0.1, 5.0, 1.0, key="osc_freq")
        amp = st.slider("amp", 0.0, 0.5, 0.1, key="osc_amp")
        if st.button("Simulate oscillating noise"):
            osc_result = dc.run_oscillating_noise(float(base_p), float(freq), float(amp))
            st.line_chart({str(f): p for f, p in zip(osc_result.factors, osc_result.p_eff)})
            st.caption("If this curve is not smooth, the smooth-noise assumption ZNE relies on breaks.")
    with tab_dm:
        if result is None:
            st.info("Needs a circuit that has already run.")
        else:
            channel = st.selectbox("Channel", ["global_depolarizing", "amplitude_damping"], key="dm_channel")
            param = st.slider("Channel parameter", 0.0, 1.0, 0.1, key="dm_param")
            if st.button("Apply channel"):
                qasm_for_dm = dc.gate_tuples_to_qasm(result.ops, result.n_qubits)
                try:
                    dm_result = dc.run_density_matrix_channel(qasm_for_dm, channel, float(param))
                    st.bar_chart({
                        "ideal": dm_result.ideal_diagonal, "noisy": dm_result.noisy_diagonal,
                    })
                except Exception as exc:
                    st.error(f"Error: {exc}")
    with tab_heal:
        if vqe_result := st.session_state.get("vqe_result"):
            st.caption("Heals the energy trajectory of the last VQE run in Chemistry.")
            if st.button("Heal vector"):
                try:
                    vectors = np.asarray(vqe_result["energy_history"], dtype=np.float64).reshape(-1, 1)
                    heal_result = dc.run_vector_healing(vectors)
                    st.line_chart({
                        "original": [v[0] for v in vectors.tolist()],
                        "healed": [v[0] for v in heal_result.healed_vectors],
                    })
                    st.caption(
                        f"Correction triggered: {heal_result.fallback_triggered} · "
                        f"reconstruction error: {heal_result.reconstruction_error:.4g}"
                    )
                except ImportError as exc:
                    st.error(f"Requires the ia_utils package: {exc}")
        else:
            st.info("Run a VQE in Chemistry first to get a trajectory to heal.")


# ── DYNAMICS ─────────────────────────────────────────────────────────────
elif section == "Dynamics":
    st.header("QM/MM & MD trajectory")
    molecules = _cached_all_molecules()
    mol_name = st.selectbox("Molecule", list(molecules.keys()), key="dyn_mol_select")

    if st.button("Compute forces"):
        try:
            forces_result = dc.compute_hellmann_feynman_forces(mol_name)
            st.session_state["forces_result"] = forces_result
            st.session_state["forces_error"] = None
        except Exception as exc:
            st.session_state["forces_result"] = None
            st.session_state["forces_error"] = str(exc)

    forces_error = st.session_state.get("forces_error")
    forces_result = st.session_state.get("forces_result")
    if forces_error:
        st.error(f"Error: {forces_error}")
    elif forces_result is not None:
        col1, col2 = st.columns(2)
        col1.metric("Energy (Hartree)", f"{forces_result['energy_hartree']:.6f}")
        col2.metric("Force norm (Hartree/Å)", f"{forces_result['force_norm']:.6f}")

    st.divider()
    n_steps = st.slider("n_steps", 1, 50, 10, key="md_n_steps")
    dt_fs = st.number_input("dt_fs", value=0.5, format="%.3f", key="md_dt_fs")
    recompute = st.checkbox(
        "Recompute the electronic state at every step (much slower, exact forces far "
        "from the starting geometry)", value=False, key="md_recompute",
    )
    if st.button("Start MD trajectory", type="primary"):
        try:
            traj = dc.run_md_trajectory(
                mol_name, int(n_steps), dt_fs=float(dt_fs), recompute_electronic_state=recompute,
            )
            st.session_state["md_traj"] = traj
            st.session_state["md_error"] = None
        except Exception as exc:
            st.session_state["md_traj"] = None
            st.session_state["md_error"] = str(exc)

    md_error = st.session_state.get("md_error")
    md_traj = st.session_state.get("md_traj")
    if md_error:
        st.error(f"Error: {md_error}")
    elif md_traj is not None:
        st.line_chart({"time_fs": md_traj["time_fs"], "energy_hartree": md_traj["energy_hartree"]})
        st.line_chart({"time_fs": md_traj["time_fs"], "force_norm": md_traj["force_norm"]})
        st.caption(
            "Energy and force norm along the trajectory. If the simulation diverges "
            "(nuclei too close), the function stops with an error message."
        )


# ── WORMHOLE ─────────────────────────────────────────────────────────────
elif section == "Wormhole":
    st.header("SYK protocol / Teleportation")
    st.caption(
        "Traversable-wormhole-inspired quantum teleportation (arXiv:2604.10090): the signal "
        "of interest is the DIFFERENCE between positive and negative mu, not a single value."
    )
    n_majorana = st.select_slider("n_majorana", options=[8, 12, 16, 20], value=8, key="wh_n_majorana")
    k_terms = st.slider("k_terms", 4, 20, 10, key="wh_k_terms")
    J = st.number_input("J", value=1.4142, format="%.4f", key="wh_J")
    mu = st.slider("mu", 1.0, 20.0, 12.0, key="wh_mu")
    t0 = st.number_input("t0", value=0.3, format="%.3f", key="wh_t0")
    t1 = st.number_input("t1", value=0.60, format="%.3f", key="wh_t1")
    with_message = st.checkbox("Inject message (with_message)", value=True, key="wh_with_message")
    method = st.selectbox("Method", ["exact", "Trotter"], key="wh_method")

    if st.button("Find a good instance"):
        seed = dc.select_good_instance(int(n_majorana), int(k_terms), float(J))
        st.session_state["wh_seed"] = seed
    seed = st.session_state.get("wh_seed", 61)
    st.caption(f"Seed in use: **{seed}** (Find a good instance searches for the one with the cleanest signal)")

    if st.button("Run protocol", type="primary"):
        fn = dc.run_wormhole_protocol_trotter if method == "Trotter" else dc.run_wormhole_protocol
        try:
            i_plus = fn(int(n_majorana), int(k_terms), float(J), float(mu), float(t0), float(t1), int(seed), with_message)
            i_minus = fn(int(n_majorana), int(k_terms), float(J), -float(mu), float(t0), float(t1), int(seed), with_message)
            st.session_state["wh_result"] = (i_plus, i_minus)
            st.session_state["wh_error"] = None
        except Exception as exc:
            st.session_state["wh_result"] = None
            st.session_state["wh_error"] = str(exc)

    wh_error = st.session_state.get("wh_error")
    wh_result = st.session_state.get("wh_result")
    if wh_error:
        st.error(f"Error: {wh_error}")
    elif wh_result is not None:
        i_plus, i_minus = wh_result
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(4, 3))
        ax.barh(["mu > 0", "mu < 0"], [i_plus, -i_minus], color=["#648fff", "#dc267f"])
        ax.set_xlabel("mutual information (signed)")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(f"I(mu=+{mu:g}) = {i_plus:.5f} · I(mu=-{mu:g}) = {i_minus:.5f}")


# ── QEC ──────────────────────────────────────────────────────────────────
elif section == "QEC":
    st.header("Quantum error correction")
    stabilizers_text = st.text_area(
        "Stabilizer generators (one per line). Default: the [[5,1,3]] code, which corrects "
        "any single-qubit error",
        value=QEC_DEFAULT_STABILIZERS, key="qec_stabilizers",
    )
    stabilizers = [s.strip() for s in stabilizers_text.splitlines() if s.strip()]
    n_qubits_qec = len(stabilizers[0]) if stabilizers else 0
    st.caption(f"Code with {n_qubits_qec} physical qubits, {len(stabilizers)} stabilizers.")

    st.subheader("Syndrome calculator")
    pauli_error = st.text_input(
        f"Pauli error ({n_qubits_qec} characters from I, X, Y, Z)",
        value="X" + "I" * max(n_qubits_qec - 1, 0), key="qec_error",
    )
    if st.button("Compute syndrome"):
        pauli_error = pauli_error.strip().upper()
        if len(pauli_error) != n_qubits_qec or set(pauli_error) - set("IXYZ"):
            st.error(f"The Pauli error needs exactly {n_qubits_qec} characters from I, X, Y, Z (got '{pauli_error}').")
        else:
            try:
                syndrome = dense_evolution.compute_syndrome(pauli_error, stabilizers)
                st.session_state["qec_syndrome"] = syndrome
                st.session_state["qec_syndrome_error"] = pauli_error
            except Exception as exc:
                st.error(f"Error: {exc}")
    if "qec_syndrome" in st.session_state:
        st.write(f"Syndrome: `{st.session_state['qec_syndrome']}`")
        st.caption(
            "Red: stabilizers that anticommute with the error (syndrome bit 1, detected). "
            "Grey: those that commute (bit 0, not detected). The marked qubit is where the "
            "injected error acts."
        )
        st.pyplot(dc.qec_syndrome_map_figure(
            stabilizers, st.session_state["qec_syndrome"], st.session_state["qec_syndrome_error"],
        ))

    st.subheader("Decoding")
    default_observed = ",".join(str(b) for b in st.session_state.get("qec_syndrome", (0, 0, 0, 1)))
    observed_text = st.text_input("Observed syndrome (comma-separated bits)", value=default_observed, key="qec_observed")
    heralded_text = st.text_input("Known erased qubits, optional (e.g. '2')", value="", key="qec_heralded")
    if st.button("Decode"):
        try:
            observed = tuple(int(x) for x in observed_text.split(",") if x.strip() != "")
            heralded = [int(x) for x in heralded_text.split(",") if x.strip() != ""]
            decoded = dense_evolution.decode_with_erasure_fallback(
                observed, heralded, n_qubits_qec, stabilizers,
            )
            st.session_state["qec_decoded"] = decoded
        except Exception as exc:
            st.session_state.pop("qec_decoded", None)
            st.error(f"Error: {exc}")
    if "qec_decoded" in st.session_state:
        decoded = st.session_state["qec_decoded"]
        if decoded is None:
            st.warning(
                "Ambiguous syndrome: more than one minimum-weight correction fits it (for "
                "example a bit-flip code like ZZI/IZZ cannot tell X from Y)."
            )
        else:
            st.success(f"Correction found: `{decoded}`")


# ── MAGIC & DIVERGENCES ──────────────────────────────────────────────────
elif section == "Magic & Divergences":
    st.header("Magic diagnostics")
    if result is None:
        st.info("Needs a circuit that has already run: open Build.")
    else:
        st.metric("Stabilizer Rényi entropy (whole state)", f"{stabilizer_renyi_entropy(result.statevector):.6f} bit")
        st.caption("Zero for every stabilizer state, positive for 'magic' (non-Clifford) states.")

        qubit_idx = st.number_input(
            "Qubit for magic_entropy (single qubit, Qiskit numbering as in Results)",
            min_value=0, max_value=result.n_qubits - 1, value=0, key="magic_qubit",
        )
        # result.statevector is in Qiskit's little-endian order; partial_trace
        # uses dense_evolution's own MSB-first convention -- the two disagree
        # on which physical qubit index N means, so the Qiskit-facing index
        # has to be flipped before calling it.
        native_qubit_idx = result.n_qubits - 1 - int(qubit_idx)
        rho = dense_evolution.partial_trace(result.statevector, result.n_qubits, [native_qubit_idx])
        st.metric(f"magic_entropy (qubit {qubit_idx})", f"{magic_entropy(rho):.6f} bit")
        st.caption("3-copy Key-Unitary construction: 0 for |0>,|1>,|+>,|->,|+i>,|-i>, 0.811 for T and H.")


# ── CONDENSED MATTER ─────────────────────────────────────────────────────
elif section == "Condensed Matter":
    st.header("Solid state (tight-binding)")
    material = st.selectbox("Material", sorted(dense_evolution.VHD_MATERIALS.keys()), key="solid_material")
    if st.button("Compute gap at Γ"):
        gap = dense_evolution.direct_gap_at_gamma(material)
        st.metric(f"Direct gap at Γ — {material}", f"{gap:.4f} eV")
    if st.button("Scan band Γ→X"):
        vbm, vbm_k, cbm, cbm_k, gap = dense_evolution.band_extrema_along_path(
            material, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("VBM (eV)", f"{vbm:.4f}")
        col2.metric("CBM (eV)", f"{cbm:.4f}")
        col3.metric("Band gap (eV)", f"{gap:.4f}")
        st.caption(
            "For indirect-gap materials (e.g. Si) this is the correct number; "
            "\"Compute gap at Γ\" alone would give the wrong value for them."
        )

    if st.button("Plot band structure (Γ→X, interactive)"):
        t_path, bands = dc.scan_bands_along_path(material, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), n_points=201)
        st.session_state["band_structure_result"] = (t_path, bands, material)

    if "band_structure_result" in st.session_state:
        t_path, bands, band_material = st.session_state["band_structure_result"]
        st.plotly_chart(dc.band_structure_figure(t_path, bands, band_material))
        st.caption(
            "All 10 sp3s* bands along Γ→X: hover a point to see the exact energy and its "
            "position on the k-path. Band 3 (red) and band 4 (blue) are the VBM/CBM that "
            "\"Scan band Γ→X\" summarizes in one number."
        )

    st.divider()
    st.header("Hubbard model")
    n_sites = st.slider("n_sites", 2, 6, 4, key="hub_n_sites")
    t_hop = st.number_input("t (hopping)", value=1.0, format="%.3f", key="hub_t")
    U_int = st.number_input("U (on-site repulsion)", value=2.0, format="%.3f", key="hub_U")
    if st.button("Build Hubbard Hamiltonian"):
        try:
            terms = dense_evolution.hubbard_hamiltonian_pauli_terms(int(n_sites), float(t_hop), float(U_int))
            n_qubits_hub = 2 * int(n_sites)
            H_hub = dense_evolution.pauli_hamiltonian_to_matrix(terms, n_qubits_hub)
            e0 = dc.ground_state_energy(H_hub)
            st.metric(f"Ground-state energy ({n_qubits_hub} qubits)", f"{e0:.6f}")
        except Exception as exc:
            st.error(f"Error: {exc}")


# ── LARGE SYSTEMS ────────────────────────────────────────────────────────
elif section == "Large Systems":
    st.header("Large Systems (MPS beyond the dense limit)")
    st.caption(
        f"Above {dc.MPS_DENSE_CONTRACTION_LIMIT} qubits there is no dense statevector array "
        "to build: run_large_circuit_mps finds the k most probable states with EXACT "
        "probabilities (not sampled), without ever materializing the full state. The "
        "Build backend/noise settings do not apply here: it is a separate execution path."
    )
    default_large_qasm = dc.gate_tuples_to_qasm(dense_evolution.ghz_state(30), 30)
    large_qasm_text = st.text_area(
        "OpenQASM 2.0 (large circuit)", value=default_large_qasm, height=200, key="large_qasm_text",
    )
    k_states = st.slider("k (how many most probable states to show)", 1, 100, 32, key="large_k")
    large_seed = st.number_input("Seed", min_value=0, max_value=2 ** 31 - 1, value=42, step=1, key="large_seed")

    if st.button("Run in MPS mode", type="primary"):
        try:
            large_result = dc.run_large_circuit_mps(large_qasm_text, k=int(k_states), seed=int(large_seed))
            st.session_state["large_mps_result"] = large_result
            st.session_state["large_mps_error"] = None
        except Exception as exc:
            st.session_state["large_mps_result"] = None
            st.session_state["large_mps_error"] = str(exc)

    large_error = st.session_state.get("large_mps_error")
    large_result = st.session_state.get("large_mps_result")
    if large_error:
        st.error(f"Error: {large_error}")
    elif large_result is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Qubits", large_result.n_qubits)
        col2.metric("Max bond used", large_result.mps_max_bond_used)
        col3.metric("MPS memory (MB)", f"{large_result.mps_memory_mb:.2f}")
        st.caption(f"Mean truncation error (JSD): {large_result.mps_avg_jsd:.2e}")
        top_states_sorted = sorted(large_result.top_k_states, key=lambda t: -t[1])
        st.dataframe(
            [{"state": bits, "probability": p} for bits, p in top_states_sorted],
            width="stretch",
        )

    with st.expander("Convergence with bond dimension"):
        st.caption(
            "mps_avg_jsd above is a mean truncation error over the whole state: it does not "
            "say whether the observable you care about has settled as the bond grows. Here "
            "the circuit is re-run at increasing bonds to check whether the expectation value "
            "stops changing. If 'chi_used' at the highest bond already equals its cap, the "
            "verdict is 'undecidable': the truncation never had room to show convergence."
        )
        bc_qasm_text = st.text_area(
            "OpenQASM 2.0 (circuit to check)", value=default_large_qasm, height=150, key="bc_qasm_text",
        )
        bc_observable = st.text_input(
            "Observable (Pauli string, e.g. ZIII)", value="Z" + "I" * 29, key="bc_observable",
        )
        bc_bonds_text = st.text_input("Bonds to test (increasing, comma-separated)", value="4,8,16,32", key="bc_bonds")
        bc_tol = st.number_input("Tolerance", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f", key="bc_tol")
        if st.button("Run convergence check"):
            try:
                bonds = [int(b) for b in bc_bonds_text.split(",") if b.strip() != ""]
                bc_result = dc.run_bond_convergence_check(bc_qasm_text, [bc_observable], bonds, tol=float(bc_tol))
                st.session_state["bond_convergence_result"] = bc_result
                st.session_state["bond_convergence_error"] = None
            except Exception as exc:
                st.session_state["bond_convergence_result"] = None
                st.session_state["bond_convergence_error"] = str(exc)

        bc_error = st.session_state.get("bond_convergence_error")
        bc_result = st.session_state.get("bond_convergence_result")
        if bc_error:
            st.error(f"Error: {bc_error}")
        elif bc_result is not None:
            verdict = bc_result.verdicts[0]
            verdict_label = {
                "converged": "✅ converged",
                "not_converged": "⚠️ not_converged",
                "undecidable": "❔ undecidable",
            }[verdict]
            st.metric("Verdict", verdict_label)
            st.dataframe(
                [
                    {
                        "bond": b, "chi_used": c, "avg_jsd": j, "budget_violations": v,
                        "|<P>|": abs(val),
                    }
                    for b, c, j, v, val in zip(
                        bc_result.bonds, bc_result.chi_used, bc_result.avg_jsd,
                        bc_result.budget_violations, bc_result.values[0],
                    )
                ],
                width="stretch",
            )
            st.caption(
                "diffs (change between consecutive bonds): "
                + ", ".join(f"{d:.2e}" for d in bc_result.diffs[0])
            )


# ── IMPORT CIRCUIT ───────────────────────────────────────────────────────
elif section == "Import Circuit":
    st.header("Noise from hardware calibration")
    st.caption(
        "Loads the calibrated error rates of a fixed Qiskit fake backend (FakeSherbrooke, "
        "IBM Eagle 127 qubits). A \"paste Qiskit/PennyLane code\" importer would need to "
        "execute arbitrary Python on the server, so it is not offered."
    )
    if st.button("Load FakeSherbrooke calibration"):
        try:
            from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
            backend = FakeSherbrooke()
            specs = dense_evolution.noise_model_from_qiskit_backend(backend)
            st.dataframe(specs[:50], width="stretch")
            st.caption(f"{len(specs)} (gate, qubit) pairs with a calibrated error rate.")
        except ImportError as exc:
            st.error(f"Requires qiskit-ibm-runtime: {exc}")
        except Exception as exc:
            st.error(f"Error: {exc}")


# ── SYSTEM ───────────────────────────────────────────────────────────────
elif section == "System":
    st.header("Limits of this machine")
    limits = dc.max_safe_dense_qubits()
    col1, col2, col3 = st.columns(3)
    col1.metric("Available RAM", f"{limits['available_mb']:,.0f} MB")
    col2.metric("Total RAM", f"{limits['total_mb']:,.0f} MB")
    col3.metric("Max dense qubits", limits["max_qubits_dense"])
    st.caption(
        f"Safety threshold: {limits['threshold_pct'] * 100:.0f}% of RAM must stay free "
        "after every allocation, computed from this machine's current memory."
    )


# ── ALL ENGINE FUNCTIONS ─────────────────────────────────────────────────
if section == "All functions":
    st.header("All engine functions")
    if result is None:
        st.info("Run a circuit in Build first.")
    else:
        names = [f for f in ex.functions() if ex.connectable(f)]
        fname = st.selectbox("Function", names, key="ex_fn")
        st.caption((ex.inspect.getdoc(getattr(ex.de, fname)) or "").splitlines()[0] if ex.inspect.getdoc(getattr(ex.de, fname)) else "")
        texts = {p: st.text_input(p, ex.example(fname, p), key=f"ex_{fname}_{p}")
                 for p, k, _ in ex.plan2(fname) if k == "literal"}
        if st.button("Run function", key="ex_run"):
            try:
                qasm = dc.gate_tuples_to_qasm(result.ops, result.n_qubits)
                out = getattr(ex.de, fname)(**ex.build_args2(fname, result, qasm, texts=texts))
                if hasattr(out, "savefig"):
                    st.pyplot(out)
                else:
                    st.write(ex.to_display(out))
            except Exception as exc:
                st.error(f"Error: {exc}")
