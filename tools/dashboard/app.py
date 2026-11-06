"""
Dense Evolution - Interactive Dashboard (entrypoint)
------------------------------------------------------------
Two pages, Build and Results. Every analysis is a window of the left panel
with an On/Off toggle, computed by the single Run button on the fixed
circuit: State analysis (entropy, magic, MPS convergence), Noise (ZNE,
density-matrix channel, cosmic-ray burst, oscillating noise, hardware
calibration), Chemistry (VQE, dynamics, vector healing), Protocols (wormhole,
error correction), Condensed matter and the Function explorer; System shows
the limits of the machine. Results has one tab for each window that is On,
in fixed rows (no scrolling: a tab that does not fit goes on the next row). Chemistry On loads the VQE ansatz into the
circuit, Off restores the previous one.

Every number shown here comes from a run of dense_evolution's engine
(dashboard_core + dense_evolution's public API). A free-text "paste
Qiskit/PennyLane code" importer would need to exec() arbitrary Python, so
the hardware-calibration window is limited to a fixed Qiskit fake backend.

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
from circuit_tools import (
    measurement_rows, outcome_rows, qiskit_code, qubits_in_qasm, state_csv, topology_qasm,
    with_input_values,
)
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

_BASE_SECTIONS = ["Build", "Results"]

_THEMES = {
    "Light": {
        "bg": "#ffffff", "side": "#f0f2f6", "fg": "#1f2430", "inp": "#ffffff", "border": "#d0d5dd",
        "accent": "#ff4b4b", "onaccent": "#ffffff", "radius": "8px", "invert": False,
        "page": "#ffffff", "sidebar": "#f0f2f6", "extra": "",
    },
    "Dark": {
        "bg": "#0e1117", "side": "#161b22", "fg": "#e6e6e6", "inp": "#1f2430", "border": "#30363d",
        "accent": "#4da3ff", "onaccent": "#06101c", "radius": "8px", "invert": True,
        "page": "#0e1117", "sidebar": "#161b22", "extra": "",
    },
    "Pink fantasy": {
        "bg": "#fff0f6", "side": "#ffe0f0", "fg": "#5b3a6b", "inp": "#ffffff", "border": "#ffb3d9",
        "accent": "#ff7ab8", "onaccent": "#ffffff", "radius": "18px", "invert": False,
        "page": "linear-gradient(135deg,#fff0f6 0%,#e4f3ff 55%,#fdeaff 100%)",
        "sidebar": "linear-gradient(180deg,#ffd6ea 0%,#d6ecff 100%)",
        "extra": (
            "button[kind=primary]{border-radius:999px!important;box-shadow:0 2px 10px #ff7ab855}"
            "[data-testid=stExpander]{box-shadow:0 2px 12px #8fd3ff55}"
            "h1,h2,h3{color:#c2539a!important}"
        ),
    },
    "Dark fantasy": {
        "bg": "#120816", "side": "#1c0b24", "fg": "#f5e9d8", "inp": "#2a1236", "border": "#ff7a00",
        "accent": "#ff7a00", "onaccent": "#120816", "radius": "10px", "invert": True,
        "page": "radial-gradient(circle at top right,#3a0d52 0%,#120816 60%)",
        "sidebar": "linear-gradient(180deg,#240a30 0%,#0f0614 100%)",
        "extra": (
            "[data-testid=stSidebar]{border-right:1px solid #b026ff}"
            "[data-testid=stExpander]{border-color:#b026ff!important}"
            "h1,h2,h3{color:#ff7a00!important;text-shadow:0 0 10px #ff7a0088}"
            "button[kind=primary]{box-shadow:0 0 14px #ff7a00aa}"
            "button[role=tab][aria-selected=true]{color:#7cfc00!important}"
        ),
    },
}

_THEME_CSS = (
    ":root{--primary-color:@accent@;--secondary-background-color:@side@}"
    "[data-testid=stApp],[data-testid=stAppViewContainer],[data-testid=stHeader]"
    "{background:@page@!important;color:@fg@}"
    "[data-testid=stSidebar],[data-testid=stSidebar]>div{background:@sidebar@!important}"
    "[data-testid=stApp] h1,[data-testid=stApp] h2,[data-testid=stApp] h3,[data-testid=stApp] h4,"
    "[data-testid=stApp] p,[data-testid=stApp] label,[data-testid=stApp] li,"
    "[data-testid=stCaptionContainer],[data-testid=stMetricValue],[data-testid=stMetricLabel],"
    "[data-testid=stWidgetLabel] p{color:@fg@!important}"
    "input,textarea,[data-baseweb=select]>div,[data-baseweb=input],[data-baseweb=textarea]"
    "{background:@inp@!important;color:@fg@!important;border-color:@border@!important}"
    "[data-baseweb=popover] *,[data-baseweb=menu] *{background:@inp@;color:@fg@}"
    "button[kind=primary]{background:@accent@!important;color:@onaccent@!important;border:0!important}"
    "button[kind=secondary]{background:@inp@!important;color:@fg@!important;"
    "border:1px solid @border@!important}"
    "[data-testid=stExpander]{border:1px solid @border@!important;border-radius:@radius@}"
    "button[role=tab]{color:@fg@!important}"
)
_THEME_INVERT = (
    "[data-testid=stDataFrame],[data-testid=stImage] img,[data-testid=stPlotlyChart],"
    "[data-testid=stCustomComponentV1]{filter:invert(0.9) hue-rotate(180deg)}"
)


def _theme_css(name):
    t = _THEMES[name]
    css = _THEME_CSS + (_THEME_INVERT if t["invert"] else "") + t["extra"]
    for key, value in t.items():
        css = css.replace(f"@{key}@", str(value))
    return f"<style>{css}</style>"



if "theme" not in st.session_state:
    wanted = st.query_params.get("theme", "Light")
    st.session_state["theme"] = wanted if wanted in _THEMES else "Light"

col_title, col_nav, col_theme = st.columns([4, 3, 2])
col_title.subheader("⚛️ Dense Evolution — Dashboard")
with col_theme:
    theme_name = st.selectbox("Theme", list(_THEMES), key="theme", label_visibility="collapsed")
if st.query_params.get("theme") != theme_name:
    st.query_params["theme"] = theme_name
st.markdown(_theme_css(theme_name), unsafe_allow_html=True)
if st.session_state.get("nav_section") not in _BASE_SECTIONS:
    st.session_state["nav_section"] = "Build"
with col_nav:
    section = st.radio(
        "Section", _BASE_SECTIONS, key="nav_section", horizontal=True,
        label_visibility="collapsed",
    )

# ── Guided tour: step overlay, only on the first run of the session ─────
_TOUR_STEPS = [
    (
        "👋 Welcome to Dense Evolution",
        "**Build** is where you make the circuit, **Results** shows what it does. "
        "On the left, the windows (Noise analysis, Chemistry, Wormhole, ...) have an "
        "On/Off toggle: the ones that are On run together with the circuit. Every "
        "circuit you run goes through the library's simulation engine.",
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

def _load_qasm(qasm, go_to_build=False):
    st.session_state["preset_select"] = "Custom"
    st.session_state["qasm_text__Custom"] = qasm
    if go_to_build:
        st.session_state["nav_section"] = "Build"


@st.cache_data
def _connectable_functions():
    return [name for name in ex.functions() if ex.connectable(name)]


def _memo(key, job):
    cache = st.session_state.setdefault("_view_cache", {})
    if key not in cache:
        cache[key] = job()
    return cache[key]


def _png(fig, width=None, height=None):
    import io

    import matplotlib.pyplot as plt

    if width:
        fig.set_size_inches(width, height)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _ran(key):
    return key in st.session_state.get("ran_with", set())


def _param(key):
    return st.session_state.get("ran_params", {}).get(key, st.session_state.get(key))


def _press_run():
    st.info("Press ▶ Run to compute this window.")


def _qsphere_figure(res):
    sphere = dc.qsphere_figure(res.statevector)
    sphere.set_size_inches(7, 7)
    sphere.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return sphere


def _show_in_editor(res):
    if res is None:
        return
    n = res.n_qubits
    grid, skipped = ce.grid_from_ops(res.ops, n)
    steps = len(grid[0])
    history = st.session_state.get("ce_hist")
    if history:
        current = history[st.session_state["ce_idx"]]
        drawn = any(cell for row in current for cell in row)
        if drawn and current != st.session_state.get("ce_installed"):
            return
    if n > 8 or steps > 24:
        st.session_state["editor_note"] = (
            f"The editor shows up to 8 qubits and 24 steps; this circuit has {n} qubits and "
            f"{steps} steps."
        )
        return
    st.session_state["editor_note"] = (
        f"{skipped} gate(s) cannot be drawn on the grid (for example id or u3); they still run."
        if skipped else None
    )
    st.session_state.update(
        editor_pending={"n": n, "steps": steps}, ce_n=n, ce_hist=[grid], ce_idx=0,
        ce_version=st.session_state.get("ce_version", 0) + 1, ce_start_version=None,
        ce_installed=grid,
    )


def _current_qasm():
    preset = st.session_state.get("preset_select", "Custom")
    default = dc.QASM_LIBRARY.get(preset, dc.QASM_LIBRARY["Bell state (2 qubit)"])
    return st.session_state.get(f"qasm_text__{preset}", default)


def _on_run():
    if not st.session_state.get("chem_on"):
        saved = st.session_state.get("chem_saved_qasm")
        if saved is not None:
            _load_qasm(saved)
            st.session_state["chem_saved_qasm"] = None
        return
    mol = _cached_all_molecules()[st.session_state["mol_select"]]
    try:
        vqe = dc.run_vqe(
            mol["symbols"], mol["geometry"], mol["charge"],
            ansatz_type=st.session_state["vqe_ansatz"], n_layers=int(st.session_state["vqe_n_layers"]),
            maxiter=int(st.session_state["vqe_maxiter"]),
            step_size=float(st.session_state["vqe_step_size"]), seed=int(st.session_state["vqe_seed"]),
        )
        st.session_state["vqe_result"] = vqe
        st.session_state["vqe_error"] = None
        if st.session_state.get("chem_saved_qasm") is None:
            st.session_state["chem_saved_qasm"] = _current_qasm()
        _load_qasm(vqe["qasm"])
    except Exception as exc:
        st.session_state["vqe_result"] = None
        st.session_state["vqe_error"] = str(exc)


def _store(key, job):
    st.session_state[f"{key}_result"] = None
    st.session_state[f"{key}_error"] = None
    try:
        st.session_state[f"{key}_result"] = job()
    except Exception as exc:
        st.session_state[f"{key}_error"] = str(exc)


def _job_burst():
    res = st.session_state.get("result")
    if res is None:
        raise ValueError("run a dense circuit first")
    gamma = float(st.session_state["cosmic_gamma"])
    burst = dc.run_cosmic_ray_burst(gamma)
    rows = []
    for t, prob in zip(burst.times_us, burst.decay_probabilities):
        noisy = dc.run_circuit_from_qasm(
            st.session_state["_run_qasm"], n_shots=int(st.session_state["_run_shots"]),
            seed=int(st.session_state["_run_seed"]), noise_model="amplitude_damping",
            noise_p=min(float(prob), 1.0),
        )
        rows.append({"time_us": t, "decay_probability": prob, "fidelity": noisy.fidelity_vs_ideal})
    return {"rows": rows, "peak_ratio": burst.peak_ratio}


def _job_oscillating():
    base = float(st.session_state["zne_noise_p"])
    return dc.run_oscillating_noise(base, float(st.session_state["osc_freq"]), float(st.session_state["osc_amp"]))


def _job_calibration():
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    return dense_evolution.noise_model_from_qiskit_backend(FakeSherbrooke())


def _job_heal():
    vqe = st.session_state.get("vqe_result")
    if vqe is None:
        raise ValueError("turn Chemistry On and press ▶ Run first: it gives the trajectory to heal")
    vectors = np.asarray(vqe["energy_history"], dtype=np.float64).reshape(-1, 1)
    return {"original": vectors[:, 0].tolist(), "healed": dc.run_vector_healing(vectors)}


def _job_function():
    res = st.session_state.get("result")
    if res is None:
        raise ValueError("run a dense circuit first")
    name = st.session_state["ex_fn"]
    texts = {
        p_name: st.session_state.get(f"ex_{name}_{p_name}", ex.example(name, p_name))
        for p_name, kind, _ in ex.plan2(name) if kind == "literal"
    }
    qasm = dc.gate_tuples_to_qasm(res.ops, res.n_qubits)
    out = getattr(ex.de, name)(**ex.build_args2(name, res, qasm, texts=texts))
    return {"name": name, "doc": (ex.inspect.getdoc(getattr(ex.de, name)) or "").splitlines()[:1], "out": out}


def _job_landscape():
    vqe = st.session_state.get("vqe_result")
    if vqe is None or vqe["ansatz_type"] != "hardware_efficient" or vqe["n_params"] < 2:
        raise ValueError("needs a hardware_efficient VQE with at least 2 parameters (Chemistry On)")
    mol = _cached_all_molecules()[st.session_state["mol_select"]]
    n_p = vqe["n_params"]
    i, j = min(int(st.session_state["land_param_i"]), n_p - 1), min(int(st.session_state["land_param_j"]), n_p - 1)
    if i == j:
        raise ValueError("pick two different parameter indices")
    base = np.asarray(vqe["params"], dtype=np.float64)
    res = int(st.session_state["land_resolution"])
    values_i = np.linspace(base[i] - np.pi, base[i] + np.pi, res)
    values_j = np.linspace(base[j] - np.pi, base[j] + np.pi, res)
    energies = dc.scan_hardware_efficient_energy_landscape(
        mol["symbols"], mol["geometry"], mol["charge"], vqe["n_layers"], vqe["hf_occupation"],
        base, i, j, values_i, values_j,
    )
    return (values_i, values_j, energies, i, j, base[i], base[j], vqe["vqe_energy_hartree"])


def _job_geometry():
    import jax
    from basis_set_exchange.lut import element_Z_from_sym

    mol = _cached_all_molecules()[st.session_state["mol_select"]]
    atomic_numbers = [element_Z_from_sym(sym) for sym in mol["symbols"]]
    n_electrons = sum(atomic_numbers) - mol["charge"]
    geometry_bohr = np.asarray(mol["geometry"], dtype=np.float64) * ANGSTROM_TO_BOHR
    energy_fn = build_energy_fn(
        atomic_numbers, [float(z) for z in atomic_numbers], n_electrons, "sto-3g", geometry_bohr,
    )
    grad_fn = jax.grad(energy_fn)
    geom = geometry_bohr.copy()
    history = [float(energy_fn(geom))]
    for _ in range(int(st.session_state["geom_opt_steps"])):
        geom = geom - float(st.session_state["geom_opt_lr"]) * np.asarray(grad_fn(geom))
        history.append(float(energy_fn(geom)))
    return {"history": history}


def _job_dynamics():
    name = st.session_state["mol_select"]
    return {
        "forces": dc.compute_hellmann_feynman_forces(name),
        "traj": dc.run_md_trajectory(
            name, int(st.session_state["md_n_steps"]), dt_fs=float(st.session_state["md_dt_fs"]),
            recompute_electronic_state=bool(st.session_state.get("md_recompute")),
        ),
    }


def _job_wormhole():
    if st.session_state.get("wh_find_on"):
        st.session_state["wh_seed"] = dc.select_good_instance(
            int(st.session_state["wh_n_majorana"]), int(st.session_state["wh_k_terms"]),
            float(st.session_state["wh_J"]),
        )
    fn = dc.run_wormhole_protocol_trotter if st.session_state["wh_method"] == "Trotter" else dc.run_wormhole_protocol
    args = (
        int(st.session_state["wh_n_majorana"]), int(st.session_state["wh_k_terms"]), float(st.session_state["wh_J"]),
    )
    rest = (
        float(st.session_state["wh_t0"]), float(st.session_state["wh_t1"]),
        int(st.session_state.get("wh_seed", 61)), bool(st.session_state["wh_with_message"]),
    )
    mu = float(st.session_state["wh_mu"])
    return fn(*args, mu, *rest), fn(*args, -mu, *rest), mu


def _job_qec():
    stabilizers = [line.strip() for line in st.session_state["qec_stabilizers"].splitlines() if line.strip()]
    n = len(stabilizers[0]) if stabilizers else 0
    error = (st.session_state.get("qec_pauli") or "X" + "I" * max(n - 1, 0)).strip().upper()
    if len(error) != n or set(error) - set("IXYZ"):
        raise ValueError(f"the Pauli error needs exactly {n} characters from I, X, Y, Z (got '{error}')")
    syndrome = dense_evolution.compute_syndrome(error, stabilizers)
    observed_text = st.session_state.get("qec_observed", "")
    observed = tuple(int(x) for x in observed_text.split(",") if x.strip()) or tuple(syndrome)
    heralded = [int(x) for x in st.session_state.get("qec_heralded", "").split(",") if x.strip()]
    decoded = dense_evolution.decode_with_erasure_fallback(observed, heralded, n, stabilizers)
    return {"stabilizers": stabilizers, "error": error, "syndrome": syndrome, "decoded": decoded}


def _job_condensed():
    material = st.session_state["solid_material"]
    start, end = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    t_path, bands = dc.scan_bands_along_path(material, start, end, n_points=201)
    return {
        "material": material,
        "gap_gamma": dense_evolution.direct_gap_at_gamma(material),
        "extrema": dense_evolution.band_extrema_along_path(material, start, end),
        "bands": (t_path, bands),
    }


def _job_hubbard():
    n_sites = int(st.session_state["hub_n_sites"])
    terms = dense_evolution.hubbard_hamiltonian_pauli_terms(
        n_sites, float(st.session_state["hub_t"]), float(st.session_state["hub_U"]),
    )
    h_hub = dense_evolution.pauli_hamiltonian_to_matrix(terms, 2 * n_sites)
    return 2 * n_sites, dc.ground_state_energy(h_hub)



with st.sidebar:
    st.markdown(
        '<style>[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{gap:0.25rem}'
        '[data-testid="stSidebarUserContent"]{padding-top:1rem}'
        '.st-key-run_button{position:sticky;top:0;z-index:10;'
        'background:var(--secondary-background-color,#f0f2f6)}</style>',
        unsafe_allow_html=True,
    )
    preset_name = st.selectbox(
        "Preset", ["Custom"] + list(dc.QASM_LIBRARY.keys()), key="preset_select",
    )
    with st.form("run_form", border=False):
        run_clicked = st.form_submit_button(
            "▶ Run", type="primary", width="stretch", key="run_button", on_click=_on_run,
        )
        default_qasm = dc.QASM_LIBRARY.get(preset_name, dc.QASM_LIBRARY["Bell state (2 qubit)"])
        qasm_text = st.text_area(
            "OpenQASM 2.0", value=default_qasm, height=120, key=f"qasm_text__{preset_name}",
        )
        col_shots, col_seed = st.columns(2)
        n_shots = col_shots.number_input("Shots", min_value=1, max_value=100_000, value=1000, step=100)
        seed = col_seed.number_input("Seed", min_value=0, max_value=2 ** 31 - 1, value=42, step=1)
        n_input = qubits_in_qasm(qasm_text)
        st.session_state["input_ones"] = [
            i for i in st.session_state.get("input_ones", []) if i < n_input
        ]
        st.multiselect(
            "Qubits that start in |1⟩", list(range(n_input)), key="input_ones",
            label_visibility="collapsed", placeholder="Qubits that start in |1⟩ (none: all in |0⟩)",
        )
        with st.expander("Noise & backend"):
            run_noise_model = st.selectbox(
                "Noise model", ["ideal", "depolarizing", "bitflip", "phaseflip",
                                "amplitude_damping", "combined"],
                key="run_noise_model",
            )
            run_noise_p = st.slider(
                "Noise strength (p)", 0.0, 1.0, 0.0, key="run_noise_p",
                help="Not used when the noise model is ideal.",
            )
            run_backend = st.selectbox(
                "Backend", ["dense", "mps"], key="run_backend",
                help="mps (Matrix Product State) uses less memory on weakly entangled "
                     "circuits. Beyond the dense limit it lists the k most probable states.",
            )
            st.slider("k (most probable states beyond the dense limit)", 1, 100, 32, key="large_k")
        with st.expander("State analysis"):
            previous_result = st.session_state.get("result")
            st.toggle("Entropy & mutual information", key="entropy_on")
            st.text_input("Subsystem A (comma-separated indices)", value="0", key="entropy_a")
            st.text_input(
                "Subsystem B (comma-separated indices)",
                value="1" if previous_result is not None and previous_result.n_qubits > 1 else "",
                key="entropy_b",
            )
            st.toggle("Magic", key="magic_on")
            st.number_input(
                "Magic: qubit (Qiskit numbering)", min_value=0, max_value=max(n_input - 1, 0), step=1,
                key="magic_qubit",
            )
            st.toggle("Exact probabilities (not sampled shots)", key="prob_exact")
            st.toggle("Step by step", key="steps_on")
            st.number_input("Gate (step by step)", min_value=1, value=1, step=1, key="sota_step")
            st.toggle("MPS convergence", key="bc_on")
            st.text_input("Pauli observable (empty: Z on qubit 0)", key="bc_observable")
            st.text_input("Bonds (increasing, comma-separated)", value="4,8,16,32", key="bc_bonds")
            st.number_input("Tolerance", 1e-6, 1.0, 1e-3, format="%.6f", key="bc_tol")
        with st.expander("Noise"):
            st.toggle("ZNE mitigation", key="zne_on")
            st.text_input("Pauli observable (empty: Z on every qubit)", key="zne_pauli")
            st.selectbox(
                "Noise model", ["depolarizing", "bitflip", "phaseflip", "amplitude_damping", "combined"],
                key="zne_noise_model",
            )
            st.slider("Noise strength (p)", 0.0, 0.5, 0.05, key="zne_noise_p")
            st.selectbox("Extrapolation method", ["richardson", "polynomial"], key="zne_method")
            st.slider("Monte Carlo trials per point", 20, 500, 200, key="zne_n_trials")
            st.toggle("Density-matrix channel", key="dm_on")
            st.selectbox("Channel", ["global_depolarizing", "amplitude_damping"], key="dm_channel")
            st.slider("Channel parameter", 0.0, 1.0, 0.1, key="dm_param")
            st.toggle("Cosmic-ray burst on the circuit", key="burst_on")
            st.number_input(
                "Baseline decay probability", 0.0, 1.0, 0.01, format="%.4f", key="cosmic_gamma",
            )
            st.toggle("Oscillating noise (on the ZNE scale factors)", key="osc_on")
            st.slider("freq", 0.1, 5.0, 1.0, key="osc_freq")
            st.slider("amp", 0.0, 0.5, 0.1, key="osc_amp")
            st.toggle("Hardware calibration (FakeSherbrooke)", key="cal_on")
        with st.expander("Chemistry"):
            st.toggle("On: the circuit becomes the VQE ansatz", key="chem_on")
            st.selectbox("Molecule", list(_cached_all_molecules().keys()), key="mol_select")
            st.selectbox("Ansatz", ["hardware_efficient", "uccsd"], key="vqe_ansatz")
            st.slider("n_layers (hardware_efficient only)", 1, 12, 4, key="vqe_n_layers")
            st.slider("Iterations (Adam)", 0, 500, 150, key="vqe_maxiter")
            st.number_input("step_size", value=0.1, format="%.4f", key="vqe_step_size")
            st.number_input("VQE seed", value=0, step=1, key="vqe_seed")
            st.toggle("Energy landscape (two parameters)", key="chem_land_on")
            st.number_input("Parameter A (index)", min_value=0, value=0, step=1, key="land_param_i")
            st.number_input("Parameter B (index)", min_value=0, value=1, step=1, key="land_param_j")
            st.slider("Grid resolution", 5, 41, 21, step=2, key="land_resolution")
            st.toggle("Optimize geometry (analytic dE/dR)", key="chem_geom_on")
            st.slider("Descent steps", 1, 50, 15, key="geom_opt_steps")
            st.number_input("Step (Bohr)", value=0.05, format="%.3f", key="geom_opt_lr")
            st.toggle("Dynamics (forces and MD)", key="dyn_on")
            st.slider("MD steps", 1, 50, 10, key="md_n_steps")
            st.number_input("dt_fs", value=0.5, format="%.3f", key="md_dt_fs")
            st.checkbox("Recompute the electronic state at every step (slower)", key="md_recompute")
            st.toggle("Heal the VQE energy trajectory", key="heal_on")
        with st.expander("Protocols"):
            st.toggle("Wormhole", key="wh_on")
            st.select_slider("n_majorana", options=[8, 12, 16, 20], value=8, key="wh_n_majorana")
            st.slider("k_terms", 4, 20, 10, key="wh_k_terms")
            st.number_input("J", value=1.4142, format="%.4f", key="wh_J")
            st.slider("mu", 1.0, 20.0, 12.0, key="wh_mu")
            st.number_input("t0", value=0.3, format="%.3f", key="wh_t0")
            st.number_input("t1", value=0.60, format="%.3f", key="wh_t1")
            st.checkbox("Inject message", value=True, key="wh_with_message")
            st.selectbox("Method", ["exact", "Trotter"], key="wh_method")
            st.checkbox("Look for a good instance first (cleanest signal)", key="wh_find_on")
            st.caption(f"Seed in use: {st.session_state.get('wh_seed', 61)}")
            st.toggle("Error correction", key="qec_on")
            st.text_area(
                "Stabilizer generators (one per line; default: the [[5,1,3]] code)",
                value=QEC_DEFAULT_STABILIZERS, height=100, key="qec_stabilizers",
            )
            st.text_input("Pauli error (empty: X on qubit 0)", key="qec_pauli")
            st.text_input("Observed syndrome (empty: the computed one)", key="qec_observed")
            st.text_input("Known erased qubits, optional (e.g. 2)", key="qec_heralded")
        with st.expander("Condensed matter"):
            st.toggle("Band structure (tight-binding)", key="cm_on")
            st.selectbox("Material", sorted(dense_evolution.VHD_MATERIALS.keys()), key="solid_material")
            st.toggle("Hubbard model", key="hub_on")
            st.slider("n_sites", 2, 6, 4, key="hub_n_sites")
            st.number_input("t (hopping)", value=1.0, format="%.3f", key="hub_t")
            st.number_input("U (on-site repulsion)", value=2.0, format="%.3f", key="hub_U")
        with st.expander("Function explorer"):
            st.toggle("On", key="fn_on")
            fn_name = st.selectbox("Function", _connectable_functions(), key="ex_fn")
            for p_name, kind, _ in ex.plan2(fn_name):
                if kind == "literal":
                    st.text_input(p_name, ex.example(fn_name, p_name), key=f"ex_{fn_name}_{p_name}")
    with st.expander("Topology"):
        pattern_labels = {
            "linear": "linear (chain)",
            "circular": "circular (ring)",
            "full": "full (all-to-all)",
            "star": "star (hub)",
            "brick": "brick (brickwork)",
        }
        topo_pattern = st.selectbox(
            "Pattern", list(pattern_labels), format_func=pattern_labels.get, key="topo_pattern",
        )
        topo_qubits = st.number_input("Qubits", 2, 12, 5, step=1, key="topo_qubits")
        topo_gate = st.selectbox("Gate", ["cx", "cz", "cy"], key="topo_gate")
        topo_reverse = st.checkbox("Reverse direction", key="topo_reverse")
        topo_hub = 0
        if topo_pattern == "star":
            if "topo_hub" in st.session_state:
                st.session_state["topo_hub"] = min(st.session_state["topo_hub"], int(topo_qubits) - 1)
            topo_hub = st.number_input(
                "Hub qubit", min_value=0, max_value=int(topo_qubits) - 1, value=0, step=1, key="topo_hub",
            )
        topo_h = st.checkbox("Hadamard on every qubit first", value=True, key="topo_h")
        try:
            topo_text = topology_qasm(
                int(topo_qubits), topo_pattern, gate=topo_gate, reverse=topo_reverse,
                hub=int(topo_hub), hadamards=topo_h,
            )
            topo_edges = [
                (a, b) for _, a, b in dense_evolution.entangling_layer(
                    int(topo_qubits), pattern=topo_pattern, gate=topo_gate,
                    reverse=topo_reverse, hub=int(topo_hub),
                )
            ]
            st.caption(f"{len(topo_edges)} edges: " + ", ".join(f"({a},{b})" for a, b in topo_edges))
        except ValueError as exc:
            topo_text = None
            st.error(str(exc))
        st.button(
            "→ Load into the Circuit Editor", on_click=_load_qasm, args=(topo_text,),
            disabled=topo_text is None, key="topo_load",
        )
    with st.expander("Export"):
        shown = st.session_state.get("result")
        st.download_button(
            "Circuit as OpenQASM", _current_qasm(), "circuit.qasm", disabled=shown is None,
            key="export_qasm",
        )
        try:
            code = (
                _memo(("qiskit",), lambda: qiskit_code(shown.ops, shown.n_qubits))
                if shown is not None else ""
            )
        except ValueError:
            code = ""
        st.download_button(
            "Circuit as Qiskit code", code, "circuit_qiskit.py", disabled=not code, key="export_qiskit",
        )
        st.download_button(
            "Final state and probabilities (CSV)",
            _memo(("state_csv",), lambda: state_csv(shown.statevector, shown.n_qubits))
            if shown is not None else "",
            "state.csv", disabled=shown is None, key="export_state",
        )
    with st.expander("System"):
        limits = dc.max_safe_dense_qubits()
        st.caption(
            f"🖥️ {limits['available_mb']:,.0f} / {limits['total_mb']:,.0f} MB free · up to "
            f"**{limits['max_qubits_dense']} qubits** dense · "
            f"{limits['threshold_pct'] * 100:.0f}% of RAM stays free after every allocation"
        )

if run_clicked:
    st.session_state["_view_cache"] = {}
    st.session_state["ran_with"] = {
        key for key in ("magic_on", "entropy_on", "steps_on", "prob_exact") if st.session_state.get(key)
    }
    st.session_state["ran_params"] = {
        key: st.session_state.get(key) for key in ("magic_qubit", "entropy_a", "entropy_b", "sota_step")
    }
    run_qasm = with_input_values(qasm_text, st.session_state.get("input_ones", []))
    st.session_state["_run_qasm"] = run_qasm
    st.session_state["_run_shots"] = int(n_shots)
    st.session_state["_run_seed"] = int(seed)
    st.session_state["large_mps_result"] = None
    try:
        if run_backend == "mps" and n_input > dc.MPS_DENSE_CONTRACTION_LIMIT:
            st.session_state["large_mps_result"] = dc.run_large_circuit_mps(
                run_qasm, k=int(st.session_state["large_k"]), seed=int(seed),
            )
            st.session_state["result"] = None
        else:
            st.session_state["result"] = dc.run_circuit_from_qasm(
                run_qasm, n_shots=int(n_shots), seed=int(seed),
                noise_model=run_noise_model, noise_p=float(run_noise_p), backend=run_backend,
            )
        st.session_state["error"] = None
    except Exception as exc:
        st.session_state["result"] = None
        st.session_state["error"] = str(exc)

    run_result = st.session_state["result"]
    for key, active, job in (
        ("zne", st.session_state.get("zne_on"), lambda: dc.run_zne_mitigation(
            dc.gate_tuples_to_qasm(run_result.ops, run_result.n_qubits),
            pauli_string=st.session_state.get("zne_pauli") or "Z" * run_result.n_qubits,
            noise_model=st.session_state["zne_noise_model"],
            noise_p=float(st.session_state["zne_noise_p"]),
            extrapolation_method=st.session_state["zne_method"],
            n_trials=int(st.session_state["zne_n_trials"]),
        )),
        ("dm", st.session_state.get("dm_on"), lambda: dc.run_density_matrix_channel(
            dc.gate_tuples_to_qasm(run_result.ops, run_result.n_qubits),
            st.session_state["dm_channel"], float(st.session_state["dm_param"]),
        )),
        ("bond_convergence", st.session_state.get("bc_on"), lambda: dc.run_bond_convergence_check(
            run_qasm,
            [st.session_state.get("bc_observable") or "Z" + "I" * (n_input - 1)],
            [int(b) for b in st.session_state["bc_bonds"].split(",") if b.strip() != ""],
            tol=float(st.session_state["bc_tol"]),
        )),
    ):
        st.session_state[f"{key}_result"] = None
        st.session_state[f"{key}_error"] = None
        if not active:
            continue
        if key != "bond_convergence" and run_result is None:
            st.session_state[f"{key}_error"] = "run a dense circuit first"
            continue
        try:
            st.session_state[f"{key}_result"] = job()
        except Exception as exc:
            st.session_state[f"{key}_error"] = str(exc)

    for key, active, job in (
        ("land", st.session_state.get("chem_on") and st.session_state.get("chem_land_on"), _job_landscape),
        ("geom", st.session_state.get("chem_geom_on"), _job_geometry),
        ("dyn", st.session_state.get("dyn_on"), _job_dynamics),
        ("wh", st.session_state.get("wh_on"), _job_wormhole),
        ("qec", st.session_state.get("qec_on"), _job_qec),
        ("cm", st.session_state.get("cm_on"), _job_condensed),
        ("hub", st.session_state.get("hub_on"), _job_hubbard),
        ("burst", st.session_state.get("burst_on"), _job_burst),
        ("osc", st.session_state.get("osc_on"), _job_oscillating),
        ("cal", st.session_state.get("cal_on"), _job_calibration),
        ("heal", st.session_state.get("heal_on"), _job_heal),
        ("fn", st.session_state.get("fn_on"), _job_function),
    ):
        if active:
            _store(key, job)
        else:
            st.session_state[f"{key}_result"] = None
            st.session_state[f"{key}_error"] = None

result = st.session_state.get("result")
error = st.session_state.get("error")
large_result = st.session_state.get("large_mps_result")
if run_clicked or "ce_installed" not in st.session_state:
    _show_in_editor(result)


def _show_large(large):
    col1, col2, col3 = st.columns(3)
    col1.metric("Qubits", large.n_qubits)
    col2.metric("Max bond used", large.mps_max_bond_used)
    col3.metric("MPS memory (MB)", f"{large.mps_memory_mb:.2f}")
    st.caption(f"Mean truncation error (JSD): {large.mps_avg_jsd:.2e}")
    top_states = sorted(large.top_k_states, key=lambda t: -t[1])
    st.dataframe([{"state": bits, "probability": p} for bits, p in top_states], width="stretch")


def _show_analysis(name):
    err = st.session_state.get(f"{name}_error")
    out = st.session_state.get(f"{name}_result")
    if err:
        st.error(f"Error: {err}")
    elif out is None:
        st.info("Turn it On in the left panel and press ▶ Run.")
    return out


def _show_zne():
    out = _show_analysis("zne")
    if out is None:
        return
    col1, col2 = st.columns(2)
    col1.metric("Ideal value", f"{out.ideal_expectation:.4f}")
    col2.metric("Extrapolated to zero noise", f"{out.zne_extrapolated:.4f}")
    st.image(_memo(("zne",), lambda: _png(dc.zne_bar_figure(
        out.noise_factors, out.noisy_expectations, out.noisy_sems,
        out.zne_extrapolated, out.ideal_expectation,
    ))), use_container_width=True)
    st.caption(
        "Each point is the value measured at that noise scale (1x, 2x, 3x...), with a ±1 SEM "
        "error bar (standard error of the mean over the Monte Carlo samples); the "
        "extrapolation estimates the zero-noise value."
    )


def _show_density_channel():
    out = _show_analysis("dm")
    if out is not None:
        st.bar_chart({"ideal": out.ideal_diagonal, "noisy": out.noisy_diagonal})


def _show_magic(res):
    if not _ran("magic_on"):
        return _press_run()
    qubit = min(int(_param("magic_qubit") or 0), res.n_qubits - 1)

    def job():
        rho = dense_evolution.partial_trace(res.statevector, res.n_qubits, [res.n_qubits - 1 - qubit])
        return stabilizer_renyi_entropy(res.statevector), magic_entropy(rho)

    whole, single = _memo(("magic", qubit), job)
    st.metric("Stabilizer Rényi entropy (whole state)", f"{whole:.6f} bit")
    st.caption("Zero for every stabilizer state, positive for 'magic' (non-Clifford) states.")
    st.metric(f"magic_entropy (qubit {qubit})", f"{single:.6f} bit")
    st.caption("3-copy Key-Unitary construction: 0 for |0>,|1>,|+>,|->,|+i>,|-i>, 0.811 for T and H.")


def _show_bond_convergence():
    out = _show_analysis("bond_convergence")
    if out is None:
        return
    verdict = {
        "converged": "✅ converged", "not_converged": "⚠️ not_converged", "undecidable": "❔ undecidable",
    }[out.verdicts[0]]
    st.metric("Verdict", verdict)
    st.dataframe(
        [
            {"bond": b, "chi_used": c, "avg_jsd": j, "budget_violations": v, "|<P>|": abs(val)}
            for b, c, j, v, val in zip(
                out.bonds, out.chi_used, out.avg_jsd, out.budget_violations, out.values[0],
            )
        ],
        width="stretch",
    )
    st.caption(
        "diffs (change between consecutive bonds): " + ", ".join(f"{d:.2e}" for d in out.diffs[0])
        + ". If chi_used at the highest bond equals its cap the verdict is 'undecidable'."
    )


def _show_chemistry():
    err = st.session_state.get("vqe_error")
    vqe = st.session_state.get("vqe_result")
    if err:
        st.error(f"VQE error: {err}")
    elif vqe is None:
        st.info("Turn Chemistry On and press ▶ Run: the circuit becomes the VQE ansatz.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("VQE energy (Hartree)", f"{vqe['vqe_energy_hartree']:.6f}")
        if vqe["exact_energy_hartree"] is not None:
            col2.metric("Exact energy (Hartree)", f"{vqe['exact_energy_hartree']:.6f}")
        st.line_chart(vqe["energy_history"])
        st.caption(
            "The energy converges toward the molecule's ground state: the smaller the gap "
            "to the exact energy, the better the circuit approximates it."
        )
        history = np.asarray(vqe["energy_history"], dtype=np.float64)
        exact = vqe["exact_energy_hartree"]
        reached_exact = exact is not None and abs(vqe["vqe_energy_hartree"] - exact) < 1.6e-3
        if len(history) >= 10 and not reached_exact:
            tail = history[-max(5, len(history) // 5):]
            spread = float(np.ptp(history)) or 1.0
            if float(np.std(np.diff(tail))) < 1e-4 * spread:
                st.warning(
                    "The curve flattened over the last iterations (almost no energy change): "
                    "possible plateau or local minimum, not necessarily the global minimum. "
                    "Try fewer n_layers or the UCCSD ansatz."
                )
    if st.session_state.get("heal_on"):
        st.subheader("Vector healing")
        heal = _show_analysis("heal")
        if heal is not None:
            st.line_chart({"original": heal["original"], "healed": [v[0] for v in heal["healed"].healed_vectors]})
            st.caption(
                f"Correction triggered: {heal['healed'].fallback_triggered} · "
                f"reconstruction error: {heal['healed'].reconstruction_error:.4g}"
            )
    if st.session_state.get("chem_land_on"):
        st.subheader("Energy landscape")
        land = _show_analysis("land")
        if land is not None:
            st.image(_memo(("land",), lambda: _png(dc.energy_landscape_figure(*land))), use_container_width=True)
    if st.session_state.get("chem_geom_on"):
        st.subheader("Geometry optimization")
        geom = _show_analysis("geom")
        if geom is not None:
            st.line_chart(geom["history"])
            st.caption(
                f"Energy: {geom['history'][0]:.6f} -> {geom['history'][-1]:.6f} Hartree "
                "(the first JAX compilation can take a few minutes)."
            )


def _show_dynamics():
    out = _show_analysis("dyn")
    if out is None:
        return
    col1, col2 = st.columns(2)
    col1.metric("Energy (Hartree)", f"{out['forces']['energy_hartree']:.6f}")
    col2.metric("Force norm (Hartree/Å)", f"{out['forces']['force_norm']:.6f}")
    traj = out["traj"]
    st.line_chart({"time_fs": traj["time_fs"], "energy_hartree": traj["energy_hartree"]})
    st.line_chart({"time_fs": traj["time_fs"], "force_norm": traj["force_norm"]})
    st.caption(
        "Energy and force norm along the trajectory. If the simulation diverges "
        "(nuclei too close), the function stops with an error message."
    )


def _show_wormhole():
    out = _show_analysis("wh")
    if out is None:
        return
    i_plus, i_minus, mu = out
    import matplotlib.pyplot as _plt

    fig, ax = _plt.subplots(figsize=(4, 3))
    ax.barh(["mu > 0", "mu < 0"], [i_plus, -i_minus], color=["#648fff", "#dc267f"])
    ax.set_xlabel("mutual information (signed)")
    fig.tight_layout()
    st.image(_memo(("wormhole",), lambda: _png(fig)), use_container_width=True)
    st.caption(
        f"I(mu=+{mu:g}) = {i_plus:.5f} · I(mu=-{mu:g}) = {i_minus:.5f}. Traversable-wormhole-inspired "
        "teleportation (arXiv:2604.10090): the signal is the DIFFERENCE between positive and negative mu."
    )


def _show_qec():
    out = _show_analysis("qec")
    if out is None:
        return
    st.write(f"Code with {len(out['stabilizers'][0])} physical qubits, {len(out['stabilizers'])} stabilizers.")
    st.write(f"Syndrome: `{out['syndrome']}`")
    st.image(_memo(("qec",), lambda: _png(dc.qec_syndrome_map_figure(
        out["stabilizers"], out["syndrome"], out["error"],
    ))), use_container_width=True)
    st.caption(
        "Red: stabilizers that anticommute with the error (syndrome bit 1, detected). Grey: those "
        "that commute (bit 0, not detected). The marked qubit is where the injected error acts."
    )
    if out["decoded"] is None:
        st.warning(
            "Ambiguous syndrome: more than one minimum-weight correction fits it (for example a "
            "bit-flip code like ZZI/IZZ cannot tell X from Y)."
        )
    else:
        st.success(f"Correction found: `{out['decoded']}`")


def _show_condensed():
    if st.session_state.get("cm_on"):
        out = _show_analysis("cm")
        if out is not None:
            vbm, _, cbm, _, gap = out["extrema"]
            st.metric(f"Direct gap at Γ — {out['material']}", f"{out['gap_gamma']:.4f} eV")
            col1, col2, col3 = st.columns(3)
            col1.metric("VBM (eV)", f"{vbm:.4f}")
            col2.metric("CBM (eV)", f"{cbm:.4f}")
            col3.metric("Band gap Γ→X (eV)", f"{gap:.4f}")
            st.caption(
                "For indirect-gap materials (e.g. Si) the Γ→X scan gives the correct gap; the direct "
                "gap at Γ alone would give the wrong value for them."
            )
            t_path, bands = out["bands"]
            st.plotly_chart(_memo(("band",), lambda: dc.band_structure_figure(t_path, bands, out["material"])))
            st.caption(
                "All 10 sp3s* bands along Γ→X: hover a point to see the exact energy and its "
                "position on the k-path. Band 3 (red) and band 4 (blue) are the VBM/CBM."
            )
    if st.session_state.get("hub_on"):
        hub = _show_analysis("hub")
        if hub is not None:
            st.metric(f"Hubbard ground-state energy ({hub[0]} qubits)", f"{hub[1]:.6f}")


def _show_burst():
    out = _show_analysis("burst")
    if out is None:
        return
    st.dataframe(out["rows"], width="stretch")
    st.bar_chart({f"{r['time_us']:g} us": r["fidelity"] for r in out["rows"]})
    st.caption(
        f"The circuit is run with amplitude damping whose strength follows the burst profile "
        f"(peak {out['peak_ratio']:.2f}x the baseline, recovery ~25 ms; arXiv:2104.05219, "
        "arXiv:2501.00331): fidelity against the ideal circuit at each time after the impact."
    )


def _show_oscillating():
    out = _show_analysis("osc")
    if out is None:
        return
    st.line_chart({"oscillating p": out.p_eff, "base p": [out.base_p] * len(out.p_eff)})
    st.caption(
        f"Noise strength against the scale factors {out.factors}, around the ZNE strength "
        f"p = {out.base_p:g}. If this curve is not smooth, the smooth-noise assumption ZNE relies on breaks."
    )


def _show_calibration():
    out = _show_analysis("cal")
    if out is None:
        return
    st.dataframe(out[:50], width="stretch")
    st.caption(f"{len(out)} (gate, qubit) pairs with a calibrated error rate (FakeSherbrooke, IBM Eagle 127 qubits).")


def _exact_probability_figure(res):
    import matplotlib.pyplot as plt

    probs = np.abs(np.asarray(res.statevector)) ** 2
    shown = [(format(i, f"0{res.n_qubits}b"), float(p)) for i, p in enumerate(probs) if p > 1e-12]
    shown = sorted(shown, key=lambda t: -t[1])[:64]
    fig, ax = plt.subplots(figsize=(8, 1.6))
    ax.bar([bits for bits, _ in shown], [p for _, p in shown], color="#648fff")
    ax.set_ylabel("probability")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    return fig


def _show_function():
    out = _show_analysis("fn")
    if out is None:
        return
    st.caption(f"{out['name']}: " + " ".join(out["doc"]))
    if hasattr(out["out"], "savefig"):
        st.image(_memo(("fn",), lambda: _png(out["out"])), use_container_width=True)
    else:
        st.write(ex.to_display(out["out"]))


def _fixed_tabs(labels):
    with st.container(key="results_tabs"):
        st.markdown(
            "<style>"
            ".st-key-results_tabs [data-baseweb=\"tab-list\"]"
            "{display:flex;flex-wrap:wrap;overflow:visible}"
            ".st-key-results_tabs [data-baseweb=\"tab-highlight\"],"
            ".st-key-results_tabs [data-baseweb=\"tab-border\"]{display:none}"
            ".st-key-results_tabs button[role=\"tab\"]{white-space:normal;"
            "border-bottom:2px solid transparent}"
            ".st-key-results_tabs button[role=\"tab\"][aria-selected=\"true\"]"
            "{border-bottom:2px solid var(--primary-color,#ff4b4b)}"
            "</style>",
            unsafe_allow_html=True,
        )
        return st.tabs(labels)


def _show_entropy(res):
    st.caption(
        "An entangled qubit has <Z>=0 even when its partner has been operated on: mutual "
        "information sees correlations a single expectation value cannot (no-signaling "
        "theorem). Qubit numbering as in Results (Qiskit convention)."
    )
    if not _ran("entropy_on"):
        return _press_run()

    def job():
        qa = [int(x) for x in _param("entropy_a").split(",") if x.strip() != ""]
        qb = [int(x) for x in _param("entropy_b").split(",") if x.strip() != ""]
        if set(qa) & set(qb):
            raise ValueError("the two subsystems must be disjoint")
        n = res.n_qubits
        native_a = [n - 1 - q for q in qa]
        native_b = [n - 1 - q for q in qb]
        rho_a = dense_evolution.partial_trace(res.statevector, n, native_a)
        rho_b = dense_evolution.partial_trace(res.statevector, n, native_b)
        return (
            dense_evolution.von_neumann_entropy(rho_a),
            dense_evolution.von_neumann_entropy(rho_b),
            dense_evolution.mutual_information(res.statevector, n, native_a, native_b),
        )

    try:
        s_a, s_b, i_ab = _memo(("entropy", _param("entropy_a"), _param("entropy_b")), job)
    except Exception as exc:
        st.error(f"Error: {exc}")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("S(A) (nat)", f"{s_a:.4f}")
    col2.metric("S(B) (nat)", f"{s_b:.4f}")
    col3.metric("I(A:B) (nat)", f"{i_ab:.4f}")


def _show_steps(res):
    if not _ran("steps_on"):
        return _press_run()
    if res.n_qubits > 10 or not res.ops:
        st.info("Available for circuits up to 10 qubits with at least one gate.")
        return
    k = min(int(_param("sota_step") or 1), len(res.ops))

    def job():
        states = sv.step_states(res.ops, res.n_qubits)
        return sv.difference_rows(states[k - 1], states[k], res.n_qubits)

    st.caption(f"Gate {k} of {len(res.ops)}: {res.ops[k - 1]} — how each amplitude changes from the previous gate.")
    st.dataframe(_memo(("steps", k), job), width="stretch")


def _active_windows(res):
    windows = (
        ("Entropy & mutual information", "entropy_on", lambda: _show_entropy(res)),
        ("Step by step", "steps_on", lambda: _show_steps(res)),
        ("Chemistry", "chem_on", _show_chemistry),
        ("Dynamics", "dyn_on", _show_dynamics),
        ("ZNE mitigation", "zne_on", _show_zne),
        ("Density-matrix channel", "dm_on", _show_density_channel),
        ("Magic", "magic_on", lambda: _show_magic(res)),
        ("MPS convergence", "bc_on", _show_bond_convergence),
        ("Cosmic-ray burst", "burst_on", _show_burst),
        ("Oscillating noise", "osc_on", _show_oscillating),
        ("Hardware calibration", "cal_on", _show_calibration),
        ("Function", "fn_on", _show_function),
        ("Wormhole", "wh_on", _show_wormhole),
        ("Error correction", "qec_on", _show_qec),
    )
    active = [(label, show) for label, key, show in windows if st.session_state.get(key)]
    if res is None:
        needs_circuit = ("Entropy & mutual information", "Step by step", "Magic")
        active = [(label, show) for label, show in active if label not in needs_circuit]
    if st.session_state.get("cm_on") or st.session_state.get("hub_on"):
        active.append(("Condensed matter", _show_condensed))
    return active


def _show_tabs(active):
    if active:
        for tab, (_, show) in zip(_fixed_tabs([label for label, _ in active]), active):
            with tab:
                show()


# ── BUILD ────────────────────────────────────────────────────────────────
if section == "Build":
    st.caption(
        "Current state: " + (
            f"{result.n_qubits}-qubit circuit already run — open Results." if result is not None and not error
            else "press ▶ Run, the first button of the sidebar, to run a new circuit."
        )
    )
    st.caption(
        "Drag gates onto the grid to build a circuit by hand: a control ● and a "
        "target in the same column form a 2-qubit gate, two × in the same column "
        "form a SWAP."
    )
    pending = st.session_state.pop("editor_pending", None)
    if pending:
        st.session_state["n_qubits_builder"] = pending["n"]
        st.session_state["n_steps_builder"] = pending["steps"]
    if st.session_state.get("editor_note"):
        st.caption(st.session_state["editor_note"])
    if "n_qubits_builder" not in st.session_state:
        try:
            st.session_state["n_qubits_builder"] = min(8, json.loads(st.query_params.get("circuit", "")).get("n", 3))
        except (ValueError, AttributeError):
            st.session_state["n_qubits_builder"] = 3
    col_nq, col_steps = st.columns(2)
    n_qubits_builder = col_nq.number_input(
        "Qubits", min_value=1, max_value=8, step=1, key="n_qubits_builder",
    )
    n_steps_builder = col_steps.number_input(
        "Steps", min_value=2, max_value=24, value=12, step=1, key="n_steps_builder",
    )
    builder_ops = ce.circuit_editor(int(n_qubits_builder), n_columns=int(n_steps_builder))

    st.button("→ Load into the Circuit Editor", on_click=_load_qasm,
              args=(dc.gate_tuples_to_qasm(ce.ops_to_tuples(builder_ops), int(n_qubits_builder)) if builder_ops else None,),
              disabled=not builder_ops, help="Place at least one gate on the grid first." if not builder_ops else None)

    if error:
        st.error(f"Circuit run failed: {error}")


# ── RESULTS ──────────────────────────────────────────────────────────────
elif section == "Results":
    active = _active_windows(result)
    if result is None and large_result is not None:
        _show_large(large_result)
        _show_tabs(active)
    elif result is None:
        st.info("No circuit run yet — open Build and press ▶ Run.")
        _show_tabs(active)
    elif error:
        st.error(f"Circuit run failed: {error}")
        _show_tabs(active)
    else:
        st.caption(
            f"Current state: {result.n_qubits}-qubit circuit — turn windows On in the left "
            "panel and press ▶ Run, or build a new one (Build)."
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
        (tab_circuit, tab_sv, tab_prob, tab_qubits, *tabs_active) = _fixed_tabs(
            ["Circuit", "Statevector", "Probabilities & Q-sphere", "Qubits"] + [label for label, _ in active]
        )
        for tab, (_, show) in zip(tabs_active, active):
            with tab:
                show()
        with tab_circuit:
            col_circuit, col_probs = st.columns([3, 2])
            col_circuit.image(
                _memo(("circuit",), lambda: _png(dc.draw_circuit_figure(result.ops, result.n_qubits))),
                use_container_width=True,
            )
            tab_probs, tab_qubit, tab_outcomes, tab_shots = col_probs.tabs(
                ["Probabilities", "Per qubit", "Outcomes", "Measurements"]
            )
            tab_probs.caption("Shots sampled from the statevector")
            tab_probs.image(
                _memo(("hist_small",), lambda: _png(dc.histogram_figure(result.counts), 6, 3.4)),
                use_container_width=True,
            )
            tab_qubit.dataframe(
                [
                    {"qubit": row["qubit"], "P(1)": row["P(1)"]}
                    for row in _memo(("qubit_table",), lambda: sv.qubit_table(result.statevector, result.n_qubits))
                ],
                width="stretch",
            )
            tab_outcomes.dataframe(outcome_rows(result.counts), width="stretch")
            tab_shots.caption("The shots one by one (the first 100): measurement, input and output.")
            tab_shots.dataframe(
                _memo(
                    ("shots",),
                    lambda: measurement_rows(
                        result.counts, st.session_state.get("input_ones", []), result.n_qubits,
                        seed=int(st.session_state.get("_run_seed", 42)),
                    ),
                ),
                width="stretch",
            )
        with tab_sv:
            st.caption(f"{result.n_qubits} qubits — {len(result.statevector)} amplitudes (Qiskit convention)")
            rows = _memo(("sv_rows",), lambda: [
                {
                    "state": format(i, f"0{result.n_qubits}b"),
                    "amplitude (re)": float(amp.real),
                    "amplitude (im)": float(amp.imag),
                    "|amplitude|": abs(amp),
                    "phase (rad)": float(np.angle(amp)),
                }
                for i, amp in enumerate(result.statevector)
                if abs(amp) > 1e-10
            ])
            st.dataframe(rows, width="stretch")
            st.dataframe(
                _memo(("qubit_table",), lambda: sv.qubit_table(result.statevector, result.n_qubits)),
                width="stretch",
            )
            st.code(_memo(("ket",), lambda: sv.ket_string(result.statevector, result.n_qubits)), language=None)
        with tab_prob:
            st.caption("Shots sampled from the computed statevector")
            color_by_phase = st.checkbox(
                "Color by phase", key="prob_color_by_phase",
                help="Colors each bar by the complex phase of that state's amplitude "
                     "(cyclic colormap); the bar height stays the shot count.",
            )
            if st.session_state.get("prob_exact") and _ran("prob_exact"):
                st.image(
                    _memo(("exact",), lambda: _png(_exact_probability_figure(result))),
                    use_container_width=True,
                )
                st.caption("Exact probabilities of the final state; the states with probability zero are not drawn.")
            else:
                st.image(
                    _memo(
                        ("hist", bool(color_by_phase)),
                        lambda: _png(
                            dc.histogram_figure(
                                result.counts,
                                statevector=result.statevector if color_by_phase else None,
                            ),
                            8,
                            1.6,
                        ),
                    ),
                    use_container_width=True,
                )
                if st.session_state.get("prob_exact"):
                    st.caption("Press ▶ Run to draw the exact probabilities.")
            st.caption("Q-sphere")
            st.image(_memo(("qsphere",), lambda: _png(_qsphere_figure(result))), use_container_width=True)
        with tab_qubits:
            st.caption("One Bloch sphere per qubit, from its reduced density matrix.")
            st.image(
                _memo(("bloch",), lambda: _png(dc.bloch_multivector_figure(result.statevector))),
                use_container_width=True,
            )
            if 2 <= result.n_qubits <= 10:
                st.caption("Qubit pairs")
                st.image(
                    _memo(("pairs",), lambda: _png(sv.half_matrix_figure(result.statevector, result.n_qubits))),
                    use_container_width=True,
                )
