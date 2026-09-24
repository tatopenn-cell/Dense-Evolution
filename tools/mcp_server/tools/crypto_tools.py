"""Tools: quantum cryptography protocols (crypto-q, promoted from
Dense-Evolution-Discovery issue #189). Registered against the shared
`mcp` instance created in server.py -- see that module's docstring for
why importing `mcp` back from there (rather than the other way around)
is safe despite looking circular."""
import json

from ..client import _request, catch_errors
from ..config import COMPUTE
from ..models import Bb84Input, DiQkdGhzInput, DickaInput
from ..server import mcp


@mcp.tool(name="dense_evolution_crypto_bb84", annotations={"title": "Run BB84 quantum key distribution", **COMPUTE})
@catch_errors
async def dense_evolution_crypto_bb84(params: Bb84Input) -> str:
    """Run real BB84 quantum key distribution: prepare -> channel ->
    measure -> sift -> QBER. Validated at N=5000 rounds, 5 independent
    seeds: QBER=0 on a perfect channel, QBER=2*p_channel/3 under isotropic
    depolarizing noise, QBER=0.25 under an intercept-resend attack
    (eve=True) -- all within +-1sigma of theory.

    Args:
        params (Bb84Input): n_rounds, p_channel, eve, seed.

    Returns:
        str: JSON with n_rounds, p_channel, eve, qber, sifted_key_length.
    """
    return json.dumps(await _request("POST", "/api/crypto/bb84", timeout=60.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_crypto_di_qkd_ghz", annotations={"title": "Run device-independent QKD via GHZ(3)", **COMPUTE})
@catch_errors
async def dense_evolution_crypto_di_qkd_ghz(params: DiQkdGhzInput) -> str:
    """Run real three-party device-independent conference key agreement
    via a GHZ(3) state (Ribeiro, Murta & Wehner 2018, arXiv:1708.00798),
    using the paper's own Parity-CHSH inequality. win_rate reaches the
    quantum maximum ~0.85355 to machine precision on an ideal channel,
    comfortably clearing the classical bound of 0.75.

    Args:
        params (DiQkdGhzInput): n_rounds, p_dep, seed.

    Returns:
        str: JSON with n_rounds, p_dep, win_rate, expected_win_rate, qber_b1, qber_b2.
    """
    return json.dumps(await _request("POST", "/api/crypto/di_qkd_ghz", timeout=60.0, json=params.model_dump()), indent=2)


@mcp.tool(name="dense_evolution_crypto_dicka", annotations={"title": "Run the multi-round DICKA protocol", **COMPUTE})
@catch_errors
async def dense_evolution_crypto_dicka(params: DickaInput) -> str:
    """Run the full multi-round DICKA structure (Appendix Protocol 2 of
    Ribeiro, Murta & Wehner 2018) around the GHZ(3) primitives:
    round selection, parameter estimation, and the abort decision.
    Deliberately does not report a secure key length -- Theorem 4's exact
    value depends on a numerical optimization the source paper never
    reduces to closed form.

    Args:
        params (DickaInput): n_rounds, gamma, beta, p_dep, seed.

    Returns:
        str: JSON with n_rounds, n_test, n_key, p_hat, beta, aborted, qber_b1, qber_b2.
    """
    return json.dumps(await _request("POST", "/api/crypto/dicka", timeout=60.0, json=params.model_dump()), indent=2)
