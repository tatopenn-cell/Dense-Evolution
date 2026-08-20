"""Molecule name resolution: short ids <-> the kernel's full catalog keys.

The kernel's own catalog keys are long, human-readable strings, e.g.
"H2 (Idrogeno) - R = 0.7414 A [equilibrio reale]" -- fine for a web page
label, error-prone for an agent to reproduce verbatim across several tool
calls (exact punctuation, accented characters, etc.). Rather than change
the kernel's catalog (that key is also what the published Composer page
uses), this adapter derives a short id from each key's leading token
(e.g. "H2", "LiH", "HeH+") and accepts either form everywhere a molecule
`name` is expected. Derived from the live catalog, not hardcoded, so it
stays correct if the catalog grows.

Cached per mapping via TTLCache (utils/cache.py) instead of a plain dict
that only ever got reset by test code -- BUG FIX: the pre-Phase-2 cache
never expired on its own, so a long-running MCP server process would
never pick up a real catalog change (e.g. after a kernel restart with a
different build). A failed fetch is also cached briefly, so a tight loop
of tool calls made while the kernel is down doesn't each wait out their
own connection/timeout error.

Shared by tools/system_tools.py (dense_evolution_list_molecules) and
tools/chemistry_tools.py (every tool taking a molecule `name`) -- lives
in its own module rather than either of those, since Phase 3 (prog.txt
Sezione 3) splits tools by topic and this is domain glue, not a tool
itself.
"""
from .client import _request
from .utils.cache import TTLCache

_molecule_catalog_cache = TTLCache(ttl_seconds=300.0, failure_ttl_seconds=10.0)
# cache value per mapping: (annotated_catalog_list, {short_id_lower: full_key})


def _short_id(full_key: str) -> str:
    return full_key.split(" (")[0].split(" -")[0].strip()


async def _get_annotated_molecule_catalog(mapping: str) -> list:
    """Catalog entries with a short `id` field added. See the cache note
    in this module's docstring."""
    cached = _molecule_catalog_cache.get(mapping)
    if cached is not None:
        return cached[0]
    try:
        catalog = await _request("GET", "/api/hamiltonians", timeout=10.0, params={"mapping": mapping})
    except Exception as e:
        _molecule_catalog_cache.set_failure(mapping, e)
        raise
    aliases = {}
    annotated = []
    for full_key, spec in catalog.items():
        short = _short_id(full_key)
        aliases[short.lower()] = full_key
        annotated.append({"id": short, "full_name": full_key, **spec})
    _molecule_catalog_cache.set(mapping, (annotated, aliases))
    return annotated


async def _resolve_molecule_name(name: str) -> str:
    """Accept either a short id ('H2') or the full catalog key and return
    the full catalog key the kernel expects. Falls back to returning the
    input unchanged if it's neither -- the kernel's own 404 (with the name
    as given) is a clearer error than silently guessing."""
    cached = _molecule_catalog_cache.get("jordan_wigner")
    if cached is None:
        await _get_annotated_molecule_catalog("jordan_wigner")
        cached = _molecule_catalog_cache.get("jordan_wigner")
    _, aliases = cached
    if name in aliases.values():
        return name
    return aliases.get(name.lower(), name)
