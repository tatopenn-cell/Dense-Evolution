"""
AI research-bridge: package a hypothesis + real Dense-Evolution results
into a clean, paste-ready text block for cross-checking against an
external AI, and log whatever comes back. Three bridges, one architecture
directly modeled on the "Aion"/Sigma prototype's proposer/critic roles
(the "matrix" archive's v31 app, see Dense-Evolution-Ising-Tests'
SOPHIA_REFLECTION.md for that lineage) -- rebuilt here against real
simulator data instead of invented "qualia" states:

  - Google (google.com's own AI Overview) -- no API exists for this
    consumer feature, so this bridge is deliberately copy/paste-only:
    build a short search query, the user pastes it into a real google.com
    tab and pastes the AI Overview text back. No scraping, ever.
  - A personal Colab notebook -- same reasoning, copy/paste-only: build a
    longer, self-contained context block to paste into a Colab cell/chat,
    and a place to paste the response back.
  - Any API the user has a key for -- the one bridge that IS automated: a
    plain OpenAI-chat-completions-compatible POST (the most portable
    common denominator across providers), sent directly from here.

Deliberately UI-free (no Streamlit import) so every function here is
testable on its own -- the Streamlit page (ui_pages/research_bridge.py)
is a thin orchestration layer on top of this.
"""
import time

import requests

__all__ = ['build_context_block', 'build_search_query', 'call_custom_api', 'new_log_entry']


def build_context_block(hypothesis: str, real_data: str = "", notes: str = "") -> str:
    """A clean, self-contained text block to paste into an external AI or
    notebook: what's being tested, the real numbers backing it (if any),
    and a direct question. `real_data` is caller-supplied plain text
    (e.g. a pasted CSV/printout of an actual Dense-Evolution run) --
    never invented here."""
    parts = [
        "Sto verificando un'ipotesi su un simulatore quantistico reale (Dense-Evolution), "
        "non con dati inventati -- controlla se questa idea esiste gia' in letteratura, "
        "e se i numeri sotto sono spiegabili con fisica nota o suggeriscono qualcosa di reale.",
        "",
        f"IPOTESI: {hypothesis.strip()}",
    ]
    if real_data.strip():
        parts += ["", "DATI REALI MISURATI:", real_data.strip()]
    if notes.strip():
        parts += ["", f"NOTE: {notes.strip()}"]
    parts += ["", "Esiste gia' un risultato noto equivalente? Se si', quale? Se no, cosa lo spiegherebbe?"]
    return "\n".join(parts)


def build_search_query(hypothesis: str, max_len: int = 200) -> str:
    """A short query for google.com's search box -- the full context
    block is too long for a search bar, this trims to the core claim."""
    query = hypothesis.strip().replace("\n", " ")
    if len(query) > max_len:
        query = query[:max_len].rsplit(" ", 1)[0] + "..."
    return query


def call_custom_api(context_block: str, api_url: str, api_key: str,
                     model: str = "", timeout: float = 60.0) -> str:
    """POST to an OpenAI-chat-completions-compatible endpoint. Raises on
    failure (requests.RequestException / KeyError on an unexpected
    response shape) -- the caller (the Streamlit page) is responsible for
    catching it and showing an error, this stays UI-free so it's
    independently testable."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model or "gpt-4o-mini",
        "messages": [{"role": "user", "content": context_block}],
    }
    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def new_log_entry(source: str, hypothesis: str, context_block: str, response: str = "",
                   model_label: str = "") -> dict:
    """One row of the session research log -- a plain dict, kept in
    st.session_state by the page, exportable as-is via json.dumps."""
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "model": model_label,
        "hypothesis": hypothesis,
        "context_block": context_block,
        "response": response,
    }
