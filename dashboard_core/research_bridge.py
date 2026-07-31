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
  - Any API the user has a key for -- automated, a plain OpenAI-chat-
    completions-compatible or Anthropic Messages POST (auto-detected),
    sent directly from here.
  - Any AI already authenticated locally via a CLI (Claude Code or
    anything else) -- also automated, but via a local subprocess instead
    of a billed API call: no separate API credits, reuses whatever
    session the user already has, and -- unlike a raw API call -- an
    agentic CLI tool brings its own tools (web search, etc.) with it
    while it works, not just plain text in/text out.

Deliberately UI-free (no Streamlit import) so every function here is
testable on its own -- the Streamlit page (ui_pages/research_bridge.py)
is a thin orchestration layer on top of this.
"""
import shlex
import subprocess
import time

import requests

__all__ = [
    'build_context_block', 'build_search_query', 'call_custom_api', 'new_log_entry',
    'build_next_search_query', 'call_local_cli',
]


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
                     model: str = "", timeout: float = 60.0, provider: str = "auto") -> str:
    """POST to an OpenAI-chat-completions-compatible endpoint, OR to the
    Anthropic Messages API -- the two have different auth headers, request
    bodies, and response shapes, so this picks the right one instead of
    assuming everyone's endpoint speaks OpenAI's dialect.

    Parameters
    ----------
    provider : 'auto' | 'openai' | 'anthropic'
        'auto' (default) detects Anthropic from `api_url` containing
        "anthropic.com"; anything else is treated as OpenAI-compatible.
        Pass 'anthropic'/'openai' explicitly to override the guess (e.g.
        a self-hosted proxy that doesn't have "anthropic.com" in its URL).

    Raises on failure (requests.RequestException / KeyError on an
    unexpected response shape) -- the caller (the Streamlit page) is
    responsible for catching it and showing an error, this stays UI-free
    so it's independently testable.
    """
    is_anthropic = provider == "anthropic" or (provider == "auto" and "anthropic.com" in api_url)

    if is_anthropic:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model or "claude-haiku-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": context_block}],
        }
        resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model or "gpt-4o-mini",
        "messages": [{"role": "user", "content": context_block}],
    }
    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_local_cli(context_block: str, command: str, timeout: float = 120.0) -> str:
    """
    Run a locally-authenticated AI CLI (Claude Code, or anything else)
    as a subprocess instead of a billed API call -- no separate API
    credits, reuses whatever session the user already has logged into on
    their own machine. The context block is piped in via stdin (the one
    interface virtually every CLI tool supports), not appended as a
    command-line argument -- avoids shell-quoting/injection entirely and
    works the same way regardless of how long the context block is.

    Parameters
    ----------
    context_block : the text to send -- piped to the subprocess's stdin.
    command : the CLI invocation, e.g. "claude -p" (Claude Code's
        non-interactive print mode) or any other tool's equivalent.
        Parsed with shlex so quoting inside `command` itself works
        normally; never built by concatenating untrusted strings.
    timeout : seconds before the subprocess is killed and a
        subprocess.TimeoutExpired is raised.

    Returns
    -------
    str
        The subprocess's stdout, stripped.

    Raises
    ------
    ValueError
        If `command` is empty.
    subprocess.CalledProcessError
        If the process exits non-zero (its stderr is included in the
        exception via `output`/`stderr` attributes).
    subprocess.TimeoutExpired
        If it doesn't finish within `timeout`.
    FileNotFoundError
        If the command itself isn't found on PATH.
    """
    if not command.strip():
        raise ValueError("command must not be empty")

    args = shlex.split(command)
    result = subprocess.run(
        args, input=context_block, capture_output=True, text=True,
        timeout=timeout, check=True,
    )
    return result.stdout.strip()


def build_next_search_query(hypothesis: str, chain_history: list,
                             api_url: str, api_key: str, model: str = "") -> dict:
    """
    Generate the next step of a chained Google search using the ReAct
    pattern (Yao et al., 2022, "ReAct: Synergizing Reasoning and Acting
    in Language Models") -- reason explicitly about what's still unknown
    before deciding the next query, instead of chaining searches blindly.

    Reasoning requires an LLM (there is no honest non-LLM substitute for
    this step -- a keyword-overlap heuristic would just be pretending to
    reason), so this returns None when no api_url/api_key is given. The
    caller should fall back to letting the user type the next query by
    hand in that case, not silently degrade to a fake heuristic.

    Parameters
    ----------
    hypothesis : the original claim being investigated.
    chain_history : list of {'query': str, 'result': str} already
        gathered in this chain -- result is the AI Overview text the
        user pasted back for that query.
    api_url, api_key, model : forwarded to call_custom_api for the
        reasoning call itself (this can be a different, cheaper/faster
        model than whatever is used elsewhere in the bridge).

    Returns
    -------
    dict | None
        None if no API credentials were given. Otherwise
        {'done': bool, 'query': str | None, 'reasoning': str} --
        'done'=True means the model judged the chain has enough to
        answer (its summary is in 'reasoning'); otherwise 'query' is the
        next search to run.
    """
    if not (api_url and api_key):
        return None

    trail = "\n\n".join(
        f"Ricerca {i + 1}: \"{h['query']}\"\nRisultato letto: {h['result']}"
        for i, h in enumerate(chain_history)
    )
    prompt = (
        "Stai seguendo il pattern ReAct (ragiona, poi agisci) per verificare un'ipotesi "
        "tramite ricerche Google concatenate, ognuna basata su cosa manca ancora dalla "
        "precedente -- non ripetere ricerche gia' fatte, e non fermarti finche' non hai "
        "davvero abbastanza per rispondere.\n\n"
        f"IPOTESI DA VERIFICARE: {hypothesis}\n\n"
        f"CRONOLOGIA RICERCHE FINORA:\n{trail if trail else '(nessuna ricerca ancora)'}\n\n"
        "Ragiona brevemente su cosa sappiamo e cosa manca ancora. Poi rispondi SOLO in uno "
        "di questi due formati esatti, nient'altro:\n"
        "SEARCH: <prossima query di ricerca, breve e specifica>\n"
        "oppure\n"
        "DONE: <riassunto finale onesto di cosa abbiamo trovato, incluso se l'ipotesi regge "
        "o no e perche'>"
    )
    reply = call_custom_api(prompt, api_url, api_key, model)
    upper = reply.strip().upper()
    if upper.startswith("DONE"):
        return {'done': True, 'query': None, 'reasoning': reply.split(':', 1)[-1].strip()}
    if upper.startswith("SEARCH"):
        return {'done': False, 'query': reply.split(':', 1)[-1].strip(), 'reasoning': reply}
    # Model didn't follow the requested format -- surface the raw reply
    # rather than guessing what it meant.
    return {'done': False, 'query': None, 'reasoning': reply}


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
