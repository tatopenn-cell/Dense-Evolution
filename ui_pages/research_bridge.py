"""
Research Bridge page -- the "serious app" version of the Aion/Sigma
prototype's proposer/critic architecture (the "matrix" archive's v31 app;
see Dense-Evolution-Ising-Tests/SOPHIA_REFLECTION.md for that lineage),
rebuilt against this project's real simulator instead of invented states.
Package a hypothesis (plus real Dense-Evolution numbers, pasted in by
hand -- nothing here invents data) into a clean context block, cross-
check it against an external AI through one of three bridges, and log
whatever comes back:

  - Google's own AI Overview (google.com search) -- copy/paste only, no
    API exists for this consumer feature.
  - A personal Colab notebook -- copy/paste only, same reasoning.
  - Any API the user has a key for -- automated, a direct POST from here.
  - Any AI already authenticated locally via CLI (Claude Code or
    anything else) -- also automated, via a local subprocess instead of
    a billed API call.
"""
import json
import urllib.parse

import streamlit as st

from dashboard_core.research_bridge import (
    build_context_block, build_search_query, call_custom_api, new_log_entry,
    build_next_search_query, call_local_cli,
)
from ui_pages.components import render_page_banner


def render():
    render_page_banner(
        "Research Bridge",
        """Impacchetta un'ipotesi (piu' i numeri reali di un tuo esperimento, se ce li hai)
        in un blocco di testo pronto da incollare, e confrontala con un'IA esterna a tua scelta —
        Google, un Colab personale, o una qualsiasi API con la tua chiave. Niente scraping,
        niente automazione dove non esiste un'API reale: tu resti il ponte fisico dove serve.""",
        accent="#f4a261", bg_from="#140a00", bg_to="#261200",
    )

    st.subheader("1. L'ipotesi")
    hypothesis = st.text_area(
        "Cosa stai verificando?", height=100,
        placeholder="Es: la topologia usata per raggiungere uno stato GHZ-classe influenza "
                    "la resilienza al rumore anche a parita' di struttura ideale...",
    )
    real_data = st.text_area(
        "Dati reali (opzionale — incolla output/CSV di un tuo esperimento Dense-Evolution)",
        height=120, placeholder="Es: output di topology_resilience_loop.py, o qualunque numero vero...",
    )
    notes = st.text_input("Note aggiuntive (opzionale)")

    if not hypothesis.strip():
        st.info("Scrivi un'ipotesi sopra per generare i blocchi da incollare.")
        return

    context_block = build_context_block(hypothesis, real_data, notes)
    log = st.session_state.setdefault("bridge_log", [])

    st.divider()
    st.subheader("2. I quattro ponti")

    tab_google, tab_colab, tab_custom, tab_local = st.tabs(
        ["🔍 Google", "📓 Colab personale", "🔑 La tua IA (API)", "💻 IA locale (CLI)"]
    )

    with tab_google:
        st.caption(
            "Nessuna API per l'AI Mode di google.com — ma ha una memoria conversazionale "
            "enorme dentro la stessa sessione: apri la ricerca UNA volta al passo 1, poi "
            "continua a scrivere i follow-up nella STESSA conversazione, senza mai riaprirla "
            "da zero (altrimenti perdi tutta la memoria accumulata). Ogni follow-up e' generato "
            "in base a cosa manca ancora (pattern ReAct: ragiona, poi agisci)."
        )

        chain_key = f"search_chain::{hash(hypothesis)}"
        chain = st.session_state.setdefault(chain_key, [])

        with st.expander("🧠 Chiave per il ragionamento tra un passo e l'altro (opzionale)"):
            st.caption(
                "Serve un LLM per decidere la ricerca successiva in modo intelligente — senza "
                "chiave, dopo ogni passo scrivi tu la query seguente a mano."
            )
            react_url = st.text_input(
                "Endpoint API", key="react_api_url",
                placeholder="https://api.openai.com/v1/chat/completions oppure https://api.anthropic.com/v1/messages",
            )
            react_key = st.text_input("Chiave API", type="password", key="react_api_key")
            react_model = st.text_input("Modello (opzionale)", key="react_model", placeholder="gpt-4o-mini")

        # Show every completed step of the chain so far.
        for i, step in enumerate(chain):
            with st.container(border=True):
                st.markdown(f"**Passo {i + 1}:** `{step['query']}`")
                st.caption(step['result'][:300] + ("..." if len(step['result']) > 300 else ""))

        # The query for the CURRENT (not-yet-answered) step.
        pending_key = f"pending_query::{hash(hypothesis)}"
        if pending_key not in st.session_state:
            st.session_state[pending_key] = build_search_query(hypothesis)
        current_query = st.session_state[pending_key]

        if len(chain) >= 6:
            st.warning("6 ricerche gia' fatte in questa catena — considera di fermarti qui.")

        if len(chain) == 0:
            st.markdown("**Prima ricerca (apri una nuova conversazione AI Mode):**")
            st.code(current_query, language=None)
            search_url = "https://www.google.com/search?q=" + urllib.parse.quote(current_query) + "&udm=50"
            st.link_button("Apri Google (nuova conversazione)", search_url)
        else:
            st.markdown(f"**Follow-up (passo {len(chain) + 1}) — incollalo nella conversazione GIA' aperta, non aprirne una nuova:**")
            st.code(current_query, language=None)
            st.caption("⚠️ Non cliccare di nuovo 'Apri Google' — perderesti la memoria dei passi precedenti.")
        google_response = st.text_area("Incolla qui la risposta di Google", key="google_response", height=150)

        col_a, col_b = st.columns(2)
        with col_a:
            step_saved = st.button("✅ Salva questo passo", key="save_google_step")
        with col_b:
            stop_and_log = st.button("🏁 Ferma e salva la catena nel log", key="stop_google_chain")

        if step_saved and google_response.strip():
            chain.append({'query': current_query, 'result': google_response.strip()})
            if react_url and react_key:
                with st.spinner("Ragiono sulla prossima ricerca (ReAct)..."):
                    try:
                        next_step = build_next_search_query(hypothesis, chain, react_url, react_key, react_model)
                    except Exception as e:
                        next_step = None
                        st.error(f"Ragionamento fallito: {e}")
                if next_step is not None:
                    if next_step['done']:
                        st.success(f"Catena completa: {next_step['reasoning']}")
                        summary = "\n\n".join(f"Passo {i+1}: {s['query']} -> {s['result']}"
                                               for i, s in enumerate(chain))
                        log.append(new_log_entry(
                            f"Google (catena ReAct, {len(chain)} passi)", hypothesis, summary,
                            f"SINTESI FINALE: {next_step['reasoning']}", react_model))
                        st.session_state[chain_key] = []
                        del st.session_state[pending_key]
                    else:
                        st.session_state[pending_key] = next_step['query'] or current_query
            st.rerun()

        if stop_and_log and chain:
            summary = "\n\n".join(f"Passo {i+1}: {s['query']} -> {s['result']}"
                                   for i, s in enumerate(chain))
            log.append(new_log_entry(f"Google (catena, {len(chain)} passi)", hypothesis, summary,
                                      chain[-1]['result']))
            st.session_state[chain_key] = []
            if pending_key in st.session_state:
                del st.session_state[pending_key]
            st.success("Catena salvata nel log.")
            st.rerun()

    with tab_colab:
        st.caption(
            "Nessuna API per un Colab personale — copia il blocco intero, incollalo in una "
            "cella/chat del tuo notebook, incolla la risposta."
        )
        st.code(context_block, language=None)
        st.link_button("Apri Colab", "https://colab.research.google.com/")
        colab_response = st.text_area("Incolla qui la risposta", key="colab_response", height=150)
        if st.button("Salva nel log", key="save_colab") and colab_response.strip():
            log.append(new_log_entry("Colab personale", hypothesis, context_block, colab_response))
            st.success("Salvato.")

    with tab_custom:
        st.caption(
            "Qui invece l'invio e' automatico — inserisci la tua chiave e l'endpoint. "
            "Riconosce da solo l'API di Anthropic (Claude) da un URL con \"anthropic.com\"; "
            "qualsiasi altro endpoint viene trattato come compatibile OpenAI chat-completions."
        )
        api_url = st.text_input(
            "Endpoint API",
            placeholder="https://api.openai.com/v1/chat/completions oppure https://api.anthropic.com/v1/messages",
        )
        api_key = st.text_input("Chiave API", type="password")
        model = st.text_input("Modello (opzionale)", placeholder="gpt-4o-mini")
        st.code(context_block, language=None)
        if st.button("🚀 Invia direttamente", type="primary", disabled=not (api_url and api_key)):
            with st.spinner("Chiamata in corso..."):
                try:
                    reply = call_custom_api(context_block, api_url, api_key, model)
                    st.session_state["custom_response"] = reply
                    log.append(new_log_entry("Chiave personale", hypothesis, context_block, reply, model))
                    st.success("Risposta ricevuta e salvata nel log.")
                except Exception as e:
                    st.error(f"Chiamata fallita: {e}")
        if st.session_state.get("custom_response"):
            st.text_area("Risposta", value=st.session_state["custom_response"], height=150, disabled=True)

    with tab_local:
        st.caption(
            "Nessun costo API separato — richiama un'IA a riga di comando che hai gia' "
            "autenticata sul PC (Claude Code o qualsiasi altra), tramite un comando locale "
            "invece di una chiamata a pagamento. Il blocco di contesto viene passato in "
            "stdin, la risposta letta da stdout."
        )
        local_command = st.text_input(
            "Comando", key="local_cli_command",
            placeholder='es: claude -p    (Claude Code, modalita\' non interattiva)',
        )
        st.code(context_block, language=None)
        if st.button("💻 Esegui in locale", type="primary", disabled=not local_command.strip()):
            with st.spinner("Esecuzione in corso..."):
                try:
                    reply = call_local_cli(context_block, local_command)
                    st.session_state["local_response"] = reply
                    log.append(new_log_entry("IA locale (CLI)", hypothesis, context_block, reply, local_command))
                    st.success("Risposta ricevuta e salvata nel log.")
                except Exception as e:
                    st.error(f"Esecuzione fallita: {e}")
        if st.session_state.get("local_response"):
            st.text_area("Risposta", value=st.session_state["local_response"], height=150, disabled=True)

    if log:
        st.divider()
        st.subheader(f"3. Log ({len(log)} voci)")
        for entry in reversed(log):
            with st.expander(f"{entry['timestamp']} — {entry['source']}"):
                st.markdown(f"**Ipotesi:** {entry['hypothesis']}")
                if entry.get('model'):
                    st.caption(f"Modello: {entry['model']}")
                st.text_area(
                    "Risposta", value=entry['response'], height=100, disabled=True,
                    key=f"log_{entry['timestamp']}_{entry['source']}",
                )
        st.download_button(
            "⬇️ Esporta log (JSON)",
            data=json.dumps(log, indent=2, ensure_ascii=False),
            file_name="research_bridge_log.json",
            mime="application/json",
        )
