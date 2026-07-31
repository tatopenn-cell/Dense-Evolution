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
"""
import json
import urllib.parse

import streamlit as st

from dashboard_core.research_bridge import (
    build_context_block, build_search_query, call_custom_api, new_log_entry,
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
    st.subheader("2. I tre ponti")

    tab_google, tab_colab, tab_custom = st.tabs(["🔍 Google", "📓 Colab personale", "🔑 La tua IA"])

    with tab_google:
        st.caption(
            "Nessuna API per l'AI Overview di google.com — copia la query, apri Google, "
            "incolla la risposta che leggi."
        )
        query = build_search_query(hypothesis)
        st.code(query, language=None)
        search_url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
        st.link_button("Apri Google", search_url)
        google_response = st.text_area("Incolla qui l'AI Overview", key="google_response", height=150)
        if st.button("Salva nel log", key="save_google") and google_response.strip():
            log.append(new_log_entry("Google (AI Overview)", hypothesis, query, google_response))
            st.success("Salvato.")

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
            "Qui invece l'invio e' automatico — inserisci la tua chiave e l'endpoint "
            "(compatibile OpenAI chat-completions)."
        )
        api_url = st.text_input("Endpoint API", placeholder="https://api.openai.com/v1/chat/completions")
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
