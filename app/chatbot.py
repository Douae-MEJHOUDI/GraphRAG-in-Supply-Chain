"""
Supply Chain Disruption Analyst — GraphRAG Chatbot
Run with: streamlit run app/chatbot.py
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nest_asyncio
nest_asyncio.apply()   # needed because Streamlit runs its own event loop

import streamlit as st
from src.pipeline import build_rag, ask, MODES

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="SupplyGraph — Disruption Analyst",
    page_icon="🔗",
    layout="wide",
)

st.title("🔗 SupplyGraph: Supply Chain Disruption Analyst")
st.caption(
    "Powered by GraphRAG (LightRAG) + Llama 3.1 via Groq · "
    "Knowledge graph built from supply chain disruption reports"
)

# --------------------------------------------------------------------------- #
# Sidebar — mode selector and info
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Query Mode")
    mode = st.radio(
        "Retrieval strategy",
        options=list(MODES.keys()),
        index=list(MODES.keys()).index("local"),  # default: local (faster)
        format_func=lambda m: f"{m.capitalize()} — {MODES[m]}",
    )
    st.divider()
    st.markdown(
        """
**Mode guide:**
- **Naive** — standard RAG, no graph. Use as *baseline*.
- **Local** — entity-level graph lookup. Good for *specific* supplier/port questions.
- **Global** — community/pattern level. Good for *broad* risk analysis.
- **Hybrid** — combines both. Best for *complex multi-hop* questions.
- **Mix** — hybrid + vector search. Most comprehensive.
        """
    )
    st.divider()
    st.markdown(
        "**Example questions:**\n"
        "- Which suppliers were impacted by the Suez Canal blockage?\n"
        "- How did the semiconductor shortage affect Toyota?\n"
        "- What ports faced congestion during COVID-19?\n"
        "- Which companies depend on TSMC?\n"
        "- What risk mitigation strategies were adopted after 2021?"
    )

# --------------------------------------------------------------------------- #
# Load RAG (cached across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading knowledge graph…")
def get_rag():
    return build_rag()

rag = get_rag()

# --------------------------------------------------------------------------- #
# Chat history
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("mode"):
            st.caption(f"Mode: {msg['mode']}")

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
if question := st.chat_input("Ask about supply chain disruptions…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Query the graph
    with st.chat_message("assistant"):
        with st.spinner(f"Querying graph ({mode} mode)…"):
            try:
                answer = ask(rag, question, mode=mode)
            except Exception as e:
                answer = f"Error: {e}\n\nMake sure you've run `python -m src.ingest` first and that your GROQ_API_KEY is set."

        st.markdown(answer)
        st.caption(f"Mode: {mode}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "mode": mode,
    })
