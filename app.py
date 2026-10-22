"""
GenAI Learning Hub
==================
An interactive Streamlit course companion covering:
  1 Foundations of GenAI · 2 GenAI Platforms · 3 Prompt basics · 4 Advanced prompting
  5.1 APIs · 5.2 LLM API integration · 6 Response handling · 7 Conversational interfaces
  8 Function calling · 9 Evaluating outputs · 10-11 Projects · Downloads & API keys

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="GenAI Learning Hub", page_icon="🧠", layout="wide")

from core.llm import DEFAULT_MODEL, GENAI_AVAILABLE, MODEL_CHOICES, env_api_key  # noqa: E402

# ---------------------------------------------------------------------------
# Sidebar: API key + model (shared by every page)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Live-demo settings")
    if env_api_key():
        st.success("API key loaded from `.env`", icon="🔑")
        override = st.text_input("Override key (optional)", type="password", key="api_key_input")
        st.session_state["api_key"] = override or None
    else:
        key = st.text_input(
            "Gemini API key",
            type="password",
            key="api_key_input",
            help="Create one free at aistudio.google.com → Get API key. It stays in this browser session only.",
        )
        st.session_state["api_key"] = key or None
        if not key:
            st.caption("No key? Every explanation and simulator still works — only the 🟢 live demos need one.")

    options = MODEL_CHOICES + (["(custom)"] if DEFAULT_MODEL in MODEL_CHOICES else [DEFAULT_MODEL, "(custom)"])
    choice = st.selectbox("Model", options, index=options.index(DEFAULT_MODEL), key="model_choice")
    if choice == "(custom)":
        choice = st.text_input("Custom model id", value=DEFAULT_MODEL, key="model_custom")
    st.session_state["model"] = choice
    if not GENAI_AVAILABLE:
        st.error("`google-genai` is not installed — live demos are disabled.")
    st.divider()

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
P = "lessons/"
pages = {
    "Start": [st.Page(P + "home.py", title="Home & course map", icon="🏠", default=True)],
    "Foundations": [
        st.Page(P + "m1_foundations.py", title="1 · Foundations of GenAI", icon="🌱"),
        st.Page(P + "m2_platforms.py", title="2 · GenAI Platforms & LLMs", icon="🏗️"),
    ],
    "Prompt Engineering": [
        st.Page(P + "m3_prompt_basics.py", title="3 · Prompt Basics", icon="✍️"),
        st.Page(P + "m4_advanced_prompting.py", title="4 · Advanced Prompting", icon="🧩"),
    ],
    "Building with APIs": [
        st.Page(P + "m51_api_basics.py", title="5.1 · How APIs Work", icon="🌐"),
        st.Page(P + "m52_api_integration.py", title="5.2 · LLM API Integration", icon="🔌"),
        st.Page(P + "m6_response_handling.py", title="6 · Handling LLM Responses", icon="🛡️"),
    ],
    "Real Applications": [
        st.Page(P + "m7_conversational.py", title="7 · Conversational Interfaces", icon="💬"),
        st.Page(P + "m8_function_calling.py", title="8 · Function Calling & Tools", icon="🛠️"),
        st.Page(P + "m9_evaluation.py", title="9 · Evaluating GenAI Outputs", icon="📏"),
    ],
    "Projects": [
        st.Page(P + "m10_scamguard.py", title="10 · ScamGuard AI", icon="🛡️"),
        st.Page(P + "m11_cinemate.py", title="11 · CineMate & Cinema Advisor", icon="🎬"),
    ],
    "Resources": [
        st.Page(P + "m12_downloads.py", title="Downloads & API keys", icon="⬇️"),
    ],
}

pg = st.navigation(pages, expanded=True)
st.session_state["_current_page"] = pg.title  # used by the request log (Module 5.2)
with st.sidebar:
    n_calls = len(st.session_state.get("call_log", []))
    st.caption(f"📒 Live API calls this session: **{n_calls}** (see 5.2 → Monitoring)")
pg.run()
