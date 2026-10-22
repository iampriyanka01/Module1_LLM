"""Auto-summary chatbot (Streamlit) - Module 7.

Run:   pip install streamlit google-genai python-dotenv
       echo GEMINI_API_KEY=your_key_here > .env      # never commit .env
       streamlit run m7_auto_summary_chatbot_app.py

Idea: every N user turns a separate call writes a 2-sentence summary. In "Summary + last K turns" mode the model receives
only that summary plus the most recent messages, so the prompt stops growing.
"""
import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")   # use "gemini-2.5-flash" if your key can't access it

st.set_page_config(page_title="Auto-summary chatbot", page_icon="💬")
st.title("💬 Auto-summary chatbot")

with st.sidebar:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or st.text_input("Gemini API key", type="password")
    every = st.number_input("Summarize every N user turns", 1, 20, 3)
    keep = st.number_input("Recent messages to keep (summary mode)", 2, 20, 4)
    mode = st.radio("Memory mode", ["Full history", "Summary + last N turns"])
    if st.button("Reset conversation"):
        st.session_state.clear()
        st.rerun()
if not key:
    st.info("Add a Gemini API key in the sidebar (or in .env) to start.")
    st.stop()
client = genai.Client(api_key=key)

ss = st.session_state
ss.setdefault("msgs", [])      # [{"role": "user"|"model", "text": str}]
ss.setdefault("summary", "")
ss.setdefault("turns", 0)
ss.setdefault("stats", [])

for m in ss.msgs:
    with st.chat_message("user" if m["role"] == "user" else "assistant"):
        st.write(m["text"])
if ss.summary:
    st.sidebar.info(f"📋 Summary: {ss.summary}")

def contents_to_send():
    recent = ss.msgs if mode == "Full history" or not ss.summary else ss.msgs[-int(keep):]
    pre = []
    if mode != "Full history" and ss.summary:
        pre = [types.Content(role="user", parts=[types.Part(text=f"(Context summary of earlier conversation: {ss.summary})")]),
               types.Content(role="model", parts=[types.Part(text="Understood.")])]
    return pre + [types.Content(role=m["role"], parts=[types.Part(text=m["text"])]) for m in recent]

if text := st.chat_input("Type your message…"):
    ss.msgs.append({"role": "user", "text": text})
    ss.turns += 1
    with st.chat_message("user"):
        st.write(text)
    sent = contents_to_send()
    resp = client.models.generate_content(model=MODEL, contents=sent)
    ss.msgs.append({"role": "model", "text": resp.text})
    u = resp.usage_metadata
    ss.stats.append({"turn": ss.turns, "messages sent": len(sent), "tokens sent": getattr(u, "prompt_token_count", None)})
    with st.chat_message("assistant"):
        st.write(resp.text)
    if ss.turns % int(every) == 0:
        convo = "\n".join(f"{m['role']}: {m['text']}" for m in ss.msgs)
        prev = f"Previous summary: {ss.summary}\n\n" if ss.summary else ""
        ss.summary = client.models.generate_content(
            model=MODEL, contents=prev + "Summarize the conversation below in AT MOST 2 short sentences (under 45 words). "
                                          "Keep names, numbers and preferences.\n\n" + convo).text.strip()
        st.rerun()

if ss.stats:
    st.subheader("Tokens sent per turn")
    st.dataframe(ss.stats, hide_index=True)
