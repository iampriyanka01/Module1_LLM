import streamlit as st

from core.llm import get_api_key, get_model
from core.ui import MODULES, module_score

st.title("🧠 GenAI Learning Hub")
st.markdown(
    "An interactive companion to the **Generative AI course**: from *what is AI?* all the way to "
    "*function calling* and *LLM-as-a-judge evaluation*. Every module has **plain-English explanations, "
    "real-life analogies, hands-on simulators, runnable code, live Gemini demos and a quiz**."
)

c1, c2, c3 = st.columns(3)
c1.metric("Modules", len(MODULES))
done = [k for k in MODULES if module_score(k) is not None]
c2.metric("Quizzes attempted", f"{len(done)} / {len(MODULES)}")
c3.metric("Live demos", "On" if get_api_key() else "Off", help=f"Model: {get_model()}. Add a Gemini key in the sidebar to turn live demos on.")

st.subheader("🗺️ The learning path")
st.caption("Each step builds on the previous one. Follow the order in the sidebar, or jump to what you need.")

path = [
    ("m1", "🌱", "AI ⊃ ML ⊃ DL ⊃ GenAI; how models pick the next word (temperature, top-k, top-p)."),
    ("m2", "🏗️", "How LLMs are built (pre-training → SFT → RLHF), open vs closed, SLMs, ethics."),
    ("m3", "✍️", "Role · Instruction · Input · Output format; zero/one/few-shot; dynamic prompts."),
    ("m4", "🧩", "Chain-of-Thought, Auto-CoT, Self-Consistency and Tree-of-Thought."),
    ("m51", "🌐", "DNS, TCP, packets, client-server, JSON, HTTP methods and status codes."),
    ("m52", "🔌", "Black-box LLMs, API keys, tokens & cost, caching, retries, security."),
    ("m6", "🛡️", "Streaming, batching, Pydantic validation, failure handling, guardrails."),
    ("m7", "💬", "Chat roles, system prompts, 'memory' illusion, context window strategies."),
    ("m8", "🛠️", "Tools, tool cards, the function-calling loop, multi & async calls, fallbacks."),
    ("m9", "📏", "Hallucinations, model quirks, metrics, rubrics, LLM-as-judge, refinement loop."),
]
cols = st.columns(2)
for i, (key, icon, desc) in enumerate(path):
    with cols[i % 2]:
        with st.container(border=True):
            score = module_score(key)
            badge = f" · quiz best **{score:.0%}**" if score is not None else ""
            st.markdown(f"**{icon} {MODULES[key]}**{badge}")
            st.caption(desc)

st.subheader("🧪 Your own course projects are inside")
st.markdown(
    """
| Your file | Where it lives in this app | What was added |
|---|---|---|
| `chatbot_app.py` (auto-summary chatbot) | **7 · Conversational Interfaces → Summarization** | Visual token savings, and the summary is fed back as context |
| `3_weatherSearchaddingLLM.py` (weather tool) | **8 · Function Calling → Live tool loop** | Full loop: tool call → your function → result back to Gemini → final answer |
| `study_buddy.py` + `evaluator.py` + `prompts.py` | **9 · Evaluating Outputs → LLM-as-judge** | Same prompts and `safe_parse_json`, with score charts |
"""
)

with st.expander("▶️ How to run live demos"):
    st.markdown(
        """
1. Get a free key at **Google AI Studio → Get API key**.
2. Either paste it in the sidebar, **or** create a `.env` file next to `app.py`:
   ```
   GEMINI_API_KEY=your-key-here
   ```
   (no quotes, no spaces around `=`; `GOOGLE_API_KEY` also works)
3. Pick a model in the sidebar. If you get a *model not found* error, choose another one.
4. Optional: add `WEATHER_API_KEY=...` (OpenWeatherMap) for real weather in Module 8. Without it, a mock weather service is used.
"""
    )
