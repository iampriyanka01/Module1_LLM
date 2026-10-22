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
QUIZ_MODS = [k for k in MODULES if k != "m12"]
c1.metric("Modules", len(QUIZ_MODS))
done = [k for k in QUIZ_MODS if module_score(k) is not None]
c2.metric("Quizzes attempted", f"{len(done)} / {len(QUIZ_MODS)}")
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
    ("m10", "🛡️", "Project: ScamGuard AI — prompt strategies, parser/validator, evaluation, Colab notebook."),
    ("m11", "🎬", "Projects: CineMate (tool-calling agent) and Cinema Advisor (structured output + judge)."),
    ("m12", "⬇️", "Download every script, notebook, project and note; full guide to API keys & secrets."),
]
cols = st.columns(2)
for i, (key, icon, desc) in enumerate(path):
    with cols[i % 2]:
        with st.container(border=True):
            score = module_score(key)
            badge = f" · quiz best **{score:.0%}**" if score is not None else ""
            st.markdown(f"**{icon} {MODULES[key]}**{badge}")
            st.caption(desc)

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
