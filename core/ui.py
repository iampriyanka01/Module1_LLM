"""
core/ui.py
==========
Small, reusable building blocks so every lesson page looks the same:
headers, analogy boxes, "try it" panels, quizzes and progress tracking.
"""

from __future__ import annotations

import streamlit as st

MODULES = {
    "m1": "1 · Foundations of Generative AI",
    "m2": "2 · Overview of GenAI Platforms",
    "m3": "3 · Basics of Prompt Engineering",
    "m4": "4 · Advanced Prompt Engineering",
    "m51": "5.1 · Basic Understanding of APIs",
    "m52": "5.2 · API Integration with LLMs",
    "m6": "6 · Advanced Handling of LLM Responses",
    "m7": "7 · Conversational Interfaces",
    "m8": "8 · Function Calling & Tools",
    "m9": "9 · Evaluating GenAI Outputs",
}


def page_header(module_key: str, subtitle: str, objectives: list[str]):
    st.title(MODULES[module_key])
    st.caption(subtitle)
    with st.expander("🎯 What you'll be able to do after this module", expanded=False):
        for o in objectives:
            st.markdown(f"- {o}")


def analogy(text: str, title: str = "Real-life analogy"):
    st.info(f"**{title}:** {text}", icon="💡")


def definition(term: str, text: str):
    st.success(f"**{term}** — {text}", icon="📘")


def warn(text: str):
    st.warning(text, icon="⚠️")


def takeaway(points: list[str]):
    with st.container(border=True):
        st.markdown("#### 🧾 Key takeaways")
        for p in points:
            st.markdown(f"- {p}")


def card(title: str, body: str, icon: str = ""):
    with st.container(border=True):
        st.markdown(f"**{icon} {title}**" if icon else f"**{title}**")
        st.markdown(body)


def cards(items: list[tuple[str, str]], cols: int = 3, icon: str = ""):
    """Grid of bordered cards: items = [(title, body), ...]."""
    columns = st.columns(cols)
    for i, (title, body) in enumerate(items):
        with columns[i % cols]:
            card(title, body, icon)


def compare(left_title: str, left: str, right_title: str, right: str, bad_good: bool = True):
    c1, c2 = st.columns(2)
    with c1:
        (st.error if bad_good else st.info)(f"**{left_title}**\n\n{left}")
    with c2:
        st.success(f"**{right_title}**\n\n{right}")


def stepper(key: str, steps: list[tuple[str, str]]):
    """Walk through numbered steps one at a time (great for lifecycles)."""
    idx_key = f"step_{key}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    i = st.session_state[idx_key]
    st.progress((i + 1) / len(steps), text=f"Step {i + 1} of {len(steps)}")
    title, body = steps[i]
    with st.container(border=True):
        st.markdown(f"### {i + 1}. {title}")
        st.markdown(body)
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("◀ Back", key=f"{key}_back", disabled=i == 0, width="stretch"):
        st.session_state[idx_key] -= 1
        st.rerun()
    if c2.button("Next ▶", key=f"{key}_next", disabled=i == len(steps) - 1, width="stretch"):
        st.session_state[idx_key] += 1
        st.rerun()


# ---------------------------------------------------------------------------
# Progress + quizzes
# ---------------------------------------------------------------------------
def _progress() -> dict:
    if "progress" not in st.session_state:
        st.session_state["progress"] = {}
    return st.session_state["progress"]


def quiz(module_key: str, questions: list[dict]):
    """questions = [{"q": str, "options": [...], "answer": int, "why": str}, ...]"""
    st.divider()
    st.subheader("✅ Knowledge check")
    st.caption("Answer all questions, then check. Your best score is shown on the Home page.")
    with st.form(f"quiz_{module_key}"):
        picks = []
        for i, q in enumerate(questions):
            picks.append(st.radio(f"**Q{i + 1}. {q['q']}**", q["options"], index=None, key=f"quiz_{module_key}_{i}"))
        submitted = st.form_submit_button("Check my answers", type="primary")
    if submitted:
        score = 0
        for i, (q, pick) in enumerate(zip(questions, picks)):
            correct = q["options"][q["answer"]]
            if pick == correct:
                score += 1
                st.success(f"Q{i + 1}: Correct — {q['why']}")
            elif pick is None:
                st.warning(f"Q{i + 1}: Not answered. Correct answer: **{correct}** — {q['why']}")
            else:
                st.error(f"Q{i + 1}: You picked “{pick}”. Correct answer: **{correct}** — {q['why']}")
        prog = _progress()
        prog[module_key] = max(prog.get(module_key, 0), score / len(questions))
        if score == len(questions):
            st.balloons()
        st.metric("Your score", f"{score} / {len(questions)}")


def module_score(module_key: str) -> float | None:
    return _progress().get(module_key)
