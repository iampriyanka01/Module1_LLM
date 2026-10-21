import re
from collections import Counter

import pandas as pd
import streamlit as st

from core.llm import generate, generate_many, run_live
from core.ui import analogy, cards, definition, page_header, quiz, stepper, takeaway

page_header(
    "m4",
    "Teach the model to think, not just answer: Chain-of-Thought, Auto-CoT, Self-Consistency and Tree-of-Thought.",
    [
        "Explain where basic prompting breaks down",
        "Apply zero-shot, few-shot and automatic Chain-of-Thought",
        "Use Self-Consistency to make answers reliable",
        "Use Tree-of-Thought for open-ended, multi-path problems",
        "Pick the right technique for a task",
    ],
)

tabs = st.tabs(["Why basic prompting fails", "Chain-of-Thought", "Few-shot CoT & bias", "Auto-CoT",
                "🗳️ Self-Consistency", "🌳 Tree-of-Thought", "Which technique?", "🧾 Key takeaways", "✅ Knowledge check"])

with tabs[0]:
    analogy("First you tell an intern “just do this”, then you show examples, then you ask them to think like an expert. "
            "But when the task needs analysis, planning or justification, you need them to *think* — and show their work.",
            title="The intern")
    cards([
        ("1 · Blind guessing", "“A man has 4 sons. Each son has a sister. How many children?” → basic prompt may say 8. "
                               "Correct: **5** (they share one sister)."),
        ("2 · Multi-hop logic fails", "“Raining + John forgot umbrella → wet?” needs: rain → no umbrella → exposed → wet."),
        ("3 · Black box", "An answer like “42” with no reasoning can't be trusted in healthcare, law or education."),
        ("4 · Poor generalisation", "“Add 3 apples and 2 bananas” works; “2 fruits and 3 more fruits” may stumble."),
        ("5 · Shallow on open-ended tasks", "“AI strategy to reduce pollution in a smart city” → generic list, no plan."),
    ], cols=5)
    st.markdown("#### 🟢 Try it: direct answer vs step-by-step")
    q = st.text_input("Puzzle", "A man has 4 sons. Each son has a sister. How many children does he have? "
                                "Answer with just a number.", key="m4_puzzle")

    def _both():
        direct = generate(q, temperature=0)[0]
        cot = generate(q.replace("Answer with just a number.", "") + " Let's think step by step, then give the final number.", temperature=0)[0]
        return direct, cot

    res = run_live("Compare", _both, key="m4_direct_cot")
    if res:
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.caption("Direct")
            st.markdown(res[0])
        with c2.container(border=True):
            st.caption("Chain-of-Thought")
            st.markdown(res[1])
        st.caption("Strong modern models often get simple puzzles right either way — the value of CoT is the visible, checkable reasoning.")

with tabs[1]:
    definition("Chain-of-Thought (CoT)", "prompting the model to break the problem down, reason step by step, and only then "
                                         "conclude — like Intern B who writes “25 × 2 = 50 pens/day, 50 × 3 = 150”.")
    st.markdown("**What happens inside?** LLMs don't *think* — they predict the next token. During training they saw "
                "countless texts where “Let's think step by step” was followed by multi-line reasoning. The phrase works like a "
                "director saying *“play it serious”* to an actor: it switches the model into a **reasoning pattern**. "
                "And because each written step becomes context for the next prediction, writing the steps out genuinely helps.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Zero-shot CoT** — just add the cue")
        st.code("If a train travels 300 miles in 5 hours, what is its average speed?\nLet's think step by step.", language="text")
        st.code("Step 1: Distance = 300 miles, Time = 5 hours\nStep 2: Speed = Distance / Time\n"
                "Step 3: 300 ÷ 5 = 60 mph\nFinal answer: 60 mph", language="text")
    with c2:
        st.markdown("**Limitations of zero-shot CoT**")
        st.markdown("- No control over reasoning **format/style** → inconsistent across runs\n"
                    "- Fails on **unfamiliar** tasks\n- No guarantee the steps are actually logical\n"
                    "- Can't follow *your* custom method unless you **show** it → few-shot CoT")

with tabs[2]:
    st.subheader("Few-shot CoT — teach by demonstration")
    st.code("""You are an intelligent HR assistant. Recommend the best-fit job title.
Format:  Resume: <text>  Reasoning: <why>  Job Title: <prediction>

Resume: Python, NumPy, Matplotlib dashboards, EDA in finance.
Reasoning: Data analysis + Python + visualisation tools used in analytics roles.
Job Title: Data Analyst

Resume: React, Node.js, REST APIs, led frontend redesign for a SaaS product.
Reasoning: Specialises in frontend/full-stack with modern web tech.
Job Title: Frontend Engineer

Resume: Hugging Face Transformers, QA pipelines, spaCy entity recognition.
Reasoning:""", language="text")
    st.markdown("**Limitations:** manual effort · good examples are hard to find · **examples can bias the model** · doesn't scale "
                "to thousands of varied tasks (e.g. categorising 10,000 invoices where “AI API usage billing” never appeared in examples).")
    st.markdown("#### 🟢 Experiment: do examples bias the answer?")
    st.caption("Same new case, two prompts. The few-shot one only shows Money laundering / Loan stacking examples; the "
               "other offers more options including *Structuring*.")
    with_fs = """We are analyzing cases to predict the fraud type.
Example 1: Small withdrawals each under the reporting threshold, total significant. Options: Money laundering, Loan stacking, Identity theft. Reasoning: avoiding detection. Fraud Type: Money laundering.
Example 2: Multiple accounts used to move funds before cash withdrawal. Options: Money laundering, Loan stacking, Identity theft. Reasoning: exploiting tracking. Fraud Type: Money laundering.
Example 3: Multiple loans from different institutions, not repaid. Options: Money laundering, Loan stacking, Identity theft. Reasoning: loans without intent to repay. Fraud Type: Loan stacking.
New case: A customer deposits large sums and immediately withdraws them in smaller amounts.
Options: Money laundering, Loan stacking, Identity theft, Structuring, Fraudulent transfer.
Give one line of reasoning, then 'Fraud Type: <answer>'."""
    without_fs = """Predict the fraud type. Reason through the case logically and pick the most appropriate type.
Case: A customer deposits large sums and immediately withdraws them in smaller amounts.
Options: Money laundering, Loan stacking, Identity theft, Structuring, Fraudulent transfer.
Give one line of reasoning, then 'Fraud Type: <answer>'."""
    res = run_live("Run both", lambda: (generate(with_fs, temperature=0)[0], generate(without_fs, temperature=0)[0]), key="m4_bias")
    if res:
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.caption("With few-shot CoT")
            st.markdown(res[0])
        with c2.container(border=True):
            st.caption("Without examples")
            st.markdown(res[1])
        st.caption("If the answers differ, the examples steered the model toward the labels it had just seen.")

with tabs[3]:
    st.subheader("Auto-CoT — the model becomes its own tutor")
    analogy("A junior analyst warms up on 2–3 practice reports from past projects before touching the client's data, then "
            "applies the same structure to the real task.", title="Warm-up before the real match")
    st.markdown("**Phase 1:** ask the model to *generate* a few similar problems and solve them step by step (not the target "
                "question yet).  \n**Phase 2:** feed those self-made examples back as a few-shot context and ask the real question.")
    st.caption("Difference from zero-shot CoT: in Phase 1 the model isn't answering your question — it's building its own reasoning context. It's a warm-up, not the test.")
    domain = st.text_input("Task type", "categorising company invoices into expense types (Travel, Meals, Office Supplies, Software, Utilities)")
    target = st.text_input("Real item", "Cloudflare annual billing — $240")

    def _auto():
        p1 = (f"Generate 3 varied example problems about {domain}. For each, write 'Item:', a short 'Reasoning:' "
              f"with 2 steps, and 'Category:'. Keep it compact.")
        examples = generate(p1, temperature=0.7)[0]
        p2 = f"Here are worked examples:\n{examples}\n\nNow solve in exactly the same format:\nItem: {target}"
        return examples, generate(p2, temperature=0)[0]

    res = run_live("Run Phase 1 → Phase 2", _auto, key="m4_auto")
    if res:
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.caption("Phase 1 · self-generated examples")
            st.markdown(res[0])
        with c2.container(border=True):
            st.caption("Phase 2 · real item, same reasoning style")
            st.markdown(res[1])

with tabs[4]:
    st.subheader("Self-Consistency — ask several times, trust the majority")
    analogy("Unsure about a math answer? Solve it again a different way, and again. If you keep getting the same answer, "
            "you're confident.", title="Double-checking")
    st.markdown("LLMs are **non-deterministic**, so even CoT can produce different answers. Self-consistency: "
                "**(1)** run the CoT prompt N times with some randomness, **(2)** collect the final answers, **(3)** pick the most frequent.")
    st.markdown("##### Offline example from the notes (security email)")
    samples = [("The sender found a major vulnerability that could lead to a data breach…", "IMPORTANT"),
               ("…but they gave no proof of concept and it is not easy to use…", "NOT IMPORTANT"),
               ("The sender is likely a hacker and may have access to the system…", "IMPORTANT")]
    for i, (r, a) in enumerate(samples, 1):
        st.markdown(f"Path {i}: *{r}* → **{a}**")
    votes = Counter(a for _, a in samples)
    st.success(f"Majority vote: **{votes.most_common(1)[0][0]}** ({votes.most_common(1)[0][1]}/3)")

    st.markdown("##### 🟢 Live self-consistency")
    question = st.text_area("Question", "When I was 6 my sister was half my age. Now I'm 70. How old is my sister?", key="m4_sc_q")
    c1, c2 = st.columns(2)
    n = c1.slider("Number of reasoning paths", 3, 9, 5, key="m4_sc_n")
    temp = c2.slider("Temperature (diversity)", 0.3, 1.5, 1.0, 0.1, key="m4_sc_t")
    prompt = question + "\nLet's think step by step. End with a final line exactly like: Final answer: <answer>"

    def _sc():
        outs = generate_many([prompt] * n, temperature=temp)
        finals = []
        for txt, err, _ in outs:
            if txt:
                m = re.findall(r"final answer\s*[:\-]\s*\**(.+)", txt, flags=re.I)
                finals.append(m[-1].strip(" *.") if m else "(no final line)")
            else:
                finals.append("(error)")
        return outs, finals

    res = run_live(f"Generate {n} reasoning paths", _sc, key="m4_sc")
    if res:
        outs, finals = res
        counts = Counter(finals)
        winner, wc = counts.most_common(1)[0]
        c1, c2 = st.columns([1, 2])
        c1.metric("Majority answer", winner, f"{wc}/{len(finals)} paths agree")
        c2.bar_chart(pd.Series(counts, name="votes"), height=200, horizontal=True)
        for i, (o, f) in enumerate(zip(outs, finals), 1):
            with st.expander(f"Path {i} → {f}"):
                st.markdown(o[0] or o[1])
    st.info("Limitation: Self-consistency is great when there's **one** right answer (math, facts, classification). For "
            "open-ended tasks it's like flipping a coin 10 times — you need Tree-of-Thought.")

with tabs[5]:
    st.subheader("Tree-of-Thought — think like a planner")
    analogy("Pitching a startup: brainstorm several ideas (branches), explore each (market? business model? tech? edge?), "
            "evaluate and prune, then commit to the strongest.", title="Startup incubator")
    st.graphviz_chart("""
digraph { rankdir=TB; node [shape=box style="rounded,filled" fontname="Helvetica" fillcolor="#f1f5f9"];
 Q [label="Problem" fillcolor="#dbeafe"]; A [label="Idea 1: grocery app"]; B [label="Idea 2: wellness platform"]; C [label="Idea 3: smart water bottle"];
 A1 [label="Crowded market ✗" fillcolor="#fee2e2"]; B1 [label="High demand, content-heavy ~" fillcolor="#fef9c3"]; C1 [label="Niche + health + IoT ✓" fillcolor="#dcfce7"];
 Q->A; Q->B; Q->C; A->A1; B->B1; C->C1; F [label="Pick best path" fillcolor="#86efac"]; C1->F; }""")
    problem = st.text_area("Open-ended problem", "Generate plot branches for a mystery novel where the detective is the murderer.", key="m4_tot")
    branches = st.slider("Branches", 2, 4, 3, key="m4_tot_b")

    def _tot():
        gen = generate(f"{problem}\nPropose {branches} clearly different approaches. Number them 1..{branches}. "
                       f"For each give 2 sentences: the idea and how it would develop.", temperature=1.0)[0]
        judge = generate(
            "You are evaluating options. For each numbered approach below, score Originality, Feasibility and Impact from 1-10, "
            "note one weakness, then choose the best and explain in 2 sentences.\n\n" + gen, temperature=0)[0]
        return gen, judge

    res = run_live("🟢 Expand branches → evaluate → select", _tot, key="m4_tot_run")
    if res:
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.caption("Step 1–2 · generate and expand branches")
            st.markdown(res[0])
        with c2.container(border=True):
            st.caption("Step 3 · evaluate, prune, select")
            st.markdown(res[1])

with tabs[6]:
    st.dataframe(pd.DataFrame({
        "": ["Thinking style", "Best for", "How", "Weakness"],
        "Chain-of-Thought": ["Linear, step by step", "Math, logic, factual QA", "“Let's think step by step” / examples", "Can be inconsistent"],
        "Self-Consistency": ["Many linear paths + vote", "One correct answer, unstable outputs", "Sample N times, majority wins", "Costs N× calls; useless for open tasks"],
        "Tree-of-Thought": ["Branching + evaluation", "Creative, strategic, planning", "Generate → expand → score → prune → choose", "Most calls, most complex"],
    }).set_index(""), width="stretch")
    st.markdown("#### Pick a technique")
    task = st.selectbox("Your task", ["Calculate compound interest over 5 years", "Decide if a support ticket is urgent (answers flip between runs)",
                                      "Plan a 3-day event with budget and venue constraints", "Categorise 10,000 invoices in a new domain",
                                      "Explain why a loan was rejected, transparently"])
    rec = {
        "Calculate compound interest over 5 years": "**Chain-of-Thought** — multi-step arithmetic benefits from visible steps (or better: a calculator tool, Module 8).",
        "Decide if a support ticket is urgent (answers flip between runs)": "**Self-Consistency** — one right label, unstable outputs → majority vote.",
        "Plan a 3-day event with budget and venue constraints": "**Tree-of-Thought** — many valid plans; explore, compare, prune.",
        "Categorise 10,000 invoices in a new domain": "**Auto-CoT** — let the model generate its own worked examples instead of hand-writing them.",
        "Explain why a loan was rejected, transparently": "**Chain-of-Thought** (few-shot, with your reasoning template) — transparency matters.",
    }
    st.success(rec[task])


# ---------------------------------------------------------------------------
with tabs[7]:
    takeaway([
        "Basic prompting guesses; advanced prompting makes the model reason, plan and justify.",
        "CoT: step-by-step reasoning (zero-shot cue, few-shot examples, or Auto-CoT self-generated examples).",
        "Few-shot examples can bias the model — choose them carefully.",
        "Self-Consistency samples multiple reasoning paths and takes the majority — reliability for single-answer tasks.",
        "Tree-of-Thought branches, evaluates and prunes — for creative and strategic problems.",
    ])


with tabs[8]:
    quiz("m4", [
        {"q": "What does “Let's think step by step” mainly do?",
         "options": ["Gives the model internet access", "Triggers a learned multi-step reasoning pattern", "Lowers the temperature", "Adds memory"],
         "answer": 1, "why": "The cue nudges the model into the reasoning pattern it saw in training."},
        {"q": "Self-Consistency picks the final answer by…",
         "options": ["Taking the longest answer", "Majority vote over several sampled reasoning paths", "Asking a human", "Using temperature 0"],
         "answer": 1, "why": "The most frequent final answer is assumed to be most reliable."},
        {"q": "Which technique fits “write three alternative endings and pick the most original”?",
         "options": ["Zero-shot", "Self-Consistency", "Tree-of-Thought", "One-shot"], "answer": 2,
         "why": "Multiple valid paths that must be compared and pruned."},
        {"q": "Phase 1 of Auto-CoT is…",
         "options": ["Solving the target question", "The model generating and solving its own example problems",
                     "A human writing examples", "Fine-tuning the model"], "answer": 1,
         "why": "It's a warm-up that builds a self-made few-shot context."},
    ])
