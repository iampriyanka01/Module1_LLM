import re
from collections import Counter

import pandas as pd
import streamlit as st

from core.llm import generate, generate_many, run_live
from core.ui import analogy, cards, definition, diagram, page_header, quiz, stepper, takeaway
from core.diagrams import show_dg

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
                "🗳️ Self-Consistency", "🌳 Tree-of-Thought", "🔧 ReAct", "Which technique?", "🧾 Key takeaways", "✅ Knowledge check"])

with tabs[0]:
    analogy("First you tell an intern “just do this”, then you show examples, then you ask them to think like an expert. "
            "But when the task needs analysis, planning or justification, you need them to *think* — and show their work.",
            title="The intern")
    show_dg("m4:0")
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
    show_dg("m4:1")
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
    st.markdown("---")
    st.markdown("#### More complex CoT examples (full prompts)")
    cot_tabs = st.tabs(["Zero-shot CoT · multi-step maths", "Few-shot CoT · structured reasoning"])
    with cot_tabs[0]:
        COT0 = ("A laptop is listed at ₹60,000. A 15% discount is applied first, then 18% GST is added on the discounted price. "
                "Three friends split the final price equally. How much does each pay?\n\nLet's think step by step. "
                "End with a line: Final answer: ₹<amount>")
        st.markdown("**Full prompt**")
        st.code(COT0, language="text", wrap_lines=True)
        st.markdown("**Typical answer**")
        st.code("Step 1: Discount = 15% of 60,000 = 9,000 → price after discount = 51,000\n"
                "Step 2: GST = 18% of 51,000 = 9,180 → final price = 60,180\n"
                "Step 3: Split by 3 → 60,180 ÷ 3 = 20,060\nFinal answer: ₹20,060", language="text")
        st.caption("Without the cue, models often apply GST to the original price (₹70,800 → ₹23,600 each) — the order of steps is the trap.")
        r = run_live("🟢 Run zero-shot CoT", lambda: generate(COT0)[0], key="m4_cot0")
        if r:
            with st.container(border=True):
                st.markdown(r)
    with cot_tabs[1]:
        COT1 = """You are a QA analyst. For each bug report decide the severity (Critical/Major/Minor).
Use exactly this format:
Report: <text>
Reasoning: <1) impact, 2) how many users, 3) workaround?>
Severity: <label>

Report: App crashes on launch for all Android 14 users.
Reasoning: 1) App unusable. 2) All Android 14 users. 3) No workaround.
Severity: Critical

Report: Profile photo is blurry on tablets only.
Reasoning: 1) Cosmetic. 2) Tablet users only. 3) Works fine on phones.
Severity: Minor

Report: Payment succeeds but the order confirmation email is delayed by up to 2 hours for about 10% of users.
Reasoning:"""
        st.markdown("**Full prompt** (the examples teach the 3-point reasoning template)")
        st.code(COT1, language="text", wrap_lines=True)
        st.markdown("**Typical answer**")
        st.code("1) Money is taken and orders work, but users get no timely proof of purchase. 2) About 10% of users. "
                "3) The order can be seen in the account page, so a workaround exists.\nSeverity: Major", language="text", wrap_lines=True)
        r = run_live("🟢 Run few-shot CoT", lambda: generate(COT1)[0], key="m4_cot1")
        if r:
            with st.container(border=True):
                st.markdown("**Reasoning:** " + r.strip())

with tabs[2]:
    st.subheader("Few-shot CoT — teach by demonstration")
    show_dg("m4:2")
    analogy("Show an intern a couple of finished reports (with the reasoning), then ask for a new one. They copy the "
            "thinking pattern instead of guessing.", title="Learning by watching")
    st.markdown("Few-shot CoT = give **one or more worked examples that include the reasoning**, then the new question. "
                "The model continues the pattern: it writes `Reasoning:` and then `Job Title:`.")
    st.markdown("##### The prompt (everything the model sees)")
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
    st.markdown("##### ✅ What the model writes next (typical answer)")
    st.code("""Reasoning: The candidate works with NLP tooling (Hugging Face Transformers, spaCy) and has built
question-answering pipelines, which are core tasks of language-model / NLP engineering roles.
Job Title: NLP Engineer""", language="text")
    st.caption("Notice the model copied the *format* of the examples (Reasoning → Job Title) and the *style* of the reasoning "
               "(skills → role). Neither example said 'NLP' — it generalised from the pattern. "
               "Typical answer shown here; the live button below gives the real output.")

    def _hr():
        p = ("You are an intelligent HR assistant. Recommend the best-fit job title.\n"
             "Format: Resume / Reasoning / Job Title\n\n"
             "Resume: Python, NumPy, Matplotlib dashboards, EDA in finance.\n"
             "Reasoning: Data analysis + Python + visualisation tools used in analytics roles.\nJob Title: Data Analyst\n\n"
             "Resume: React, Node.js, REST APIs, led frontend redesign for a SaaS product.\n"
             "Reasoning: Specialises in frontend/full-stack with modern web tech.\nJob Title: Frontend Engineer\n\n"
             "Resume: Hugging Face Transformers, QA pipelines, spaCy entity recognition.\nReasoning:")
        return generate(p)[0]

    hr = run_live("▶ Run this prompt live", _hr, key="m4_hr")
    if hr:
        with st.container(border=True):
            st.caption("Live model output (continues after 'Reasoning:')")
            st.markdown("**Reasoning:** " + hr.strip())

    st.markdown("**Limitations:** manual effort · good examples are hard to find · **examples can bias the model** · "
                "doesn't scale to thousands of varied tasks (e.g. categorising 10,000 invoices where “AI API usage "
                "billing” never appeared in examples).")

    st.markdown("#### 🟢 Experiment: do examples bias the answer?")
    st.markdown("Same new case — *a customer deposits large sums and immediately withdraws them in smaller amounts* — "
                "asked **three ways**. The options now include **Structuring**, which is arguably the most precise label.")
    FRAUD_Q = ("New case: A customer deposits large sums and immediately withdraws them in smaller amounts.\n"
               "Options: Money laundering, Loan stacking, Identity theft, Structuring, Fraudulent transfer.\n"
               "Give one line of reasoning, then 'Fraud Type: <answer>'.")
    ex1 = "Example: Small withdrawals each under the reporting threshold, total significant.\nReasoning: avoiding detection.\nFraud Type: Money laundering."
    ex2 = "Example: Multiple accounts used to move funds before cash withdrawal.\nReasoning: exploiting tracking.\nFraud Type: Money laundering."
    ex3 = "Example: Multiple loans from different institutions, not repaid.\nReasoning: loans without intent to repay.\nFraud Type: Loan stacking."
    p_zero = "Predict the fraud type. Reason through the case logically and pick the most appropriate type.\n" + FRAUD_Q
    p_one = "Predict the fraud type. Here is one worked example:\n" + ex1 + "\n\n" + FRAUD_Q
    p_few = "Predict the fraud type. Here are worked examples:\n" + "\n".join([ex1, ex2, ex3]) + "\n\n" + FRAUD_Q
    VARIANTS = [
        ("A · Zero-shot CoT (no examples)", p_zero,
         "Reasoning: Depositing a large sum then withdrawing it in smaller amounts is designed to stay below reporting "
         "thresholds, which matches the textbook definition of breaking money into smaller pieces.\n**Fraud Type: Structuring**",
         "No examples → the model uses its own knowledge and picks the most *specific* label."),
        ("B · One-shot CoT (1 example)", p_one,
         "Reasoning: Like the example, the customer is breaking up money to avoid detection, so it matches money laundering.\n"
         "**Fraud Type: Money laundering**",
         "One example is enough to pull the answer towards its label and its wording ('avoid detection')."),
        ("C · Few-shot CoT (3 examples, 2 of them 'Money laundering')", p_few,
         "Reasoning: Based on the examples, small withdrawals to avoid detection resemble money laundering.\n"
         "**Fraud Type: Money laundering**",
         "The majority label in the examples dominates, and the model never even considers 'Structuring'."),
    ]
    st.markdown("###### ✅ Typical outputs (always shown — no API key needed)")
    cols = st.columns(3)
    for col, (title, pr, ans, why) in zip(cols, VARIANTS):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption("Full prompt sent to the model:")
            st.code(pr, language="text", wrap_lines=True)
            st.caption("Typical answer:")
            st.markdown(ans)
            st.caption(why)
    st.info("**Lesson:** the more (and the more one-sided) your examples, the more the model copies them — even when a better "
            "option exists. Use *balanced, diverse* examples, or keep the option list the same in every example.")
    def _bias():
        return [generate(p)[0] for _, p, _, _ in VARIANTS]

    res = run_live("▶ Run all three live (3 model calls)", _bias, key="m4_bias3")
    if res:
        cols = st.columns(3)
        for col, (title, _, _, _), out in zip(cols, VARIANTS, res):
            with col.container(border=True):
                st.caption(f"LIVE · {title}")
                st.markdown(out)
        st.caption("Live outputs can differ from the typical ones above — that variation is itself the point.")

with tabs[3]:
    st.subheader("Auto-CoT — the model becomes its own tutor")
    show_dg("m4:3")
    analogy("A junior analyst warms up on 2–3 practice reports from past projects before touching the client's data, then "
            "applies the same structure to the real task.", title="Warm-up before the real match")
    st.markdown("**The problem it solves:** few-shot CoT needs *you* to hand-write good examples for every task. "
                "Auto-CoT lets the **model write the examples itself**, then reuses them.")
    st.markdown("##### The 2 phases at a glance")
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Phase 1 — Warm-up (model teaches itself)**")
        st.markdown("You do **not** ask your real question. You ask: *“Make up 3 similar problems and solve each step by step.”*")
    with c2.container(border=True):
        st.markdown("**Phase 2 — Real task (model applies its playbook)**")
        st.markdown("You paste the self-made examples into a new prompt as few-shot examples, then ask the **real** question.")
    st.caption("Difference from zero-shot CoT: zero-shot answers your question straight away; Auto-CoT first builds its own "
               "examples (warm-up), then answers (the test).")

    st.markdown("##### ✅ Worked example (no API key needed): categorising invoices")
    AUTO_P1 = ("Generate 3 varied example problems about categorising company invoices into expense types "
               "(Travel, Meals, Office Supplies, Software, Utilities). For each: 'Item:', 2-step 'Reasoning:', 'Category:'.")
    AUTO_EX = ("Item: Train ticket to Bangalore — ₹1,850\n"
               "Reasoning: 1) Train tickets pay for moving a person between cities. 2) That is a transport cost.\n"
               "Category: Travel\n\n"
               "Item: Printer paper, 10 reams — ₹3,200\n"
               "Reasoning: 1) Paper is a consumable used in the office. 2) It is a physical supply, not software or food.\n"
               "Category: Office Supplies\n\n"
               "Item: Zoom annual plan — ₹14,000\n"
               "Reasoning: 1) Zoom is a subscription tool. 2) Recurring access to an application is a software cost.\n"
               "Category: Software")
    AUTO_TARGET = "Cloudflare annual billing — $240"
    AUTO_ANS = ("Reasoning: 1) Cloudflare provides web hosting/security services billed as a subscription. "
                "2) Paying yearly for an online service is a recurring software/cloud cost.\n**Category: Software**")
    with st.container(border=True):
        st.markdown("**① Phase 1 prompt** (no real item yet)")
        st.code(AUTO_P1, language="text")
        st.markdown("**② Model's self-generated examples** (these become the few-shot examples)")
        st.code(AUTO_EX, language="text")
        st.markdown("**③ Phase 2 prompt** = those examples + the real item")
        st.code("Here are worked examples:\n" + AUTO_EX + "\n\nNow solve in exactly the same format:\nItem: " + AUTO_TARGET, language="text")
        st.markdown("**④ Final answer** (same style as its own examples)")
        st.success(AUTO_ANS)
    st.caption("You wrote **zero** examples. “AI API usage billing” or any new expense type works the same way — just change the Phase 1 topic.")
    st.warning("Real Auto-CoT (Zhang et al., 2022) also **clusters** a large question set and picks one representative per cluster, "
               "so the self-made examples are diverse and not all the same kind. This demo shows the simplified two-phase idea from the notes.")

    st.markdown("##### 🟢 Try it live with your own task")
    domain = st.text_input("Task type", "categorising company invoices into expense types (Travel, Meals, Office Supplies, Software, Utilities)")
    target = st.text_input("Real item", AUTO_TARGET)

    def _auto():
        p1 = (f"Generate 3 varied example problems about {domain}. For each, write 'Item:', a short 'Reasoning:' "
              f"with 2 steps, and 'Category:'. Keep it compact.")
        examples = generate(p1)[0]
        p2 = f"Here are worked examples:\n{examples}\n\nNow solve in exactly the same format:\nItem: {target}"
        return examples, generate(p2)[0]

    res = run_live("▶ Run Phase 1 → Phase 2 (2 model calls)", _auto, key="m4_auto")
    if res:
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.caption("LIVE · Phase 1 · self-generated examples")
            st.markdown(res[0])
        with c2.container(border=True):
            st.caption("LIVE · Phase 2 · real item, same reasoning style")
            st.markdown(res[1])

with tabs[4]:
    st.subheader("Self-Consistency — ask several times, trust the majority")
    show_dg("m4:4")
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
    temp = c2.slider("Temperature (diversity)", 0.3, 1.5, 1.0, 0.1, key="m4_sc_t", help="Ignored on Gemini 3.x models, which use their default sampling (still varied enough to show disagreement).")
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
    st.markdown("---")
    st.markdown("#### More complex self-consistency example (full prompt + 5 paths)")
    SC_P = ("A shop sells pens in packs of 6 for ₹45, and single pens for ₹9 each. What is the cheapest way to buy at least 20 pens, "
            "and what does it cost?\n\nLet's think step by step. End with: Final answer: <cost>")
    st.markdown("**Full prompt (sent 5 times)**")
    st.code(SC_P, language="text", wrap_lines=True)
    sc_paths = [("3 packs = 18 pens (₹135) + 2 singles (₹18) = 20 pens", "₹153"),
                ("4 packs = 24 pens, cost 4 × 45", "₹180"),
                ("20 singles × 9", "₹180"),
                ("Pack is ₹7.5/pen vs ₹9 single, so use 3 packs (18) then 2 singles: 135 + 18", "₹153"),
                ("Compare 3 packs + 2 singles (153) with 4 packs (180): 153 is cheaper", "₹153")]
    for i, (r_, a_) in enumerate(sc_paths, 1):
        st.markdown(f"Path {i}: *{r_}* → **{a_}**")
    v = Counter(a for _, a in sc_paths)
    st.success(f"Majority vote: **{v.most_common(1)[0][0]}** ({v.most_common(1)[0][1]}/5) — paths 2 and 3 are valid purchases but not the cheapest; the vote filters them out.")

with tabs[5]:
    st.subheader("Tree-of-Thought — think like a planner")
    analogy("Pitching a startup: brainstorm several ideas (branches), explore each (market? business model? tech? edge?), "
            "evaluate and prune, then commit to the strongest.", title="Startup incubator")
    diagram("""
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
    st.markdown("---")
    st.markdown("#### More complex ToT example: planning with constraints (3 prompts)")
    TOT_BRIEF = ("Plan a 2-day offsite for 20 people near Pune. Budget ₹1,50,000 total. Must include one team-building activity "
                 "and one learning session. Two team members are vegetarian. No travel longer than 2 hours.")
    T1 = f"{TOT_BRIEF}\n\nStep 1 — Propose 3 clearly different themes (e.g. adventure, wellness, workshop-heavy). One line each."
    T2 = ("For each theme below, expand into a mini-plan: venue type, day-1/day-2 outline, rough cost split. "
          "Then score Fit-to-constraints, Cost and Team-benefit from 1-10 and name one risk.\n\n<THEMES FROM PROMPT 1>")
    T3 = "Compare the scored plans. Eliminate the weakest, merge any good ideas from it into the winner, and output the final plan with total cost.\n\n<MINI-PLANS FROM PROMPT 2>"
    st.markdown("**Prompt 1 — generate branches**")
    st.code(T1, language="text", wrap_lines=True)
    st.markdown("**Prompt 2 — expand and evaluate each branch**")
    st.code(T2, language="text", wrap_lines=True)
    st.markdown("**Prompt 3 — prune and select**")
    st.code(T3, language="text", wrap_lines=True)
    st.markdown("**Typical flow of results**")
    st.code("""P1 → A) Adventure (trek + camp)   B) Wellness retreat (yoga + talks)   C) Workshop-heavy (hackathon + games)
P2 → A: Fit 6 (camp meals hard for vegetarians), Cost 8, Benefit 8, risk: weather
     B: Fit 9, Cost 6, Benefit 7, risk: slow pace
     C: Fit 8, Cost 9, Benefit 6, risk: feels like work
P3 → Drop C. Keep B as base, add a half-day team-building trek from A. Final: ~₹1.42 L, resort 90 min from Pune,
     Day 1 yoga + learning session, Day 2 trek + dinner, vegetarian meals confirmed.""", language="text", wrap_lines=True)
    st.caption("Self-consistency would just vote on one answer here; ToT keeps several partial plans alive, scores them and merges the best parts.")

    def _tot2():
        a = generate(T1)[0]
        b = generate(T2.replace("<THEMES FROM PROMPT 1>", a))[0]
        c = generate(T3.replace("<MINI-PLANS FROM PROMPT 2>", b))[0]
        return a, b, c
    r = run_live("🟢 Run the 3-prompt ToT (3 calls)", _tot2, key="m4_tot_plan")
    if r:
        for t_, x in zip(["① Branches", "② Expanded and scored", "③ Pruned and selected"], r):
            with st.container(border=True):
                st.caption(t_)
                st.markdown(x)

with tabs[6]:
    st.subheader("ReAct — Reason + Act (think, use a tool, observe, repeat)")
    show_dg("m4:6")
    analogy("A detective doesn't solve the case from the armchair. They think (“who had the key?”), act (check the "
            "records), read what they find, then think again.", title="Think → look it up → think again")
    st.markdown("CoT only *thinks*. **ReAct** (Yao et al., 2022) lets the model also **act**: call a tool (search, calculator, "
                "database), read the **Observation**, and reason again. This fixes the gap the notes mention — "
                "CoT can be logical but **wrong because the model lacks the facts**.")
    st.markdown("**The loop:** `Thought → Action → Observation → Thought → … → Final Answer`. "
                "Your *code* runs the Action and pastes the Observation back; the model never executes anything itself.")

    st.markdown("##### The ReAct prompt template")
    REACT_SYS = """Answer the question using this loop. Available tools:
  lookup[topic]  - returns a short fact
  calc[expression] - evaluates arithmetic

Use exactly this format, one step at a time:
Thought: <what you need next>
Action: <tool>[<input>]
(then STOP and wait for the Observation)

When you know the answer, write:
Thought: I now know the answer
Final Answer: <answer>"""
    st.code(REACT_SYS, language="text")

    st.markdown("##### Full prompt sent on the first step")
    st.code(REACT_SYS + "\n\nQuestion: What is the population of Japan multiplied by 2?", language="text", wrap_lines=True)
    st.caption("On each later step your code appends the model's Thought/Action and the tool's Observation to this same prompt and calls the model again.")
    st.markdown("##### ✅ Worked trace (no key needed)")
    st.code("""Question: What is the population of Japan multiplied by 2?

Thought: I need Japan's population first.
Action: lookup[population of Japan]
Observation: About 124 million (2024).

Thought: Now multiply by 2.
Action: calc[124 * 2]
Observation: 248

Thought: I now know the answer
Final Answer: About 248 million.""", language="text")
    st.caption("The **Observation** lines come from tools, not the model. Without them the model would answer from memory and could be outdated.")

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**CoT**  \nThinks only. Good for logic and arithmetic. Can confidently use wrong or old facts.")
    with c2.container(border=True):
        st.markdown("**ReAct**  \nThinks + uses tools. Good when answers need fresh facts, calculations or databases. "
                    "Needs a loop in your code and a step limit.")
    st.info("This is the idea behind **function calling** (Module 8) and agents: modern APIs return a structured tool call "
            "instead of free text, but the Thought → Action → Observation loop is the same.")

    st.markdown("##### 🟢 Run a live ReAct loop")
    st.caption("Tools are simulated locally: `lookup` knows a few facts, `calc` does arithmetic. Max 5 steps = up to 5 model calls.")
    rq = st.text_input("Question", "What is the population of Japan multiplied by 2?", key="m4_react_q")
    FACTS = {"population of japan": "About 124 million (2024).", "population of india": "About 1.43 billion (2023).",
             "height of mount everest": "8,849 metres.", "capital of australia": "Canberra."}

    def _tool(name, arg):
        if name == "lookup":
            return FACTS.get(arg.strip().lower(), "No result found.")
        if name == "calc":
            if not re.fullmatch(r"[0-9+\-*/(). %]+", arg):
                return "Invalid expression."
            try:
                return str(eval(arg, {"__builtins__": {}}, {}))  # digits/operators only
            except Exception as e:  # noqa: BLE001
                return f"Error: {e}"
        return f"Unknown tool {name}."

    def _react():
        trace = f"Question: {rq}\n"
        for _ in range(5):
            out = generate(REACT_SYS + "\n\n" + trace)[0] or ""
            out = out.split("Observation:")[0].rstrip()
            trace += out + "\n"
            if "Final Answer:" in out:
                break
            m = re.search(r"Action:\s*(\w+)\[(.*)\]", out)
            if not m:
                trace += "Observation: (no valid Action found — use tool[input])\n"
                continue
            trace += f"Observation: {_tool(m.group(1), m.group(2))}\n"
        return trace

    res = run_live("▶ Run ReAct loop", _react, key="m4_react")
    if res:
        st.code(res, language="text")
        st.caption("Lines starting with “Observation:” were written by your code, everything else by the model.")


with tabs[7]:
    show_dg("m4:7", max_w=7.0, max_h=7.5)
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
with tabs[8]:
    takeaway([
        "Basic prompting guesses; advanced prompting makes the model reason, plan and justify.",
        "CoT: step-by-step reasoning (zero-shot cue, few-shot examples, or Auto-CoT self-generated examples).",
        "Few-shot examples can bias the model — choose them carefully.",
        "Self-Consistency samples multiple reasoning paths and takes the majority — reliability for single-answer tasks.",
        "Tree-of-Thought branches, evaluates and prunes — for creative and strategic problems.",
        "Bias grows with examples: zero-shot < one-shot < few-shot (one-sided) — balance and diversify your examples.",
        "Auto-CoT = warm-up (model writes its own worked examples) then test (real question); the paper also clusters questions for diversity.",
        "ReAct = Thought → Action → Observation loop: the model reasons, your code runs a tool, the result goes back in.",
        "CoT can still be wrong when the *knowledge* is wrong (e.g. tax slabs) — reasoning is not the same as facts.",
    ])
    st.markdown("##### 📚 Further reading (original papers)")
    st.markdown(
        "- [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)\n"
        "- [Zero-shot CoT: “Let's think step by step” (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916)\n"
        "- [Auto-CoT (Zhang et al., 2022)](https://arxiv.org/abs/2210.03493)\n"
        "- [Self-Consistency (Wang et al., 2022)](https://arxiv.org/abs/2203.11171)\n"
        "- [Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601)\n"
        "- [ReAct: reasoning + acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)")


with tabs[9]:
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
        {"q": "In the fraud experiment, why did the few-shot prompt answer “Money laundering” instead of “Structuring”?",
         "options": ["Structuring is not a real fraud type", "Its examples were mostly labelled Money laundering, so the model copied that pattern",
                     "Temperature was too high", "The model ignored the examples"], "answer": 1,
         "why": "One-sided examples bias the answer towards the labels the model just saw."},
        {"q": "In ReAct, who produces the “Observation” lines?",
         "options": ["The model, from memory", "Your code, after running the tool the model asked for",
                     "The user", "The tokenizer"], "answer": 1,
         "why": "The model only proposes an Action; the application executes it and feeds back the result."},
        {"q": "Which prompt gives the model one worked example with reasoning?",
         "options": ["Zero-shot CoT", "One-shot CoT", "Self-Consistency", "Tree-of-Thought"], "answer": 1,
         "why": "One-shot = exactly one demonstration; few-shot = several."},
        {"q": "What does the real Auto-CoT paper add to the simple two-phase idea?",
         "options": ["It fine-tunes the model", "It clusters questions and picks a representative from each for diverse examples",
                     "It removes reasoning", "It uses a calculator"], "answer": 1,
         "why": "Cluster-then-sample keeps self-generated examples diverse."},
    ])
