import json
import math
import re
import time
from collections import Counter

import pandas as pd
import streamlit as st

from core.course_prompts import EVALUATOR_SYSTEM_PROMPT, STUDY_BUDDY_SYSTEM_PROMPT, safe_parse_json
from core.llm import (call_with_retry, friendly_error, generate, get_client, get_model, llm_ready, log_call, response_text,
                      run_live)
from core.ui import analogy, cards, compare, definition, page_header, quiz, stepper, takeaway

page_header(
    "m9",
    "Generating is half the job. Learn to detect hallucinations, stress-test quirks, measure quality and improve it systematically.",
    [
        "Explain the five root causes of hallucination and classify errors",
        "Describe structural limits, input sensitivity, behavioural quirks and numerical fragility",
        "Design quirk-surface tests",
        "Use traditional and LLM-era metrics, the metric layering model and rubrics",
        "Run model-based evaluation (LLM-as-judge)",
        "Improve prompts with the try → learn → tweak → repeat cycle",
    ],
)

tabs = st.tabs(["🌀 Hallucinations", "Model limitations", "🧪 Quirk-surface tests", "📐 Traditional metrics",
                "🏗️ Metric layering", "🧑‍⚖️ Rubrics & human judging", "🤖 LLM-as-judge (your Study Buddy)",
                "🔁 Refinement cycle", "✏️ Prompt editing strategies", "🧾 Key takeaways", "✅ Knowledge check"])

# ---------------------------------------------------------------------------
with tabs[0]:
    definition("AI hallucination", "output that sounds plausible but is false, unsupported, or inconsistent with the data "
               "provided — like an overconfident intern who'd rather answer than admit they don't know.")
    st.markdown("#### Five root causes")
    cards([
        ("1 · Training data gaps & biases", "Missing or skewed data → the model pattern-fills. Ask about new Indian labour laws and it "
                                            "adapts mostly-Western patterns."),
        ("2 · Next-token objective", "Trained to predict the likely next word, not to check truth. **Fluency ≠ truth.**"),
        ("3 · Overconfidence", "Fine-tuning rewards complete, helpful answers, so it rarely says “I don't know”."),
        ("4 · No grounding / retrieval", "Static model, dynamic world: “90 days visa-free in Germany” after the rule changed to 30."),
        ("5 · Prompt ambiguity", "“Tell me about Germany's visa rules” — tourist? business? which passport? It assumes."),
    ], cols=5)
    st.markdown("#### Taxonomy")
    compare("Factual fabrication", "Invented names, statistics or events presented as true. 2023: a US lawyer filed a brief with "
            "ChatGPT-invented court cases and was penalised.",
            "Citation hallucination", "Fake but real-looking references: “Smith, J. & Wong, L. (2019). *Advances in AI Security*. "
            "J. Emerging Tech, 34(2)…” — authors, volume and DOI don't exist.", bad_good=False)
    st.markdown("#### 🟢 Experiment: the non-existent physics term")
    st.caption("“Cycloidal inverted electromagnon” is not an established concept. Does the model invent one? Does an honesty instruction help?")
    q = "What is a cycloidal inverted electromagnon? Provide references and citations."
    guard = (" If this is not an established concept or you are not sure the references exist, say so explicitly instead of "
             "inventing details or citations.")
    res = run_live("Ask with and without an honesty instruction", lambda: (generate(q)[0], generate(q + guard)[0]), key="m9_halluc")
    if res:
        c1, c2 = st.columns(2)
        with c1.container(border=True, height=380):
            st.caption("Plain prompt")
            st.markdown(res[0])
        with c2.container(border=True, height=380):
            st.caption("With “say if you're not sure”")
            st.markdown(res[1])
        st.caption("Modern models are better at refusing, but check any citations they give — that's the habit this module builds.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("A good evaluator thinks like a detective: not just *what* went wrong, but *why*.")
    with st.expander("1 · Structural limitations — what the model can't know, fetch or hold", expanded=True):
        st.markdown("- **Stale knowledge** — training cutoff: misses last quarter's earnings, amended laws, recalled drugs.\n"
                    "- **No real-time sensing** — “Weather in Mumbai now?” → “Mumbai is typically humid…” instead of 31°C and raining. "
                    "Benchmarks should reward *use tool → fetch → answer* and penalise invented specifics.\n"
                    "- **Context window boundaries** — skipped clauses in a 120-page manual, forgotten troubleshooting steps.")
    with st.expander("2 · Input sensitivity — small wording, big difference"):
        st.markdown("- **Phrasing effects** — “Tell me about WWII” (story) vs “List three key political events in bullets” (precision).\n"
                    "- **Order effects in few-shot** — the last example dominates; put the humorous one last and the bot cracks jokes.\n"
                    "- **Instruction hierarchy conflicts** — system says “concise”, user says “detailed”; the system usually wins.")
    with st.expander("3 · Behavioural patterns in generation"):
        st.markdown("- **Verbosity drift** — essays when you asked for a summary, or two lines when you wanted depth.\n"
                    "- **Hedging / over-caution** — disclaimers instead of breathing tips for presentation nerves.\n"
                    "- **Mode collapse at low temperature** — “The sun was shining. The sky was blue.” every single time.")
    with st.expander("4 · Numerical & logical fragility"):
        st.markdown("- **Inconsistent arithmetic** — it predicts what a correct-looking number looks like; ROI and loan maths go wrong.\n"
                    "- **Schema misalignment in tables** — columns drift (“Status” becomes “Available?”), CSV commas break.")
    analogy("Phones are stress-tested for overheating before shipping. LLM apps get quirk-surface tests before deployment.",
            title="Product testing")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Proactively expose weak spots")
    st.markdown("Three test families: **contrastive prompts** (same request, different wording), **numeric stress tests** "
                "(conversions, running totals), **multi-turn memory checks**.")
    tests = [
        {"family": "Numeric stress", "name": "Unit conversion", "prompt": "Convert 25 miles per hour to kilometers per hour. Reply with only the number rounded to 2 decimals.",
         "check": lambda t: bool(re.search(r"40\.2[0-4]", t)), "expect": "≈ 40.23–40.24"},
        {"family": "Numeric stress", "name": "Running total", "prompt": "A person starts with $100, spends $25, earns $50, then spends $30. Final balance? Reply with only the number.",
         "check": lambda t: bool(re.search(r"\b95\b", t)), "expect": "95"},
        {"family": "Numeric stress", "name": "Litres → ml", "prompt": "How many milliliters are in 2.5 liters? Reply with only the number.",
         "check": lambda t: "2500" in t.replace(",", ""), "expect": "2500"},
        {"family": "Contrastive", "name": "Active voice", "prompt": "Summarize the effects of climate change on polar bears in exactly 2 sentences.",
         "check": lambda t: len(re.findall(r"[.!?](\s|$)", t.strip())) == 2, "expect": "exactly 2 sentences"},
        {"family": "Contrastive", "name": "Passive voice", "prompt": "Summarize how polar bears are affected by climate change in exactly 2 sentences.",
         "check": lambda t: len(re.findall(r"[.!?](\s|$)", t.strip())) == 2, "expect": "exactly 2 sentences"},
    ]
    st.dataframe(pd.DataFrame([{k: v for k, v in t.items() if k in ("family", "name", "prompt", "expect")} for t in tests]),
                 hide_index=True, width="stretch")

    def _suite():
        rows = []
        for t in tests:
            txt, _, secs = generate(t["prompt"], temperature=0)
            rows.append({"test": t["name"], "family": t["family"], "pass": "✅" if t["check"](txt) else "❌",
                         "output": txt.strip()[:160], "latency s": round(secs, 2)})
        chat = get_client().chats.create(model=get_model())
        call_with_retry(lambda: chat.send_message("My favorite color is blue. I also enjoy hiking in the mountains."))
        call_with_retry(lambda: chat.send_message("Tell me a fun fact about owls in one sentence."))
        ans = response_text(call_with_retry(lambda: chat.send_message(
            "Suggest ONE gift for a friend who likes my favorite color and my favourite activity. One line.")))
        ok = "blue" in ans.lower() and any(w in ans.lower() for w in ("hik", "trek", "trail", "backpack", "mountain", "outdoor"))
        rows.append({"test": "Gift suggestion (turn 3)", "family": "Multi-turn memory", "pass": "✅" if ok else "❌",
                     "output": ans.strip()[:160], "latency s": None})
        return rows

    res = run_live("🟢 Run the quirk-surface test suite", _suite, key="m9_suite", spinner="Running 6 tests…")
    if res:
        df = pd.DataFrame(res)
        st.metric("Pass rate", f"{(df['pass'] == '✅').mean():.0%}")
        st.dataframe(df, hide_index=True, width="stretch")
        st.caption("Automated checks like these become your Tier-1 signals. Re-run after every prompt or model change (regression testing).")

# ---------------------------------------------------------------------------
def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


with tabs[3]:
    st.subheader("Classic metrics — great for yes/no and numbers")
    st.markdown("#### Confusion matrix → accuracy, precision, recall, F1 (spam filter)")
    c1, c2 = st.columns(2)
    with c1:
        tp = st.number_input("TP — spam caught", 0, 10000, 5)
        fn = st.number_input("FN — spam missed (in inbox)", 0, 10000, 3)
    with c2:
        fp = st.number_input("FP — good mail sent to spam", 0, 10000, 2)
        tn = st.number_input("TN — good mail kept", 0, 10000, 90)
    total = tp + tn + fp + fn or 1
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    st.dataframe(pd.DataFrame({"Predicted SPAM": [f"TP = {tp}", f"FP = {fp}"], "Predicted NOT SPAM": [f"FN = {fn}", f"TN = {tn}"]},
                              index=["Actual SPAM", "Actual NOT SPAM"]), width="stretch")
    m = st.columns(4)
    m[0].metric("Accuracy", f"{acc:.1%}", help="(TP+TN)/total — overall correctness")
    m[1].metric("Precision", f"{prec:.1%}", help="TP/(TP+FP) — of everything flagged, how much was really spam? (avoid false alarms)")
    m[2].metric("Recall", f"{rec:.1%}", help="TP/(TP+FN) — of all real spam, how much did we catch? (avoid misses)")
    m[3].metric("F1", f"{f1:.2f}", help="Harmonic mean of precision and recall")
    if st.button("Show the accuracy paradox"):
        st.warning("99 normal emails + 1 spam, and a lazy model that always says “Not spam”: accuracy **99%**, recall **0%**. "
                   "Accuracy misleads when one class is rare.")

    st.markdown("#### BLEU-style word overlap — and why it fails for LLMs")
    ref = st.text_input("Human reference", "The cat sat on the mat")
    cand = st.text_input("Model output", "Kitty rested on the floor rug")
    rt, ct = ref.lower().split(), cand.lower().split()
    scores = []
    for n in (1, 2):
        cn, rn = ngrams(ct, n), ngrams(rt, n)
        overlap = sum(min(c, rn[g]) for g, c in cn.items())
        scores.append(overlap / max(sum(cn.values()), 1))
    bp = math.exp(1 - len(rt) / len(ct)) if 0 < len(ct) < len(rt) else 1.0
    bleu2 = bp * math.sqrt(scores[0] * scores[1]) if all(scores) else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Unigram precision", f"{scores[0]:.0%}")
    c2.metric("Bigram precision", f"{scores[1]:.0%}")
    c3.metric("BLEU-2 (simplified)", f"{bleu2:.2f}")
    st.caption("Same meaning, different words → low score. BLEU (word match), ROUGE (content coverage) and METEOR (synonyms) "
               "compare to a reference — but a creative poem has no single reference, and none of them checks truth.")

    st.markdown("#### Regression metrics — MAE vs RMSE (house prices, ₹ lakh)")
    errs = st.data_editor(pd.DataFrame({"actual": [51.0, 29.5, 80.0, 45.0], "predicted": [50.0, 30.0, 70.0, 45.5]}), key="m9_reg")
    diff = errs["predicted"] - errs["actual"]
    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{diff.abs().mean():.2f}", help="Average absolute miss")
    c2.metric("RMSE", f"{math.sqrt((diff ** 2).mean()):.2f}", help="Punishes big misses more — try making one prediction way off")
    st.info("Why these fall short for LLMs: no single right answer, multiple sub-tasks (intent, retrieval, generation), and they "
            "can't judge helpfulness, tone, creativity or **truthfulness** — the lawyer's fake cases would score fine on word overlap.")

# ---------------------------------------------------------------------------
KYOTO_SPOTS = {"Fushimi Inari Shrine", "Kinkaku-ji", "Kiyomizu-dera", "Arashiyama Bamboo Grove", "Gion", "Nishiki Market",
               "Ginkaku-ji", "Philosopher's Path", "Nijo Castle", "Tofuku-ji", "Pontocho", "Sanjusangen-do"}
with tabs[4]:
    st.subheader("Three tiers: automated signals → human judgment → aggregate decision")
    st.markdown("Example: an AI travel planner answers “Plan a 3-day trip to Kyoto focused on culture, temples and food.”")
    itinerary = {
        "days": [
            {"day": 1, "activities": [{"time": "08:00", "place": "Fushimi Inari Shrine"}, {"time": "12:30", "place": "Nishiki Market"},
                                      {"time": "16:00", "place": "Gion"}]},
            {"day": 2, "activities": [{"time": "09:00", "place": "Kinkaku-ji"}, {"time": "13:00", "place": "Arashiyama Bamboo Grove"},
                                      {"time": "13:00", "place": "Tokyo Tower"}]},
            {"day": 3, "activities": [{"time": "09:00", "place": "Kiyomizu-dera"}, {"time": "14:00", "place": "Fushimi Inari Shrine"}]},
        ]
    }
    raw = st.text_area("AI-generated itinerary (JSON) — edit it and re-check", json.dumps(itinerary, indent=1), height=230)
    st.markdown("##### Tier 1 · Automated signals")
    checks = []
    try:
        it = json.loads(raw)
        days = it.get("days", [])
        checks.append(("Structure valid (3 days, each with activities)", len(days) == 3 and all(d.get("activities") for d in days)))
        places = [a.get("place") for d in days for a in d.get("activities", [])]
        bad = [p for p in places if p not in KYOTO_SPOTS]
        checks.append((f"Destination accuracy (all in Kyoto){' — not in Kyoto: ' + ', '.join(bad) if bad else ''}", not bad))
        dups = [p for p, c in Counter(places).items() if c > 1]
        checks.append((f"No repetition{' — repeated: ' + ', '.join(dups) if dups else ''}", not dups))
        clash = [d["day"] for d in days if len({a.get("time") for a in d["activities"]}) < len(d["activities"])]
        checks.append((f"Time allocation (no clashes){' — clash on day ' + ', '.join(map(str, clash)) if clash else ''}", not clash))
        checks.append(("Length: 3–4 activities per day", all(3 <= len(d["activities"]) <= 4 for d in days)))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        checks.append((f"Schema check — invalid JSON/structure: {e}", False))
    for name, ok in checks:
        st.markdown(f"{'✅' if ok else '❌'} {name}")
    auto = sum(ok for _, ok in checks) / len(checks)
    st.markdown("##### Tier 2 · Human judgments (you are the reviewer)")
    c = st.columns(4)
    intent = c[0].slider("Intent fit", 1, 5, 4)
    engage = c[1].slider("Engagement", 1, 5, 3)
    culture = c[2].slider("Cultural sensitivity", 1, 5, 5)
    novelty = c[3].slider("Novelty", 1, 5, 3)
    st.markdown("##### Tier 3 · Aggregation & decision")
    c = st.columns(3)
    w1 = c[0].slider("Weight: structure & factuality", 0, 100, 40, format="%d%%")
    w2 = c[1].slider("Weight: human quality", 0, 100, 40, format="%d%%")
    w3 = c[2].slider("Weight: cultural/contextual", 0, 100, 20, format="%d%%")
    wsum = (w1 + w2 + w3) or 1
    overall = (auto * 5 * w1 + (intent + engage + novelty) / 3 * w2 + culture * w3) / wsum
    st.metric("Overall score", f"{overall:.2f} / 5", "ship" if overall >= 4 and auto == 1 else "flag for prompt fix", delta_color="normal" if overall >= 4 and auto == 1 else "inverse")
    st.caption("Real systems run Tier 1 on every output (e.g. 5,000 itineraries), sample 5–10% for Tier 2, and track Tier 3 on dashboards.")

# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Qualitative assessment — the human touch")
    st.markdown("Quantitative metrics (accuracy, relevance, coherence, creativity, format compliance, length & latency) tell you "
                "*what* happened. Humans tell you whether it *helped*.")
    st.markdown("#### Build a fair rubric")
    st.markdown("1. **Tie to the problem statement** (mission “make learning simple” → criterion *clarity*)  \n"
                "2. **Observable evidence** (Accuracy 5 = all facts correct; 1 = factual errors)  \n"
                "3. **Weighting** (what matters most)  \n4. **Anchor examples** (gold / adequate / poor)")
    c1, c2, c3 = st.columns(3)
    a = c1.slider("Accuracy score", 1, 5, 5, key="rb_a")
    t = c2.slider("Tone score", 1, 5, 3, key="rb_t")
    d = c3.slider("Depth score", 1, 5, 4, key="rb_d")
    wa = c1.slider("Accuracy weight", 0.0, 1.0, 0.4, 0.05)
    wt = c2.slider("Tone weight", 0.0, 1.0, 0.3, 0.05)
    wd = c3.slider("Depth weight", 0.0, 1.0, 0.3, 0.05)
    tot = wa + wt + wd or 1
    st.metric("Weighted overall", f"{(a * wa + t * wt + d * wd) / tot:.2f}",
              help="Notes' example: 5×0.4 + 3×0.3 + 4×0.3 = 4.1")
    if abs(tot - 1) > 1e-6:
        st.caption(f"Weights sum to {tot:.2f} — normalised automatically.")
    st.markdown("#### Anchor examples — “Explain gravity”")
    st.dataframe(pd.DataFrame([
        ("5 · Gold", "Gravity is the force that pulls things toward the Earth, like why an apple falls from a tree. It's what keeps planets orbiting the sun."),
        ("3 · Adequate", "Gravity is a force that pulls things. It's why things fall."),
        ("1 · Poor", "Gravity is like magic pulling stuff down."),
    ], columns=["Score", "Answer"]), hide_index=True, width="stretch")
    st.markdown("#### Comparative judging (A/B) — which would you rather read as a student?")
    ab = st.radio("Q: Explain gravity", ["A: Gravity is the universal force of attraction exerted by all matter upon all other matter.",
                                         "B: Gravity is why an apple falls from a tree — an invisible force pulling things toward the Earth."],
                  index=None, key="m9_ab")
    if ab:
        st.success("Most reviewers pick **B** — simpler and relatable. A/B judging is fast and great for comparing prompt versions.")
        note = st.text_input("Annotation note — *why* did you pick it?", placeholder="e.g. Uses a real-life example; A uses jargon for a 6th grader")
        if note:
            st.info(f"Saved note: “{note}” — annotations turn scores into concrete fixes for the prompt.")

# ---------------------------------------------------------------------------
def judge(question: str, answer: str) -> tuple[dict, str]:
    """Stateless evaluation: a fresh call per answer, so earlier verdicts can't bias the score."""
    eval_input = (f"Original Prompt (User Query):\n{question}\n\nTarget Model's Output:\n{answer}\n\n"
                  "Please evaluate using the rubric and return ONLY the JSON specified.")
    txt, resp, secs = generate(eval_input, system=EVALUATOR_SYSTEM_PROMPT, temperature=0)
    data = safe_parse_json(txt)
    missing = [k for k in ("scores", "feedback", "overall_summary") if k not in data]
    if missing:
        raise ValueError(f"Evaluator JSON missing key(s): {missing}")
    return data, txt


with tabs[6]:
    st.subheader("Model-based evaluation with your own Study Buddy + Evaluator")
    st.markdown("**Target model** (Study Buddy) answers → **Evaluator model** applies the rubric → scores + feedback. "
                "Scalable, consistent and 24/7 — but it can't fully capture nuance, and it can be wrong too.")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box style="rounded,filled" fontname="Helvetica"];
 Q [label="Student question" fillcolor="#f1f5f9"]; T [label="Target model\\nStudy Buddy" fillcolor="#dbeafe"];
 E [label="Evaluator model\\n(rubric → strict JSON)" fillcolor="#fef9c3"]; P [label="safe_parse_json()\\n+ key check" fillcolor="#dcfce7"];
 S [label="Scores · feedback · summary" fillcolor="#86efac"]; Q->T->E->P->S; Q->E [style=dashed label=" original prompt"]; }""")
    with st.expander("Prompts used (from your prompts.py)"):
        st.markdown("**STUDY_BUDDY_SYSTEM_PROMPT**")
        st.code(STUDY_BUDDY_SYSTEM_PROMPT.strip(), language="text")
        st.markdown("**EVALUATOR_SYSTEM_PROMPT**")
        st.code(EVALUATOR_SYSTEM_PROMPT.strip(), language="text")
    with st.expander("🔍 Two things worth fixing in the original project"):
        st.markdown(
            "1. **Rubric ↔ schema mismatch.** The rubric defines 5 dimensions (accuracy, clarity & coherence, tone, relevance, "
            "conciseness) but the JSON asks for 6 scores, splitting *clarity* and *coherence*. Evaluators then guess what "
            "“clarity” means — exactly what *Observable evidence* in a fair rubric prevents.\n"
            "2. **Stateful judge.** `create_evaluator_chat()` is called once and reused, so each verdict sees earlier "
            "verdicts in its chat history — the opposite of the docstring's “judge each answer independently”. Here each "
            "evaluation is a fresh, stateless call.")
    mode = st.radio("What to evaluate", ["Ask Study Buddy a question", "Paste my own answer (try a wrong one!)"], horizontal=True)
    question = st.text_input("Student question", "Why does the moon change shape?", key="m9_q")
    if mode.startswith("Paste"):
        answer_in = st.text_area("Answer to grade", "The moon appears to change shape because of Earth's shadow falling on it.", key="m9_a")

    def _run():
        if mode.startswith("Ask"):
            txt, _, _ = generate(question, system=STUDY_BUDDY_SYSTEM_PROMPT)
        else:
            txt = answer_in
        verdict, raw = judge(question, txt)
        return txt, verdict, raw

    res = run_live("🟢 Answer & evaluate", _run, key="m9_judge", spinner="Study Buddy is answering, then the evaluator grades…")
    if res:
        ans, verdict, raw = res
        with st.chat_message("assistant", avatar="🎓"):
            st.markdown(ans)
        scores = {k: float(v) for k, v in verdict["scores"].items() if isinstance(v, (int, float, str)) and str(v).replace(".", "", 1).isdigit()}
        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("Average score", f"{sum(scores.values()) / max(len(scores), 1):.2f} / 5")
            st.bar_chart(pd.Series(scores, name="score"), height=260, horizontal=True)
        with c2:
            st.dataframe(pd.DataFrame({"dimension": list(verdict["feedback"]), "feedback": list(verdict["feedback"].values())}),
                         hide_index=True, width="stretch")
            st.info(verdict["overall_summary"])
        with st.expander("Raw evaluator output"):
            st.code(raw, language="json")

# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Try → learn → tweak → repeat")
    analogy("A chef tastes, adjusts the seasoning, tastes again. Prompts are never “set and forget”.", title="Chef")
    stepper("refine", [
        ("Define target behaviour", "From the problem statement & rubric: summaries of 50-page papers must be **< 500 words** and keep **all key findings**."),
        ("Generate baseline outputs", "Run the current prompt on N representative samples (e.g. 10 long papers) — the “before” picture."),
        ("Score against metrics", "Length? Missing key ideas? Easy to read? Record scores and notes."),
        ("Diagnose failure patterns", "Pattern: very long inputs → overly long summaries that skip sections. The model can't prioritise at volume."),
        ("Edit the prompt", "Role + explicit constraint + focus: “You are an expert academic summarizer… ~400 words… critical findings, methods, "
                            "conclusions… no introductory fluff.” Optionally add few-shot examples."),
        ("Re-test (A/B) & compare", "Run old (A) vs new (B) on the same inputs, re-measure. Not good enough? Loop again."),
    ])
    st.markdown("#### 🟢 Mini A/B test")
    source = st.text_area("Long-ish input", (
        "Researchers studied 1,200 students across 40 schools for two years. Students who used spaced repetition apps for 15 minutes "
        "daily scored 18% higher on final exams than the control group. The effect was strongest in mathematics (24%) and weakest in "
        "literature (7%). However, students who used the apps for more than 45 minutes daily showed no additional benefit and reported "
        "higher fatigue. The authors recommend short, consistent sessions and note limitations: self-reported usage data and a "
        "sample drawn only from urban schools. ") * 2, height=150)
    key_points = st.text_input("Key facts the summary must contain (comma-separated)", "18%, mathematics, 45 minutes, urban")
    pa = st.text_area("Prompt A (baseline)", "Summarize this:\n{text}", key="m9_pa")
    pb = st.text_area("Prompt B (refined)", "You are an expert academic summarizer. In at most 60 words, state the key finding with numbers, "
                      "where it was strongest, the over-use caveat, and one limitation. No introduction.\n{text}", key="m9_pb")
    kp = [k.strip().lower() for k in key_points.split(",") if k.strip()]

    def _ab():
        out = []
        for name, p in (("A", pa), ("B", pb)):
            txt, _, secs = generate(p.replace("{text}", source), temperature=0)
            out.append({"prompt": name, "words": len(txt.split()), "key facts kept": f"{sum(k in txt.lower() for k in kp)}/{len(kp)}",
                        "latency s": round(secs, 2), "output": txt})
        return out

    res = run_live("Run A vs B", _ab, key="m9_ab_run")
    if res:
        st.dataframe(pd.DataFrame(res).drop(columns="output"), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        for col, r in zip((c1, c2), res):
            with col.container(border=True):
                st.caption(f"Prompt {r['prompt']}")
                st.markdown(r["output"])

# ---------------------------------------------------------------------------
with tabs[8]:
    strategies = {
        "1 · Explicit task framing": ("The system prompt is the AI component's constructor: purpose, rules and limits set before any "
                                      "user data arrives — like your Study Buddy prompt (style, tone, safety & scope).",
                                      'chat = client.chats.create(\n    model=MODEL,\n    config=types.GenerateContentConfig(system_instruction=STUDY_BUDDY_SYSTEM_PROMPT),\n)'),
        "2 · Output schema instructions": ("A strict API contract: the rest of your code expects exact JSON. Deviation breaks the pipeline.",
                                           'prompt = f"""From the invoice text below, extract the entities.\nYour response MUST be a single valid JSON object, no text before or after:\n'
                                           '{{"invoice_id": "string", "vendor_name": "string", "total_amount": "float", "due_date": "YYYY-MM-DD"}}\nINVOICE TEXT:\n---\n{invoice_text}\n---"""\n'
                                           'data = json.loads(client.models.generate_content(model=MODEL, contents=prompt,\n'
                                           '         config=types.GenerateContentConfig(response_mime_type="application/json")).text)'),
        "3 · Source-grounding requests": ("Sandbox the knowledge: inject trusted context (RAG) and forbid outside knowledge.",
                                          'prompt = f"""Answer based *exclusively* on the DOCUMENT CONTEXT. Do not use outside knowledge.\n'
                                          'If the answer is not in the context, say: \'The answer is not available in the provided documentation.\'\n'
                                          'DOCUMENT CONTEXT:\n---\n{relevant_doc_chunk}\n---\nUSER QUESTION: {user_question}"""'),
        "4 · Error-handling instructions": ("The prompt's own try/except: define a safe, machine-readable fallback for ambiguity or "
                                            "out-of-scope input (“I'm here to help with school topics!…”), plus code-level exception handling.",
                                            'try:\n    response = chat.send_message(user_input)\n    print(f"Study Buddy: {response.text}")\n'
                                            'except Exception as e:\n    print("Study Buddy: Oops! Something went wrong on my end. Please try asking in a different way.")'),
    }
    for title, (why, code) in strategies.items():
        with st.container(border=True):
            st.markdown(f"**{title}** — {why}")
            st.code(code, language="python")
    st.markdown("#### 🟢 Try source grounding")
    doc = st.text_area("Document context", "Refund policy: Items can be returned within 15 days of delivery. Refunds are issued to the "
                       "original payment method within 5-7 business days. Gift cards are non-refundable.", key="m9_doc")
    uq = st.text_input("Question", "Can I get a refund on a gift card? And what about shipping fees?", key="m9_uq")
    gp = (f"Answer based exclusively on the DOCUMENT CONTEXT. If any part of the answer is not in the context, say "
          f"'The answer is not available in the provided documentation.' for that part.\nDOCUMENT CONTEXT:\n---\n{doc}\n---\nUSER QUESTION: {uq}")
    res = run_live("Ask grounded", lambda: generate(gp, temperature=0)[0], key="m9_ground")
    if res:
        with st.container(border=True):
            st.markdown(res)
        st.caption("Shipping fees aren't in the document — a grounded answer should say so instead of guessing.")


# ---------------------------------------------------------------------------
with tabs[9]:
    takeaway([
        "Evaluating output is as important as engineering input.",
        "Hallucinations come from data gaps, next-token objectives, overconfidence, no grounding and vague prompts.",
        "Stress-test quirks: contrastive phrasing, numeric stress, multi-turn memory.",
        "Layer metrics: automated checks → human judgment → weighted decisions; use fair rubrics with anchors.",
        "LLM-as-judge scales evaluation; iterate prompts with A/B tests, schemas, grounding and failsafes.",
    ])


with tabs[10]:
    quiz("m9", [
        {"q": "A model invents a journal article with authors and DOI. This is…", "options": ["Mode collapse", "Citation hallucination",
         "Verbosity drift", "Order effect"], "answer": 1, "why": "Fake references that look real."},
        {"q": "A fraud model flags 10 transactions, 4 are truly fraud. Which metric is 40%?", "options": ["Recall", "Precision", "Accuracy", "F1"],
         "answer": 1, "why": "Precision = TP / everything flagged."},
        {"q": "Why should an LLM judge evaluate each answer in a fresh call?", "options": ["It's cheaper", "So earlier verdicts in chat history can't bias the score",
         "JSON parsing needs it", "Temperature must be 1"], "answer": 1, "why": "Independent judgments are more consistent."},
        {"q": "“Answer only from the provided document” is which strategy?", "options": ["Explicit task framing", "Output schema",
         "Source grounding", "Mode collapse"], "answer": 2, "why": "It constrains the knowledge base to trusted data."},
        {"q": "First step of the iterative refinement cycle?", "options": ["Edit the prompt", "Define target behaviour", "A/B test", "Fine-tune the model"],
         "answer": 1, "why": "You can't improve what you haven't defined."},
    ])
