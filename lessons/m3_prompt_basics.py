import pandas as pd
import streamlit as st

from core.llm import generate, run_live
from core.ui import analogy, compare, definition, page_header, quiz, takeaway

page_header(
    "m3",
    "If LLMs are powerful engines, prompts are the controls. Learn to drive them like a pro.",
    [
        "Explain what a prompt is and why wording changes results",
        "Build prompts from Role, Instruction, Input and Output format",
        "Avoid the 'loss in the middle' problem",
        "Apply clarity, specificity, tone, constraints and intent framing",
        "Use zero-shot, one-shot, few-shot and role-based prompting",
        "Generate prompts dynamically from templates",
    ],
)

tabs = st.tabs(["What is a prompt?", "🧱 Prompt builder", "📉 Loss in the middle", "🎨 Crafting",
                "🧪 Technique lab", "⚙️ Dynamic prompting", "🧾 Key takeaways", "✅ Knowledge check"])

with tabs[0]:
    definition("Prompt", "the instruction or input you give a language model to get a response.")
    definition("Prompt engineering", "the craft of writing smart, structured, context-aware instructions that guide the "
                                     "model to useful, high-quality output — knowing *how* to ask, *what* to include.")
    analogy("Tell a waiter “Get me food” and you get something random. Say “A spicy paneer tikka starter, served hot, "
            "and a lemon soda — not too salty” and you get exactly what you imagined.", title="Restaurant")
    compare("Same model, vague prompt", "“Tell me something about climate change.”\n\n→ generic, surface-level answer",
            "Same model, engineered prompt",
            "“Act as an environmental journalist. Write a 200-word report explaining the recent rise in global "
            "temperatures, with statistics and real examples.”\n\n→ focused, detailed, high-quality")
    st.caption("You're not changing the model — only the way you talk to it.")

with tabs[1]:
    st.subheader("The four building blocks of a great prompt")
    st.markdown("Fill each block and watch the prompt assemble. Toggle blocks off to see what the model loses.")
    presets = {
        "Career coach (from the notes)": ("You are a professional career coach with 10+ years of experience.",
                                          "Review the following resume and suggest improvements that would make it more appealing for product management roles.",
                                          "Resume: B.Tech CSE 2023. 1.5 years as software engineer at a fintech startup. Built payment dashboards in React. Led a 3-person hackathon team that won 1st prize. Skills: Python, SQL, React, Figma.",
                                          "Provide your feedback in bullet points, under 150 words, using a friendly, constructive tone."),
        "Travel caption writer": ("You are a travel blogger who writes short Instagram captions.",
                                  "Write a caption for my photo.",
                                  "Photo: sunrise over the backwaters of Alleppey, a houseboat in the foreground.",
                                  "One line, under 15 words, 2 emojis, no hashtags."),
        "Strict grammar teacher": ("You are a strict grammar teacher correcting essays.",
                                   "Correct the grammar and explain each fix.",
                                   "Text: Me and him goes to the market yesterday and buyed many vegetable.",
                                   "Return a table with columns: Original | Corrected | Rule."),
    }
    preset = st.selectbox("Start from an example", list(presets))
    r, i, inp, o = presets[preset]
    c1, c2 = st.columns(2)
    with c1:
        use_r = st.toggle("🎭 Role — who should the AI be?", True, key=f"use_r_{preset}")
        role = st.text_area("Role", r, key=f"role_{preset}", disabled=not use_r, height=70)
        use_i = st.toggle("🎯 Instruction — what should it do?", True, key=f"use_i_{preset}")
        instr = st.text_area("Instruction", i, key=f"instr_{preset}", disabled=not use_i, height=70)
    with c2:
        use_in = st.toggle("📥 Input — what information does it need?", True, key=f"use_in_{preset}")
        inp_t = st.text_area("Input / context", inp, key=f"inp_{preset}", disabled=not use_in, height=70)
        use_o = st.toggle("📐 Output format — how should it look?", True, key=f"use_o_{preset}")
        out = st.text_area("Output format", o, key=f"out_{preset}", disabled=not use_o, height=70)
    parts = [x for x, on in [(role, use_r), (instr, use_i), (inp_t, use_in), (out, use_o)] if on and x.strip()]
    final_prompt = "\n\n".join(parts)
    st.markdown("**Assembled prompt**")
    st.code(final_prompt or "(empty)", language="text")
    missing = [n for n, on in [("Role", use_r), ("Instruction", use_i), ("Input", use_in), ("Output format", use_o)] if not on]
    if missing:
        st.warning(f"Missing: {', '.join(missing)}. Expect " +
                   ("a generic, guessed answer (no input = no context = poor output)." if "Input" in missing else
                    "a less controlled answer."))
    res = run_live("Run this prompt", lambda: generate(final_prompt)[0], key="m3_builder")
    if res:
        with st.container(border=True):
            st.markdown(res)

with tabs[2]:
    st.subheader("Models skim the middle — just like you skim a long WhatsApp message")
    st.markdown("LLMs pay most attention to the **start** (where instructions usually are) and the **end** (recency). "
                "Instructions buried in the middle of a long prompt can get ignored.")
    pos = st.radio("Where is the key instruction placed?", ["Start", "Middle", "End", "Start + reminder at end"], horizontal=True)
    filler = ("The city council met on Tuesday to discuss parking. Residents raised concerns about traffic near the "
              "market. A new bus route was proposed. The weather was pleasant. Several shopkeepers spoke about rent. ") * 6
    instruction = "IMPORTANT: Answer in exactly 3 bullet points and end with the word DONE."
    task = "Summarize the text above."
    half = len(filler) // 2
    layouts = {
        "Start": f"{instruction}\n\n{filler}\n\n{task}",
        "Middle": f"{filler[:half]}\n{instruction}\n{filler[half:]}\n\n{task}",
        "End": f"{filler}\n\n{task} {instruction}",
        "Start + reminder at end": f"{instruction}\n\n{filler}\n\n{task}\n— Reminder: exactly 3 bullet points, end with DONE.",
    }
    # The U-shaped attention curve: strong at the start and end, weak in the middle (illustrative values)
    st.bar_chart(pd.DataFrame({"relative attention (illustrative)": [0.9, 0.3, 0.8]}, index=["1 · beginning", "2 · middle", "3 · end"]),
                 height=200)
    where = {"Start": "1 · beginning", "Middle": "2 · middle", "End": "3 · end", "Start + reminder at end": "1 · beginning and 3 · end"}[pos]
    (st.error if pos == "Middle" else st.success)(f"Your instruction sits in **{where}**.")
    with st.expander("See the full prompt"):
        st.text(layouts[pos])

    def _test():
        txt = generate(layouts[pos], temperature=0.2)[0]
        bullets = sum(1 for line in txt.splitlines() if line.strip()[:1] in "-*•")
        return txt, bullets, txt.strip().rstrip(".").endswith("DONE")

    res = run_live("🟢 Test whether the model followed the instruction", _test, key=f"m3_lim_{pos}")
    if res:
        txt, bullets, done = res
        c1, c2 = st.columns(2)
        c1.metric("Bullet points", bullets, delta="OK" if bullets == 3 else "not 3", delta_color="normal" if bullets == 3 else "inverse")
        c2.metric("Ends with DONE", "Yes" if done else "No")
        st.markdown(txt)
        st.caption("Modern models are better at this, but the risk grows with prompt length. Try the Middle position a few times.")
    st.markdown("#### Four fixes")
    st.markdown("1. **Put critical instructions early** — “Write a short (under 200 words), friendly blog post…”\n"
                "2. **Repeat key points at the end** — “— Reminder: summarize in 3 bullets.”\n"
                "3. **Use clear formatting** — numbered steps, bullets, bold.\n"
                "4. **Break long prompts into smaller calls** — summarize → tweet → why it matters.")

with tabs[3]:
    st.subheader("Prompt crafting — how you say it")
    analogy("“Is it bad if something's wrong with the papers or like if it's fake?” confuses a lawyer. “How do I verify a "
            "land title legally in Delhi, and what red flags should I watch for?” gets a precise answer.", title="Talking to a lawyer")
    crafting = {
        "Clarity & conciseness": ("Tell me something helpful.", "Give me 3 quick tips to stay focused while studying for exams."),
        "Specificity (who/what/why/for whom)": ("Write a message.", "Write a thank-you email after a job interview for a marketing role."),
        "Tone & style": ("Explain machine learning.", "Explain machine learning like I'm 10 years old, in a fun tone."),
        "Depth & constraints": ("Tell me about electric cars.", "List 5 pros and 5 cons of electric cars, each under 20 words."),
        "Intent framing (how to think)": ("Give me startup ideas.", "Help me brainstorm sustainability startup ideas — don't finalize anything yet; include a pro and a con for each."),
    }
    tech = st.radio("Technique", list(crafting), horizontal=True)
    poor, better = crafting[tech]
    c1, c2 = st.columns(2)
    p1 = c1.text_area("❌ Poor prompt", poor, key=f"poor_{tech}")
    p2 = c2.text_area("✅ Better prompt", better, key=f"better_{tech}")
    res = run_live("Compare outputs", lambda: (generate(p1)[0], generate(p2)[0]), key=f"m3_craft_{tech}")
    if res:
        c1, c2 = st.columns(2)
        with c1.container(border=True, height=350):
            st.markdown(res[0])
        with c2.container(border=True, height=350):
            st.markdown(res[1])

with tabs[4]:
    st.subheader("Zero-shot · One-shot · Few-shot · Role-based")
    analogy("Teaching a math problem: just give it (zero-shot), show one solved example (one-shot), show several "
            "(few-shot), or ask them to act as a math examiner (role-based).", title="Teaching a student")
    tasks = {
        "Passive voice": {
            "instruction": "Convert the sentence to passive voice.",
            "examples": [("The chef cooked the meal.", "The meal was cooked by the chef."),
                         ("The dog chased the cat.", "The cat was chased by the dog."),
                         ("Priya wrote the report.", "The report was written by Priya.")],
            "query": "The teacher praised the student.",
            "role": "You are an English grammar teacher.",
        },
        "Sentiment (custom labels)": {
            "instruction": "Classify the customer message as POSITIVE, NEUTRAL or VERY_NEGATIVE. Reply with the label only.",
            "examples": [("Loved the quick delivery!", "POSITIVE"), ("Where is my order? It's been 5 days, this is unacceptable.", "VERY_NEGATIVE"),
                         ("Can I change my delivery address?", "NEUTRAL")],
            "query": "I've been waiting for an order update for 5 days now. This is frustrating.",
            "role": "You are a customer-support triage specialist.",
        },
        "Travel captions (style)": {
            "instruction": "Write an Instagram caption for the place.",
            "examples": [("Rome", "Exploring the streets of Rome — history in every step."), ("Bali", "Sunsets in Bali hit different."),
                         ("Swiss Alps", "Woke up in the Swiss Alps. Coffee, calm, and clouds.")],
            "query": "Ladakh",
            "role": "You are a travel blogger who writes short, punchy captions.",
        },
    }
    tname = st.selectbox("Task", list(tasks))
    t = tasks[tname]

    def build(kind: str) -> str:
        ex = lambda pairs: "\n".join(f"Input: {a}\nOutput: {b}\n" for a, b in pairs)
        if kind == "Zero-shot":
            return f"{t['instruction']}\nInput: {t['query']}\nOutput:"
        if kind == "One-shot":
            return f"{t['instruction']}\n\nExample:\n{ex(t['examples'][:1])}\nNow try this one:\nInput: {t['query']}\nOutput:"
        if kind == "Few-shot":
            return f"{t['instruction']}\n\nExamples:\n{ex(t['examples'])}\nInput: {t['query']}\nOutput:"
        return f"{t['role']} {t['instruction']}\nInput: {t['query']}\nOutput:"

    kinds = ["Zero-shot", "One-shot", "Few-shot", "Role-based"]
    cols = st.columns(4)
    for col, k in zip(cols, kinds):
        with col:
            st.markdown(f"**{k}**")
            st.code(build(k), language="text")
    res = run_live("🟢 Run all four", lambda: {k: generate(build(k), temperature=0.3)[0] for k in kinds}, key=f"m3_tech_{tname}")
    if res:
        cols = st.columns(4)
        for col, k in zip(cols, kinds):
            with col.container(border=True):
                st.caption(k)
                st.markdown(res[k])
        st.caption("Look at format consistency: few-shot outputs usually copy the example style most closely.")

    st.markdown("#### When to use which")
    st.dataframe(pd.DataFrame({
        "Technique": kinds,
        "What you give": ["Task only", "1 input→output example", "2–5 examples", "A persona + task"],
        "Strength": ["Simple, flexible, no data needed", "Shows format & style fast", "Learns patterns, handles subtle tasks",
                     "Adds domain depth & tone"],
        "Limitation": ["Unpredictable on complex/rare tasks", "May inherit bias; weak for complex tasks",
                       "Only as good as the examples", "May overfit the role → rigid replies"],
    }), hide_index=True, width="stretch")
    st.info("Few-shot and role-based combine well: *“You are a French teacher. Examples: Hello → Bonjour … Translate: What time is it?”*")

with tabs[5]:
    st.subheader("Dynamic prompting — prompts that adapt like a good waiter")
    analogy("A waiter suggests the candlelight menu to a celebrating couple, the kids' combo to a family and quick-serve "
            "items to someone in a hurry. Same job, prompt adapted to context.", title="Restaurant")
    template = st.text_input("Template (use {placeholders})",
                             "You are a {role}. Summarize the following {topic} in {style} style for a {audience} audience:\n\n{text}")
    c1, c2, c3, c4 = st.columns(4)
    values = {
        "role": c1.selectbox("role", ["financial analyst", "school teacher", "social media strategist"]),
        "topic": c2.selectbox("topic", ["budget report", "product launch brief", "research abstract"]),
        "style": c3.selectbox("style", ["concise", "bullet-point", "storytelling"]),
        "audience": c4.selectbox("audience", ["non-finance", "Gen Z", "executive"]),
        "text": st.text_area("text", "Q3 revenue grew 15% to ₹12 Cr, driven by subscription renewals. Marketing spend "
                                     "rose 30%. Net margin fell from 18% to 14% due to new hires in sales."),
    }
    try:
        dyn = template.format(**values)
        st.code(dyn, language="text")
    except KeyError as e:
        dyn = None
        st.error(f"Unknown placeholder {e} — available: {', '.join(values)}")
    st.code('prompt = template.format(role=role, topic=topic, style=style, audience=audience, text=user_text)', language="python")
    if dyn:
        res = run_live("Run the generated prompt", lambda: generate(dyn)[0], key="m3_dyn")
        if res:
            with st.container(border=True):
                st.markdown(res)
    st.markdown("**Where it's used:** chatbots (prompt built from the last message) · personalisation (“Hi Rahul, based on "
                "your last booking to Jaipur…”) · chaining (summary → cover letter → interview questions) · app templates.")


# ---------------------------------------------------------------------------
with tabs[6]:
    takeaway([
        "A great prompt = Role + Instruction + Input + Output format.",
        "Put critical instructions first, repeat them at the end, format clearly, split long tasks.",
        "Craft with clarity, specificity, tone, constraints and intent framing.",
        "Zero-shot for simple tasks, few-shot to teach a pattern, role-based for tone and expertise — combine them.",
        "Real apps build prompts dynamically from templates and user context.",
    ])


with tabs[7]:
    quiz("m3", [
        {"q": "Which component tells the model how the answer should look?",
         "options": ["Role", "Instruction", "Input", "Output format"], "answer": 3,
         "why": "Output format defines tone, length, structure (bullets, JSON, table…)."},
        {"q": "A key rule hidden in the middle of a 3-page prompt is ignored. This is…",
         "options": ["Hallucination", "Loss in the middle", "Overfitting", "Mode collapse"], "answer": 1,
         "why": "Models attend most to the start and end of long prompts."},
        {"q": "You show the model 3 labelled examples before the real input. That's…",
         "options": ["Zero-shot", "One-shot", "Few-shot", "Role-based"], "answer": 2, "why": "2–5 examples = few-shot."},
        {"q": "“Summarize this {topic} for a {audience} audience” filled in by code is…",
         "options": ["Dynamic prompting", "RLHF", "Tokenization", "Chain-of-thought"], "answer": 0,
         "why": "Templates + runtime values = dynamic prompts."},
    ])
