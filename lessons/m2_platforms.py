import zlib

import pandas as pd
import streamlit as st

from core import sim
from core.llm import generate, get_client, get_model, run_live
from core.ui import analogy, cards, compare, definition, page_header, quiz, stepper, takeaway

page_header(
    "m2",
    "How a Large Language Model is born, which platforms exist, and how to use them responsibly.",
    [
        "Explain the three training stages: pre-training, supervised fine-tuning and RLHF",
        "Compare open-source and closed-source LLMs",
        "Explain what 'billions of parameters' means and why Small Language Models matter",
        "Test LLMs with good vs bad prompts",
        "Identify ethical risks: bias, privacy, IP, misinformation, deepfakes",
    ],
)

tabs = st.tabs(["What is an LLM?", "How LLMs are built", "🧮 Tokenization", "🏆 RLHF ranking game",
                "Open vs closed", "Parameters & SLMs", "Good vs bad prompts", "Ethics"])

with tabs[0]:
    analogy("Your phone's predictive keyboard guesses your next word using only your messages. Now imagine a "
            "system that has read nearly every book, article, Wikipedia page and public web page ever published. "
            "That's a Large Language Model.")
    definition("Large Language Model (LLM)", "a deep-learning model (usually a Transformer) trained on massive text to "
                                             "predict the next token — which lets it complete sentences, follow context "
                                             "across long conversations, mimic tone and intent, and answer coherently.")
    cards([
        ("Predicts", "whole sentences and paragraphs, not just one word"),
        ("Understands context", "across long conversations"),
        ("Mimics", "human tone, emotion and intent"),
        ("Generates", "coherent, creative responses"),
    ], cols=4)

with tabs[1]:
    st.subheader("From raw text to a helpful assistant — like a student becoming an expert")
    stepper("llm_build", [
        ("Stage 1 · Generative pre-training — collect data",
         "**Challenge: the model doesn't know language.** Researchers gather *hundreds of billions of words* "
         "where humans express themselves: Wikipedia, books, news, Reddit/Stack Overflow/Quora, product reviews, "
         "research papers."),
        ("Stage 1 · Clean the data",
         "**Challenge: raw internet data is a mess** — ads, spam, repeated headers, broken characters. "
         "Pipelines remove duplicates, filter spam & boilerplate, strip unsafe content, normalise formatting."),
        ("Stage 1 · Tokenize",
         "**Challenge: machines don't understand words.** Text is split into **tokens** (`battery`, `bat`+`tery`, "
         "punctuation) and mapped to numeric IDs. *(Try the Tokenization tab.)*"),
        ("Stage 1 · Learn patterns → Base model",
         "The model learns **patterns, not facts** — “I'm feeling anxious” is often followed by “can't sleep”. "
         "Result: a **base GPT model** that completes sentences fluently but **doesn't follow instructions**, isn't "
         "polite and doesn't know true from false."),
        ("Stage 2 · Supervised fine-tuning (SFT)",
         "Feed **labeled prompt → ideal response** pairs written by humans, e.g. "
         "*“Translate to Hindi: Climate change” → “जलवायु परिवर्तन”*. The model answers, the answer is compared with "
         "the correct one (**loss**), and an **optimizer** updates the weights — millions of times. "
         "Now it can answer questions, format output and stay on topic. It still lacks *judgment*."),
        ("Stage 3 · RLHF — collect human preferences",
         "The SFT model is like *a helpful intern with no common sense*: correct but blunt, overconfident, sometimes unsafe. "
         "It generates several answers per prompt; **humans rank them** best → worst."),
        ("Stage 3 · RLHF — Reward model + reinforcement learning",
         "Rankings train a **reward model** that *scores* text by how much humans would like it (it doesn't generate). "
         "The LLM generates → reward model scores → high score = reward, low = penalty → update (e.g. with **PPO**). "
         "Repeated thousands of times, the model learns to answer the way humans prefer: helpful, polite, safe."),
    ])
    st.graphviz_chart(
        """
digraph { rankdir=LR; node [shape=box style="rounded,filled" fontname="Helvetica"];
 A [label="Internet-scale text" fillcolor="#f1f5f9"]; B [label="Base model\\n(next-token predictor)" fillcolor="#dbeafe"];
 C [label="SFT model\\n(follows instructions)" fillcolor="#bfdbfe"]; D [label="Aligned assistant\\n(helpful · polite · safe)" fillcolor="#86efac"];
 A -> B [label=" pre-training"]; B -> C [label=" labeled pairs"]; C -> D [label=" RLHF (human rankings)"]; }
"""
    )

with tabs[2]:
    st.subheader("Machines read numbers, not words")
    st.caption("This is an **illustrative** sub-word tokenizer. Real tokenizers (BPE, SentencePiece) learn their splits from data, but the idea is the same.")
    text = st.text_area("Type anything", "Unbelievably, the battery life of iPhone 14 lasts a day!", key="m2_tok")
    toks = sim.toy_tokenize(text)
    colors = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fecaca", "#ddd6fe", "#fbcfe8"]
    html = " ".join(
        f"<span style='background:{colors[i % len(colors)]};color:#111;padding:2px 4px;border-radius:4px;"
        f"margin:2px;display:inline-block;font-family:monospace'>{t.replace(' ', '·')}</span>"
        for i, t in enumerate(toks)
    )
    st.markdown(html, unsafe_allow_html=True)
    ids = [zlib.crc32(t.encode()) % 50000 for t in toks]  # stable fake vocabulary ids
    st.caption(f"{len(toks)} tokens → IDs like {ids[:8]}{' …' if len(ids) > 8 else ''}  ·  `·` marks a leading space")
    if get_client() is not None and st.button("Count with Gemini's real tokenizer"):
        try:
            n = get_client().models.count_tokens(model=get_model(), contents=text).total_tokens
            st.success(f"Gemini counts **{n} tokens** for this text (toy tokenizer said {len(toks)}).")
        except Exception as e:  # noqa: BLE001
            st.error(str(e)[:300])

with tabs[3]:
    st.subheader("Be the human in RLHF")
    st.markdown("Prompt: **“Explain depression to someone going through it.”** The model produced four answers. "
                "All are technically correct — rank them as a human reviewer would.")
    answers = {
        "A": "Depression is a mental disorder classified by the DSM-5…",
        "B": "You're not alone. Many people struggle with this. It's okay to seek help.",
        "C": "It's when you feel sad for a long time.",
        "D": "Try meditation. It works for most people.",
    }
    for k, v in answers.items():
        st.markdown(f"**{k}.** {v}")
    ranking = st.multiselect("Click answers from **best → worst**", list(answers), key="m2_rank", max_selections=4)
    if len(ranking) == 4:
        reward = {a: 1.0 - i / 3 for i, a in enumerate(ranking)}
        st.bar_chart(pd.Series(reward, name="reward score").sort_values(ascending=False), height=220)
        st.markdown(
            f"Your ranking becomes training data for the **reward model**. It learns that answers like **{ranking[0]}** "
            "deserve high scores. During RL, the LLM is rewarded for producing answers that score like that. "
            "Most reviewers pick **B** — supportive and human — which is exactly the judgment SFT alone can't teach."
        )

with tabs[4]:
    compare("🔓 Open-source LLMs",
            "Code/weights are public: anyone can **use, modify, fine-tune, run offline**.\n\n"
            "*Analogy:* an electric-car engine whose blueprints are shared — anyone can build or improve it.\n\n"
            "**Gemma** (Google, 2B/7B) · **LLaMA 3** (Meta, 8B/70B) · **Qwen** (Alibaba, multilingual) · "
            "**BLOOM** (BigScience, 40+ languages) · **Falcon** (TII UAE, 40B+, 1T tokens)",
            "🔒 Closed-source LLMs",
            "Privately owned; you use them **through an app or API** — the inner workings are hidden.\n\n"
            "*Analogy:* a supercar you can only rent through the maker's app — you can't open the hood.\n\n"
            "**GPT-4** (OpenAI, ChatGPT) · **Claude** (Anthropic) · **Gemini** (Google DeepMind, multimodal)",
            bad_good=False)
    st.markdown("#### Comparison metrics")
    st.dataframe(pd.DataFrame({
        "Metric": ["Performance", "Accessibility", "Customization", "Community", "Infrastructure"],
        "Closed-source": ["Usually strongest out of the box, multimodal", "Internet + API key, rate limits, mostly paid",
                          "Prompting / limited API fine-tuning", "Great docs, but can't fix the model",
                          "Provider's cloud (low latency, high availability)"],
        "Open-source": ["Catching up fast; may need fine-tuning", "Free to download, runs offline",
                        "Retrain, fine-tune, change architecture", "Huge GitHub / Hugging Face / Discord communities",
                        "Your machine or your own paid cloud"],
    }), hide_index=True, width="stretch")
    analogy("Student A builds a career-counsellor bot on ChatGPT: polished, but can't change its behaviour, can't work "
            "offline, pays per call. Student B uses LLaMA 3: more work, but hosts it locally and trains it on local "
            "career data — it's *his* chatbot.", title="Two students")
    definition("Multimodal", "one model that understands and generates across text, images, audio, video and files — "
                             "e.g. explain a textbook diagram + summarise PDF notes + answer a spoken question at once.")

with tabs[5]:
    definition("Parameter", "a learned weight inside the network — a 'memory slot' adjusted during training by "
                            "backpropagation and gradient descent. 13B parameters = 13 billion tuned numbers.")
    sizes = pd.DataFrame({"model": ["Phi-2", "Gemma", "LLaMA 3 (small)", "Falcon", "Qwen", "GPT-3"],
                          "billions of parameters": [1.3, 2, 8, 40, 72, 175]}).set_index("model")
    st.bar_chart(sizes, height=260, horizontal=True)
    st.caption("More parameters → more capacity and complexity, but also more hardware needed.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🚗 Small Language Models")
        analogy("An LLM is a 100-seat luxury bus; an SLM is a compact car — fewer seats, but faster and cheaper "
                "for most trips.", title="Bus vs car")
        st.markdown("- **Lightweight** — laptops, phones, edge devices\n- **Fast** — real-time responses\n"
                    "- **Private** — data stays on-device\n- **Cheap** — training < $1,000, no API fees\n"
                    "- Examples: **Phi** (Microsoft, 1.3B), **Gemma** (2B/7B)")
    with c2:
        st.markdown("### Limitations of SLMs")
        st.markdown("1. **Limited knowledge** — vague on niche/new topics\n"
                    "2. **Weaker multi-step reasoning** — e.g. the bananas puzzle below\n"
                    "3. **Reduced multimodality** — often need external image encoders\n"
                    "4. **More hallucination** — may invent a citation\n"
                    "5. **Poor generalisation** — an HR-tuned model struggles with legal questions")
        st.code("John bought three apples and two bananas, then gave one banana\n"
                "to his friend. How many bananas does he have left?   → 1", language="text")

with tabs[6]:
    st.subheader("🟢 Hands-on: good vs bad prompts")
    st.caption("The course asks you to try both kinds of prompt on several models. Here you can run them side by side.")
    c1, c2 = st.columns(2)
    bad = c1.text_area("Bad (vague) prompt", "tell me about climate change", key="m2_bad")
    good = c2.text_area("Good (specific) prompt",
                        "Act as an environmental journalist. In 120 words, explain why global temperatures are rising, "
                        "with 2 real examples, for a high-school audience.", key="m2_good")
    res = run_live("Run both prompts", lambda: (generate(bad)[0], generate(good)[0]), key="m2_live")
    if res:
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.markdown(res[0])
        with c2.container(border=True):
            st.markdown(res[1])
    st.markdown("**Try on other platforms:** [Hugging Face Chat](https://huggingface.co/chat/) · "
                "[ChatGPT](https://chatgpt.com/) · [Gemini](https://gemini.google.com/app) · "
                "[AI Studio](https://aistudio.google.com/) · [Claude](https://claude.ai/new)")

with tabs[7]:
    st.subheader("Ethical considerations")
    issues = [
        ("⚖️ Bias & fairness", "A career bot trained on historical data suggests engineering to boys and nursing to girls.",
         "2018: Amazon scrapped a hiring tool that downgraded résumés containing “women's”.",
         "Balanced datasets · audit outputs · humans in high-stakes decisions"),
        ("🔐 Privacy", "Private notes uploaded to an AI tool may end up in training data.",
         "2023: Samsung engineers pasted source code into ChatGPT; Samsung banned internal use.",
         "Don't paste sensitive data · use tools that delete/anonymise · know laws like GDPR"),
        ("©️ Intellectual property", "Who owns an AI image of a famous character doing yoga?",
         "2023: Getty Images sued Stability AI over training on copyrighted images.",
         "Use licensed/public-domain data · label AI content · give credit"),
        ("❓ Misinformation", "AI says “Elon Musk invented the internet” — confident and wrong (hallucination).",
         "2023: a New York lawyer submitted ChatGPT-invented court cases and was sanctioned.",
         "Fact-check · combine with trusted sources · train on verified data"),
        ("🎭 Deepfakes & misuse", "Fake but realistic video/voice used to deceive or blackmail.",
         "A deepfake of President Zelenskyy urging troops to surrender circulated online.",
         "Watermark/label · deepfake detection · public awareness"),
    ]
    for title, scenario, real, fix in issues:
        with st.expander(title):
            st.markdown(f"**Scenario:** {scenario}\n\n**Real world:** {real}\n\n**How to handle:** {fix}")
    st.markdown("**Build AI ethically:** Transparency · Inclusivity · Accountability · Consent · Continuous feedback "
                "(e.g. Google's 2018 AI Principles).")

takeaway([
    "LLMs are built in 3 stages: pre-training (language patterns) → SFT (follow instructions) → RLHF (human preferences).",
    "Text is tokenized into numeric IDs; the model learns patterns, not facts.",
    "Closed models = top performance via API; open models = control, privacy, offline use, customization.",
    "Parameters are learned weights; SLMs trade some capability for speed, cost and privacy.",
    "Responsible AI addresses bias, privacy, IP, hallucinations and deepfakes.",
])

quiz("m2", [
    {"q": "What does the base model (after pre-training) NOT do well?",
     "options": ["Complete sentences", "Generate fluent text", "Follow instructions politely", "Use grammar"],
     "answer": 2, "why": "Instruction-following comes from supervised fine-tuning; alignment from RLHF."},
    {"q": "In RLHF, what does the reward model do?",
     "options": ["Generates answers", "Scores answers by predicted human preference", "Tokenizes text", "Collects web data"],
     "answer": 1, "why": "It is a judge trained on human rankings — it doesn't generate text."},
    {"q": "A hospital needs a chatbot whose data never leaves its servers. Best fit?",
     "options": ["A closed model via public API", "An open-source / small model hosted locally", "Any model — privacy is the same", "A GAN"],
     "answer": 1, "why": "Open/small models can run on-premise, keeping data private."},
    {"q": "“7B” in a model name means…",
     "options": ["7 billion training documents", "7 billion learned parameters", "7 billion users", "Version 7 beta"],
     "answer": 1, "why": "It's the number of learned weights."},
])
