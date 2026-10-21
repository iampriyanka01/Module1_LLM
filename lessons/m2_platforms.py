import zlib

import pandas as pd
import streamlit as st

from core import sim
from core.llm import generate, get_client, get_model, run_live
from core.ui import analogy, cards, compare, definition, diagram, page_header, quiz, takeaway
from core.diagrams import show_dg


# ---------------------------------------------------------------------------
# Diagrams for "How LLMs are built": one overview + one detail picture per step
# ---------------------------------------------------------------------------
_NODE = 'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=14 margin="0.2,0.12" color="#475569" penwidth=1.4];'
_EDGE = 'edge [color="#475569" fontname="Helvetica" fontsize=12 penwidth=1.4];'
_LOOP = 'color="#dc2626" fontcolor="#dc2626" style=dashed constraint=false'


def build_overview_html(stage: int) -> str:
    """Pipeline as a row of chips; the active stage (1-3) is orange."""
    def chip(text, kind, active=False):
        bg = "#fb923c" if active else {"proc": "#e2e8f0", "data": "#f1f5f9", "base": "#dbeafe", "sft": "#bfdbfe", "done": "#86efac"}[kind]
        bd = "3px solid #c2410c" if active else "1px solid #64748b"
        return (f"<div style='background:{bg};color:#111;border:{bd};border-radius:10px;padding:6px 7px;"
                f"text-align:center;font-size:0.8rem;line-height:1.2'>{text}</div>")
    arrow = "<div style='font-size:1.1rem;color:#64748b'>→</div>"
    items = [chip("Internet-scale<br>text", "data"), chip("<b>Stage 1</b><br>Pre-training", "proc", stage == 1),
             chip("Base model<br><small>completes text</small>", "base"), chip("<b>Stage 2</b><br>Supervised<br>fine-tuning", "proc", stage == 2),
             chip("SFT model<br><small>follows instructions</small>", "sft"), chip("<b>Stage 3</b><br>RLHF", "proc", stage == 3),
             chip("Aligned assistant<br><small>helpful · polite · safe</small>", "done")]
    row = arrow.join(items)
    return f"<div style='display:flex;flex-wrap:wrap;align-items:center;gap:4px;margin:6px 0 14px 0'>{row}</div>"


BUILD_DETAIL = {
    0: ("digraph { pad=0.4; rankdir=LR; " + _NODE + " " + _EDGE + """
      W [label="Wikipedia" fillcolor="#f1f5f9"]; K [label="Books" fillcolor="#f1f5f9"]; N [label="News sites" fillcolor="#f1f5f9"];
      R [label="Reddit · Stack Overflow\\nQuora" fillcolor="#f1f5f9"]; V [label="Product reviews" fillcolor="#f1f5f9"]; P [label="Research papers" fillcolor="#f1f5f9"];
      C [label="Raw corpus\\nhundreds of billions of words" shape=cylinder fillcolor="#fed7aa"];
      W -> C; K -> C; N -> C; R -> C; V -> C; P -> C; }"""),
    1: ("digraph { pad=0.4; rankdir=TB; " + _NODE + " " + _EDGE + """
      C [label="Raw corpus\\n(ads, spam, junk)" shape=cylinder fillcolor="#fecaca"];
      a [label="Remove\\nduplicates"]; b [label="Filter spam &\\nboilerplate"]; c [label="Strip unsafe /\\nirrelevant content"]; d [label="Normalise\\nformatting"];
      O [label="Clean corpus" shape=cylinder fillcolor="#bbf7d0"];
      C -> a -> b -> c -> d -> O; }"""),
    2: ("digraph { pad=0.4; rankdir=TB; " + _NODE + " " + _EDGE + """
      T [label="Text:\\n'battery life lasts a day'" fillcolor="#f1f5f9"];
      K [label="Tokens:\\n[bat] [tery] [ life] [ lasts] [ a] [ day]" fillcolor="#fde68a"];
      I [label="Numeric IDs:\\n[4521] [9833] [1207] [8812] [64] [3150]" fillcolor="#bfdbfe"];
      M [label="Model input" shape=ellipse fillcolor="#dbeafe"];
      T -> K [label=" split into tokens"]; K -> I [label=" look up IDs"]; I -> M; }"""),
    3: ("digraph { pad=0.4; rankdir=TB; " + _NODE + " " + _EDGE + """
      I [label="Token IDs of a sentence\\n'The battery life of iPhone 14 ...'" fillcolor="#f1f5f9"];
      M [label="Transformer\\npredicts the next token" fillcolor="#dbeafe"];
      G [label="Guess:\\n'is decent'" fillcolor="#fef9c3"]; R [label="Real next words:\\n'lasts a day'" fillcolor="#fef9c3"];
      L [label="Loss\\n(how wrong?)" fillcolor="#fecaca"]; U [label="Update weights" fillcolor="#fecaca"];
      B [label="Base GPT model" shape=ellipse fillcolor="#86efac"];
      I -> M -> G -> L; R -> L; L -> U; U -> M [label=" repeat billions of times" """ + _LOOP + """];
      U -> B [style=dotted label=" when trained"]; }"""),
    4: ("digraph { pad=0.4; rankdir=TB; " + _NODE + " " + _EDGE + """
      D [label="SFT dataset\\nprompt -> ideal response\\n('Translate to Hindi: Climate change'\\n-> 'जलवायु परिवर्तन')" shape=note fillcolor="#fef9c3"];
      B [label="Base model" shape=ellipse fillcolor="#dbeafe"];
      O [label="Model's response" fillcolor="#f1f5f9"]; C [label="Compare with the\\ncorrect response\\n= loss" fillcolor="#fecaca"];
      Z [label="Optimizer\\nadjusts weights" fillcolor="#fecaca"];
      S [label="SFT model\\nfollows instructions" shape=ellipse fillcolor="#86efac"];
      D -> B [label=" prompt"]; B -> O -> C; D -> C [style=dotted label=" correct answer"]; C -> Z; Z -> B [label=" millions of times" """ + _LOOP + """];
      Z -> S [style=dotted label=" when trained"]; }"""),
    5: ("digraph { pad=0.4; rankdir=TB; " + _NODE + " " + _EDGE + """
      P [label="Prompt:\\n'Explain depression to\\nsomeone going through it'" fillcolor="#f1f5f9"];
      M [label="SFT model" shape=ellipse fillcolor="#bfdbfe"];
      A [label="Answer A (clinical)\\nAnswer B (supportive)\\nAnswer C (short)\\nAnswer D (advice)" shape=note fillcolor="#fef9c3"];
      H [label="Human reviewers\\nrank best to worst" shape=box fillcolor="#fed7aa"];
      R [label="Ranked list:\\nB > C > A > D" shape=note fillcolor="#bbf7d0"];
      P -> M -> A -> H -> R; }"""),
    6: ("digraph { pad=0.4; rankdir=TB; " + _NODE + " " + _EDGE + """
      R [label="Rankings" shape=note fillcolor="#fef9c3"]; W [label="Reward model\\n(a judge: scores text,\\ndoes not write it)" fillcolor="#fed7aa"];
      P [label="Prompt" fillcolor="#f1f5f9"]; M [label="LLM" shape=ellipse fillcolor="#bfdbfe"];
      O [label="Response" fillcolor="#f1f5f9"]; S [label="Score" fillcolor="#fef9c3"];
      U [label="High score = reward\\nLow score = penalty\\n-> update (PPO)" fillcolor="#fecaca"];
      OUT [label="Aligned assistant" shape=ellipse style=filled fillcolor="#86efac"];
      R -> W [label=" trains"]; P -> M -> O -> W -> S -> U; U -> M [label=" thousands of times" """ + _LOOP + """];
      U -> OUT [style=dotted label=" when trained"]; }"""),
}
BUILD_STAGE = [1, 1, 1, 1, 2, 3, 3]


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
                "Open vs closed", "Parameters & SLMs", "Good vs bad prompts", "Ethics", "🧾 Key takeaways", "✅ Knowledge check"])

with tabs[0]:
    analogy("Your phone's predictive keyboard guesses your next word using only your messages. Now imagine a "
            "system that has read nearly every book, article, Wikipedia page and public web page ever published. "
            "That's a Large Language Model.")
    show_dg("m2:0")
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
    BUILD_STEPS = [
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
         "Now it can answer questions, format answers (bullets, summaries), sound polite or concise and stay on topic. "
         "Unlike Stage 1 (no labels), this is **supervised** learning: the correct answer is known. It still lacks *judgment*: "
         "it can sound robotic, miss nuance or hallucinate."),
        ("Stage 3 · RLHF — collect human preferences",
         "The SFT model is like *a helpful intern with no common sense*: correct but blunt, overconfident, sometimes unsafe. "
         "It generates several answers per prompt; **humans rank them** best → worst."),
        ("Stage 3 · RLHF — Reward model + reinforcement learning",
         "Rankings train a **reward model** that *scores* text by how much humans would like it (it doesn't generate). "
         "The LLM generates → reward model scores → high score = reward, low = penalty → update (e.g. with **PPO**). "
         "Repeated thousands of times, the model learns to answer the way humans prefer: helpful, polite, safe."),
    ]
    if "m2_build_step" not in st.session_state:
        st.session_state["m2_build_step"] = 0
    i = st.session_state["m2_build_step"]
    st.progress((i + 1) / len(BUILD_STEPS), text=f"Step {i + 1} of {len(BUILD_STEPS)}")
    title, body = BUILD_STEPS[i]
    with st.container(border=True):
        st.markdown(f"### {i + 1}. {title}")
        st.markdown(body)
    st.markdown("**The whole pipeline** (orange = the stage you are reading about)")
    st.markdown(build_overview_html(BUILD_STAGE[i]), unsafe_allow_html=True)
    st.markdown("**What happens in this step**")
    _l, mid, _r = st.columns([1, 4, 1])
    with mid:
        diagram(BUILD_DETAIL[i], max_w=6.5, max_h=6.2)
    st.caption("Red dashed arrows are the learning loop: the model is corrected and tries again.")
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("◀ Back", key="m2_build_back", disabled=i == 0, width="stretch"):
        st.session_state["m2_build_step"] -= 1
        st.rerun()
    if c2.button("Next ▶", key="m2_build_next", disabled=i == len(BUILD_STEPS) - 1, width="stretch"):
        st.session_state["m2_build_step"] += 1
        st.rerun()
    with st.expander("After each stage, what can the model do?"):
        st.dataframe(pd.DataFrame({
            "Stage": ["1 · Pre-training", "2 · Supervised fine-tuning", "3 · RLHF"],
            "Learns from": ["Hundreds of billions of words, no labels", "Human-written prompt → response pairs (supervised)", "Human rankings of its own answers"],
            "Result": ["Base model: completes text, but ignores instructions, isn't polite, doesn't know true from false",
                       "Follows instructions, formats answers, stays on topic; still blunt and lacks judgment",
                       "Helpful, polite and safer; answers the way humans prefer"],
        }), hide_index=True, width="stretch")

with tabs[2]:
    st.subheader("Machines read numbers, not words")
    show_dg("m2:2")
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
    show_dg("m2:3")
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
    show_dg("m2:4")
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
    with st.expander("Meet the models (with a typical use)"):
        st.dataframe(pd.DataFrame({
            "Model": ["Gemma (Google)", "LLaMA 3 (Meta)", "Qwen (Alibaba)", "BLOOM (BigScience)", "Falcon (TII, UAE)",
                      "GPT-4 (OpenAI)", "Claude (Anthropic)", "Gemini (Google DeepMind)"],
            "Type": ["Open", "Open", "Open", "Open", "Open", "Closed", "Closed", "Closed"],
            "Known for": ["2B / 7B sizes, same research as Gemini, runs on local machines", "8B and 70B sizes, competes with closed models",
                          "Multilingual (English, Chinese and more), enterprise-grade", "1000+ researchers, 40+ languages, AI democratisation",
                          "40B+ parameters, trained on 1 trillion tokens", "Powers ChatGPT; used through the app or API",
                          "Known for being ethical and safe; used in business tools", "Multimodal; integrated with Google Search and Workspace"],
            "Example use": ["Student's offline study assistant", "AI teacher for rural schools, tuned on local textbooks",
                            "English + Mandarin customer-support bot", "News summaries in Swahili and Bengali",
                            "Private legal assistant for contract review", "Writing, coding, brainstorming", "Productivity and business apps",
                            "Explain a diagram + summarise PDF notes from a voice question"],
        }), hide_index=True, width="stretch")
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
    show_dg("m2:5")
    st.markdown("**How parameters are formed:** a Transformer has layers, and the layers contain neurons. Each neuron has **weights and biases**: those are the parameters. "
                "During training, backpropagation and gradient descent adjust them based on the model's error on each example.")
    st.markdown("**What the parameter count tells you:** *capacity* (how much it can learn) · *complexity* (how deep and wide it is) · "
                "*resource requirement* (how much hardware it needs).")
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
        with st.expander("Each challenge, with an example"):
            st.markdown(
                "- **Limited knowledge scope:** ask Phi-2 or Gemma about the latest quantum-computing breakthrough and the answer may be vague or outdated. *Matters for medical, legal and policy use.*\n"
                "- **Complex reasoning:** the bananas puzzle can be answered wrongly or without the steps. *Matters for finance and planning tools.*\n"
                "- **Reduced multimodality:** GPT-4o or Gemini take an image directly; small models like LLaVA often need an external image encoder, which is slower and harder to integrate.\n"
                "- **Security and hallucination:** asked for a research-paper citation, a small model may invent a title, authors and journal. *Matters for academia, journalism, healthcare.*\n"
                "- **Poor generalisation:** a model fine-tuned on HR questions does badly on legal compliance or sales strategy.")
        st.code("John bought three apples and two bananas, then gave one banana\n"
                "to his friend. How many bananas does he have left?   → 1", language="text")

with tabs[6]:
    st.subheader("🟢 Hands-on: good vs bad prompts")
    show_dg("m2:6")
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
    show_dg("m2:7")
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


# ---------------------------------------------------------------------------
with tabs[8]:
    takeaway([
        "LLMs are built in 3 stages: pre-training (language patterns) → SFT (follow instructions) → RLHF (human preferences).",
        "Text is tokenized into numeric IDs; the model learns patterns, not facts.",
        "Closed models = top performance via API; open models = control, privacy, offline use, customization.",
        "Parameters are learned weights; SLMs trade some capability for speed, cost and privacy.",
        "Responsible AI addresses bias, privacy, IP, hallucinations and deepfakes.",
    ])
    with st.expander("Conclusion from the course notes"):
        st.markdown(
            "1. **Training pipeline:** generative pre-training on massive text with the Transformer → supervised fine-tuning with labeled prompt-response pairs → RLHF to align behaviour with human preferences.\n"
            "2. **Open vs closed:** open models (LLaMA, Gemma, Qwen) give transparency, customisation and offline use; closed models (ChatGPT, Gemini, Claude) give high performance but less flexibility.\n"
            "3. **Small Language Models:** Phi and Gemma (2B / 7B) are lightweight alternatives for mobile, offline and data-sensitive uses.\n"
            "4. **Ethics:** bias, privacy breaches, misinformation and misuse (deepfakes). Practise transparency, fairness and continuous human oversight.\n"
            "5. **Hands-on:** try open models (Hugging Face) and closed tools (ChatGPT, Gemini, Claude) with good and bad prompts and compare.")
    st.markdown("**Resources:** open models: [Hugging Face](https://huggingface.co/) · [HuggingChat](https://huggingface.co/chat/). "
                "Closed models: [ChatGPT](https://chatgpt.com/) · [Gemini](https://gemini.google.com/app) · [AI Studio](https://aistudio.google.com/prompts/new_chat) · [Claude](https://claude.ai/new). "
                "Trends: [AI news](https://www.artificialintelligence-news.com) · [Artificial Analysis](https://artificialanalysis.ai/)")


with tabs[9]:
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
