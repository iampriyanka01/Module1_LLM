import pandas as pd
import streamlit as st

from core import sim
from core.llm import generate, run_live
from core.ui import analogy, cards, definition, page_header, quiz, stepper, takeaway

page_header(
    "m1",
    "From rule-based systems to machines that create — and the dials that control their creativity.",
    [
        "Explain AI, ML and DL and how they nest inside each other",
        "Tell supervised, unsupervised and reinforcement learning apart",
        "Walk through the ML life cycle and the deep-learning training loop",
        "Describe what makes Generative AI different and how it evolved",
        "Use temperature, top-k and top-p intuitively",
        "Weigh the benefits and risks of GenAI",
    ],
)

tabs = st.tabs(["AI · ML · DL", "Types of ML", "ML life cycle", "Deep learning", "Generative AI",
                "🎛️ Temperature / Top-k / Top-p", "Benefits & risks"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Rewinding the clock: how we got to ChatGPT")
    c1, c2 = st.columns([3, 2])
    with c1:
        definition("Artificial Intelligence (AI)",
                   "the ability of machines to perform tasks that normally need human intelligence — "
                   "learning, problem-solving, decision-making.")
        st.markdown(
            """
**Three eras, one idea — getting smarter about *how* machines decide:**

1. **Rule-based AI** — a human writes every rule.
   ```
   IF temperature > 100°F THEN show alert "High fever detected"
   ```
   Predictable and logical, but it **can't adapt**: change the situation slightly and it fails.
   *Used in:* early chatbots, chess programs, thermostats.
2. **Machine Learning** — the machine learns patterns **from data** (spam or not? cat or dog? which movie next?).
3. **Modern / Generative AI** — learns from massive data and **creates** new text, images, code — like ChatGPT.
"""
        )
    with c2:
        st.graphviz_chart(
            """
digraph {
  rankdir=TB; node [shape=box style="rounded,filled" fontname="Helvetica"];
  AI [label="Artificial Intelligence\\n(umbrella: any 'smart' behaviour)" fillcolor="#dbeafe"];
  ML [label="Machine Learning\\n(learns from data)" fillcolor="#bfdbfe"];
  DL [label="Deep Learning\\n(multi-layer neural networks)" fillcolor="#93c5fd"];
  GA [label="Generative AI\\n(creates new content)" fillcolor="#60a5fa" fontcolor="white"];
  AI -> ML [label=" subset"]; ML -> DL [label=" subset"]; DL -> GA [label=" powers"];
}
"""
        )
        st.caption("AI is the umbrella · ML is a method within AI · DL is a technique within ML.")

    st.markdown("#### Comparison: AI vs ML vs DL")
    st.dataframe(pd.DataFrame({
        "Aspect": ["Scope", "How it works", "Data needed", "Human effort", "Example"],
        "AI": ["Broadest — any intelligent behaviour", "Rules *or* learning", "Can be none (rules)",
               "Write rules / design system", "Thermostat, chess engine, Alexa"],
        "ML": ["Subset of AI", "Learns patterns from data", "Moderate, often labeled",
               "Choose features & model", "Spam filter, Netflix recommendations"],
        "DL": ["Subset of ML", "Many layers of neurons learn features themselves",
               "Large (often unstructured)", "Mostly architecture & data", "Face ID, speech recognition, ChatGPT"],
    }), hide_index=True, width="stretch")

    st.markdown("#### Where AI is used today")
    cards([
        ("🏥 Healthcare", "Detects diseases like cancer from medical images; IBM Watson Health suggests treatment plans."),
        ("💳 Finance", "Real-time fraud detection by spotting unusual spending patterns."),
        ("🎧 Customer service", "24/7 chatbots; Alexa and Google Assistant respond to voice commands."),
        ("🚗 Transportation", "Self-driving cars (e.g., Tesla Autopilot) perceive roads and decide."),
        ("🛒 E-commerce", "Recommendations learned from your behaviour — like Netflix using watch history."),
    ], cols=5)

# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Machines learn like people do — three ways")
    analogy("Some students learn with a teacher (answers given), some explore on their own, "
            "and some learn by trial and error. Machines do the same.")
    a, b, c = st.columns(3)
    with a:
        with st.container(border=True):
            st.markdown("### 👩‍🏫 Supervised")
            st.markdown("Trained on **input → label** pairs. Learns the mapping, then predicts labels for new inputs.")
            st.markdown("**Regression** → a *number* (house price from size, rooms, location).")
            st.markdown("**Classification** → a *category* (email: Spam / Not Spam).")
    with b:
        with st.container(border=True):
            st.markdown("### 🧭 Unsupervised")
            st.markdown("*Learning without a teacher*: no labels, finds structure on its own — like sorting a messy room.")
            st.markdown("**Clustering** → who belongs together? (Netflix groups sci-fi bingers).")
            st.markdown("**Dimensionality reduction** → simplify 1000 recipe ingredients into themes: sweet, spicy, fresh, savory.")
    with c:
        with st.container(border=True):
            st.markdown("### 🐶 Reinforcement")
            st.markdown("An **agent** acts in an **environment**, gets **rewards/penalties**, and learns a **policy** "
                        "that maximises total reward. *Try → feedback → improve → repeat* (like training a dog with treats).")
            st.markdown("**Model-based** → builds a mental map of the maze and plans ahead.")
            st.markdown("**Model-free** → no map, learns purely from experience.")

    st.markdown("#### 🎮 Which type of learning is it?")
    scenarios = {
        "Predict tomorrow's temperature from past weather data": ("Supervised — Regression", "Output is a continuous number and we have past answers."),
        "Group customers by shopping behaviour, no labels given": ("Unsupervised — Clustering", "No labels; we look for natural groups."),
        "Teach a robot to walk by rewarding forward steps": ("Reinforcement", "Learning by trial, error and reward."),
        "Decide whether a tumour image is benign or malignant": ("Supervised — Classification", "Output is one of a fixed set of classes."),
        "Compress 500 survey questions into 5 key themes": ("Unsupervised — Dimensionality reduction", "Fewer features, same information."),
    }
    pick = st.selectbox("Pick a scenario", list(scenarios))
    guess = st.radio("Your answer", ["Supervised — Regression", "Supervised — Classification",
                                     "Unsupervised — Clustering", "Unsupervised — Dimensionality reduction",
                                     "Reinforcement"], index=None, horizontal=True, key="m1_guess")
    if guess:
        ans, why = scenarios[pick]
        (st.success if guess == ans else st.error)(f"{'Correct!' if guess == ans else 'Not quite.'} **{ans}** — {why}")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Life cycle of a machine-learning project")
    stepper("ml_lifecycle", [
        ("Collect data", "Machines learn only from the data you give them. **Quality decides accuracy** — "
                         "outdated or wrong data → wrong predictions."),
        ("Prepare the data", "Randomise (avoid bias), clean missing/unwanted values, restructure, visualise, then "
                             "**split** into a *training set* (to learn) and a *testing set* (to check)."),
        ("Choose a model", "Pick a model that fits the task (prediction, image, speech) and the data type "
                           "(numerical vs categorical)."),
        ("Train", "Feed training data so the model learns patterns. More (good) training → better performance."),
        ("Evaluate", "Test on **unseen** data. Testing on training data gives unrealistically high accuracy."),
        ("Tune parameters", "Adjust settings (hyper-parameters) chosen by the programmer to boost accuracy."),
        ("Predict", "Use the model on new, real-world data."),
    ])

# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Deep learning: layers of artificial neurons")
    definition("Neural network", "interconnected nodes (neurons) that process data and learn patterns — loosely "
                                 "inspired by the brain. 'Deep' = many hidden layers.")
    st.graphviz_chart(
        """
digraph {
  rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#eef2ff" fontname="Helvetica"];
  D [label="1 · Data\\n(images, text, audio)"]; I [label="2 · Input layer\\n(one neuron per feature)"];
  H [label="3 · Hidden layers\\n(edges → shapes → objects)" fillcolor="#c7d2fe"];
  O [label="4 · Output layer\\n(class / number / sequence)"]; L [label="5 · Loss\\n(how wrong?)" fillcolor="#fee2e2"];
  B [label="6 · Backpropagation\\n(blame flows backward)" fillcolor="#fee2e2"]; W [label="7 · Weight update\\n(gradient descent)" fillcolor="#fee2e2"];
  T [label="8 · Trained model" fillcolor="#dcfce7"];
  D->I->H->O->L->B->W; W->H [label=" repeat many times" style=dashed]; O->T [style=dotted];
}
"""
    )
    st.markdown("#### 🧪 Mini experiment: one neuron learning by gradient descent")
    st.caption("Goal: learn `y = 2x`. The neuron starts with a random weight, measures its loss, and nudges the weight.")
    lr = st.slider("Learning rate (try 1.4+ to see it overshoot and diverge)", 0.01, 1.5, 0.1, 0.01)
    epochs = st.slider("Training iterations", 1, 40, 15)
    xs = [1, 2, 3, 4]
    w, rows = 0.2, []
    for ep in range(epochs):
        preds = [w * x for x in xs]
        loss = sum((p - 2 * x) ** 2 for p, x in zip(preds, xs)) / len(xs)
        grad = sum(2 * (p - 2 * x) * x for p, x in zip(preds, xs)) / len(xs)
        rows.append({"iteration": ep + 1, "weight": w, "loss": loss})
        w -= lr * grad / 10
    df = pd.DataFrame(rows).set_index("iteration")
    c1, c2 = st.columns(2)
    c1.line_chart(df["loss"], height=220)
    c2.line_chart(df["weight"], height=220)
    st.caption(f"Final weight ≈ **{w:.3f}** (target 2.0). Too high a learning rate overshoots; too low learns slowly.")

# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Generative AI — machines that create")
    definition("Generative AI", "AI that creates *new* data — text, images, audio, video — by learning patterns from "
                                "the data it was trained on. Traditional AI classifies or predicts; GenAI produces.")
    analogy("Type “Write a poem about a rainy evening in Mumbai” and you get a brand-new poem in seconds. "
            "You didn't provide the poem — the model imagined it from patterns it learned.")
    st.markdown("#### 70 years: from mimicry to mastery")
    timeline = [
        ("1950", "Turing Test", "“Can machines think?” — the seed of conversational AI."),
        ("1966", "ELIZA", "Rule-based MIT chatbot mimicking a therapist. No understanding, just mirroring."),
        ("1970s", "Markov chains", "First statistical text generation: next word by probability."),
        ("1985", "Boltzmann machines", "Hinton's stochastic networks model probability distributions."),
        ("2000s", "LDA & RNNs", "Hidden topics; RNNs gain short memory for sequences (but forget long context)."),
        ("2014", "GANs", "Generator vs discriminator compete → realistic faces, art, deepfakes."),
        ("2017", "Transformer", "“Attention Is All You Need” — whole sequences at once, long-range context."),
        ("2018", "BERT", "Reads bidirectionally → deep language understanding (Google Search)."),
        ("2019–20", "GPT-2 / GPT-3", "Web-scale pre-training; essays, code, music."),
        ("2021", "Codex · DALL·E · CLIP", "Text → code and text → image."),
        ("2022–25", "ChatGPT · GPT-4 · Gemini · Claude · Mistral", "Multimodal, memory-enabled, reasoning partners."),
    ]
    year = st.select_slider("Slide through history", [t[0] for t in timeline], value="2017")
    for y, name, desc in timeline:
        if y == year:
            st.success(f"**{y} — {name}**: {desc}")
    st.dataframe(pd.DataFrame(timeline, columns=["Year", "Milestone", "Why it mattered"]), hide_index=True, width="stretch")

    st.markdown("#### Hugging Face — the open-model library")
    cards([
        ("Pre-trained models", "Thousands of free models to download."),
        ("Transformers library", "Open-source Python library to run them."),
        ("Spaces", "Deploy and share AI apps like chatbots."),
        ("Inference API", "Free API for quick prototyping."),
        ("HuggingChat", "Try open LLMs in the browser."),
        ("AI agents", "e.g. Open Computer Agent that drives a virtual computer."),
    ], cols=3)

# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("How a model picks the next word")
    st.markdown(
        "A language model doesn't know the answer in advance — it scores every possible next token, then "
        "**samples** one. Three dials control that choice:"
    )
    cards([
        ("🌡️ Temperature — the risk dial", "Low (0.1–0.3): confident, always the top word.\n\n"
                                          "Mid (0.5–0.7): balanced.\n\nHigh (0.8–1.0+): adventurous, more random."),
        ("🔝 Top-k", "Only consider the **k** most likely tokens. Fixed-size shortlist — cuts off nonsense."),
        ("🎯 Top-p (nucleus)", "Keep the smallest group of tokens whose probabilities add up to **p**. "
                               "The shortlist grows when the model is unsure and shrinks when it's confident."),
    ])
    sentence = st.radio("Sentence", [sim.CAT_SENTENCE, sim.STARS_SENTENCE], horizontal=True)
    base = sim.CAT_PROBS if sentence == sim.CAT_SENTENCE else sim.STARS_PROBS
    c1, c2, c3 = st.columns(3)
    t = c1.slider("Temperature", 0.05, 2.0, 1.0, 0.05, key="m1_t")
    k = c2.slider("Top-k", 1, 10, 10, key="m1_k")
    p = c3.slider("Top-p", 0.1, 1.0, 1.0, 0.05, key="m1_p")
    df = sim.decode_table(base, t, k, p)
    l, r = st.columns([3, 2])
    with l:
        st.bar_chart(df.set_index("token")[["original %", "final sampling %"]], height=300, stack=False)
    with r:
        st.dataframe(df, hide_index=True, width="stretch", height=300)
    if st.button("🎲 Generate 20 completions with these settings"):
        draws = sim.sample(df, 20)
        st.write(" · ".join(f"**{w}**" for w in draws))
        st.caption(f"{len(set(draws))} distinct words in 20 tries.")
    with st.expander("What to notice"):
        st.markdown(
            "- Temperature **0.05** → essentially *greedy*: almost always “mat”/“twinkled”.\n"
            "- Temperature **2.0** flattens the bars → “spaceship” and “banana” become real possibilities.\n"
            "- **Top-k = 3** keeps exactly 3 words no matter how the probabilities look.\n"
            "- **Top-p = 0.9** keeps however many words are needed to cover 90% — a *dynamic* shortlist.\n"
            "- The two filters can be combined — a token must pass both."
        )

    st.markdown("#### 🟢 Live: the same prompt at different temperatures")
    prompt = st.text_input("Prompt", "Write a one-line tagline for a neighbourhood chai stall.", key="m1_live_prompt")

    def _temps():
        return [(temp, generate(prompt, temperature=temp, max_tokens=None)[0]) for temp in (0.0, 0.7, 1.5)]

    res = run_live("Run at T = 0.0, 0.7 and 1.5", _temps, key="m1_live")
    if res:
        cols = st.columns(3)
        for col, (temp, txt) in zip(cols, res):
            with col:
                with st.container(border=True):
                    st.markdown(f"**Temperature {temp}**")
                    st.write(txt)
        st.caption("Run it twice: T = 0 should barely change, T = 1.5 should surprise you.")
    st.code(
        """from google import genai
from google.genai import types

client = genai.Client(api_key=API_KEY)
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Write a one-line tagline for a chai stall.",
    config=types.GenerateContentConfig(temperature=1.2, top_p=0.9, top_k=40),
)
print(resp.text)""",
        language="python",
    )

# ---------------------------------------------------------------------------
with tabs[6]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ✅ Benefits")
        for title, ex in [
            ("Enhanced productivity", "drafts email replies, summarises documents, writes reports"),
            ("Creativity booster", "a writer explores alternative story endings"),
            ("Personalisation at scale", "product descriptions tailored to each shopper"),
            ("Education & learning", "AI tutors explain physics in plain words, make quizzes"),
            ("Data augmentation", "synthetic medical records to train models without privacy risk"),
        ]:
            st.markdown(f"- **{title}** — *e.g.* {ex}")
    with c2:
        st.markdown("### ⚠️ Challenges")
        for title, ex in [
            ("Bias & misinformation", "a writing tool repeats stereotypes found in web data"),
            ("Plagiarism & originality", "AI art that resembles existing works"),
            ("Job displacement", "auto content generators replacing junior copywriters"),
            ("Security & privacy", "deepfake audio impersonating public figures, phishing"),
            ("Over-reliance", "a student copies AI answers without checking them"),
        ]:
            st.markdown(f"- **{title}** — *e.g.* {ex}")

takeaway([
    "AI ⊃ ML ⊃ DL; Generative AI is built on deep learning and *creates* new content.",
    "Supervised (labels), unsupervised (patterns) and reinforcement (rewards) are the three ways machines learn.",
    "Deep learning loops: predict → measure loss → backpropagate → update weights → repeat.",
    "Temperature controls randomness; top-k and top-p limit which tokens can be sampled.",
    "GenAI boosts productivity and creativity but brings bias, IP, privacy and over-reliance risks.",
])

quiz("m1", [
    {"q": "Which statement about AI, ML and DL is correct?",
     "options": ["ML is broader than AI", "DL is a technique within ML, which is within AI", "DL and AI are the same", "ML doesn't need data"],
     "answer": 1, "why": "AI is the umbrella, ML a method in it, DL a technique in ML."},
    {"q": "Predicting a house price is an example of…",
     "options": ["Classification", "Clustering", "Regression", "Reinforcement learning"],
     "answer": 2, "why": "The output is a continuous number."},
    {"q": "You want factual, repeatable answers. Which temperature?",
     "options": ["0.1–0.3", "0.8–1.0", "1.5+", "Temperature doesn't matter"],
     "answer": 0, "why": "Low temperature makes the model pick the most probable tokens."},
    {"q": "How is top-p different from top-k?",
     "options": ["Top-p always keeps exactly p tokens", "Top-p keeps a dynamic set covering probability p",
                 "Top-p only works at temperature 0", "There is no difference"],
     "answer": 1, "why": "Top-k is a fixed count; top-p adapts to how spread-out the probabilities are."},
    {"q": "Which 2017 paper introduced the Transformer?",
     "options": ["Generative Adversarial Nets", "Attention Is All You Need", "BERT", "ELIZA"],
     "answer": 1, "why": "Transformers process whole sequences with attention, capturing long-range context."},
])
