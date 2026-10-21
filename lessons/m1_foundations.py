import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core import sim
from core.ui import analogy, cards, definition, page_header, quiz, takeaway

page_header(
    "m1",
    "From rule-based systems to machines that create — and the dials that control their creativity.",
    [
        "Explain AI, ML and DL and how they nest inside each other",
        "Tell supervised, unsupervised and reinforcement learning apart, with an example of each technique",
        "Understand data handling (collect, clean, split) and its role in training models",
        "Walk through the ML life cycle and the deep-learning training loop",
        "Describe what makes Generative AI different and how it evolved",
        "Use temperature, top-k and top-p intuitively",
        "Weigh the benefits and risks of GenAI",
    ],
)

tabs = st.tabs(["AI · ML · DL", "Types of ML", "ML life cycle", "Deep learning", "Generative AI",
                "🎛️ Temperature / Top-k / Top-p", "Benefits & risks", "🧾 Key takeaways", "✅ Knowledge check"])


# ---------------------------------------------------------------------------
# Helpers for this page
# ---------------------------------------------------------------------------
LIFECYCLE = [
    ("Collect data", "📥",
     "Machines learn only from the data you give them. **Quality decides accuracy**: outdated or wrong data gives wrong predictions.",
     "We gather 5,000 past house sales: size, rooms, location and the **price each one sold for**."),
    ("Prepare the data", "🧹",
     "Randomise (to avoid bias), clean missing or unwanted values, restructure, visualise, then **split** into a "
     "*training set* (to learn from) and a *testing set* (to check the result).",
     "Remove sales with a missing size, fix a typo like a 2-million-sq-ft flat, shuffle the rows, then keep 4,000 for training and 1,000 hidden for testing."),
    ("Choose a model", "🧩",
     "Pick a model that fits the task (prediction, image, speech) and the data type (numerical vs categorical).",
     "We want a **number** (a price), so we pick a regression model, for example linear regression."),
    ("Train", "🏋️",
     "Feed the training data so the model learns patterns. More (good) training usually means better performance.",
     "The model sees 4,000 houses and adjusts itself until its guessed prices are close to the real ones."),
    ("Evaluate", "📏",
     "Test on **unseen** data. Testing on the training data gives unrealistically high accuracy.",
     "We ask it to price the 1,000 hidden houses and find it is off by about $20k on average. Is that good enough?"),
    ("Tune parameters", "🎛️",
     "Adjust settings (hyper-parameters) chosen by the programmer to boost accuracy, then train and evaluate again.",
     "Try a different model, add a 'distance to metro' feature, or change the learning rate. Now the average error drops to $12k."),
    ("Predict", "🔮",
     "Use the model on new, real-world data. Keep watching it: if the world changes, go back to step 1 with fresh data.",
     "A seller types in a new flat's details and gets an instant estimate. Next year the market shifts, so we collect new sales and repeat."),
]


def lifecycle_dot(active: int) -> str:
    """Graphviz source for the ML life cycle (two rows, big text) with step `active` (0-based) highlighted."""
    nodes = []
    for i, (name, icon, _, _) in enumerate(LIFECYCLE):
        fill = "#fb923c" if i == active else ("#bfdbfe" if i < active else "#f1f5f9")
        nodes.append(f'  n{i} [label="{i + 1}. {name}" fillcolor="{fill}"];')
    red = 'style=dashed color="#dc2626" fontcolor="#dc2626" constraint=false'
    green = 'style=dashed color="#16a34a" fontcolor="#16a34a" constraint=false'
    return (
        'digraph { rankdir=TB; nodesep=0.5; ranksep=0.9; '
        'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=18 margin="0.25,0.15" color="#475569" penwidth=1.5]; '
        'edge [color="#475569" fontname="Helvetica" fontsize=14 penwidth=1.6];\n'
        + "\n".join(nodes) +
        "\n  { rank=same; n0; n1; n2; n3; }\n  { rank=same; n4; n5; n6; }\n"
        "  n0 -> n1 -> n2 -> n3;\n"
        "  n3 -> n4;\n"
        "  n6 -> n5 -> n4 [dir=back];\n"
        f'  n4 -> n1 [label=" poor accuracy:\\n fix the data" {red}];\n'
        f'  n5 -> n3 [label=" retrain" {red}];\n'
        f'  n6 -> n0 [label=" world changes: collect new data" {green}];\n}}'
    )



# --- tiny maze used to contrast model-based and model-free reinforcement learning -------------
MZ_N, MZ_START, MZ_GOAL, MZ_TRAP = 4, (0, 0), (3, 3), (1, 2)
MZ_ACTS = {"↑": (-1, 0), "↓": (1, 0), "←": (0, -1), "→": (0, 1)}


def mz_step(state, act):
    """The world's rules: where you land and what you get. A model-based agent is GIVEN these rules."""
    r, c = state
    dr, dc = MZ_ACTS[act]
    nr, nc = min(max(r + dr, 0), MZ_N - 1), min(max(c + dc, 0), MZ_N - 1)
    ns = (nr, nc)
    if ns == MZ_GOAL:
        return ns, 10.0, True
    if ns == MZ_TRAP:
        return ns, -10.0, True
    return ns, -1.0, False


def mz_model_free(episodes: int):
    """Q-learning: no map, only experience. Returns (policy, steps_taken)."""
    rng = np.random.default_rng(3)
    Q = {(r, c): {a: 0.0 for a in MZ_ACTS} for r in range(MZ_N) for c in range(MZ_N)}
    steps = 0
    for _ in range(episodes):
        st_ = MZ_START
        for _ in range(40):
            a = rng.choice(list(MZ_ACTS)) if rng.random() < 0.3 else max(Q[st_], key=Q[st_].get)
            ns, rew, done = mz_step(st_, a)
            steps += 1
            Q[st_][a] += 0.5 * (rew + (0 if done else 0.9 * max(Q[ns].values())) - Q[st_][a])
            st_ = ns
            if done:
                break
    return {s_: (max(q, key=q.get) if any(q.values()) else "·") for s_, q in Q.items()}, steps


def mz_model_based():
    """Value iteration: uses the known rules to plan 'in its head'. Zero real attempts."""
    V = {(r, c): 0.0 for r in range(MZ_N) for c in range(MZ_N)}
    for _ in range(30):
        for s_ in V:
            if s_ in (MZ_GOAL, MZ_TRAP):
                continue
            V[s_] = max(rew + (0 if done else 0.9 * V[ns]) for ns, rew, done in (mz_step(s_, a) for a in MZ_ACTS))
    pol = {}
    for s_ in V:
        if s_ in (MZ_GOAL, MZ_TRAP):
            continue
        pol[s_] = max(MZ_ACTS, key=lambda a: (lambda x: x[1] + (0 if x[2] else 0.9 * V[x[0]]))(mz_step(s_, a)))
    return pol


def mz_show(policy: dict):
    rows = []
    for r in range(MZ_N):
        row = {}
        for c in range(MZ_N):
            cell = (r, c)
            row[f"col {c}"] = "🏆 treasure" if cell == MZ_GOAL else "⚠️ trap" if cell == MZ_TRAP else (
                ("S " if cell == MZ_START else "") + policy.get(cell, "·"))
            if cell == MZ_START and cell not in (MZ_GOAL, MZ_TRAP):
                row[f"col {c}"] = "🚩 start " + policy.get(cell, "·")
        rows.append(row)
    st.table(pd.DataFrame(rows, index=[f"row {r}" for r in range(MZ_N)]))
    # follow the learned policy from the start
    pos, path = MZ_START, 0
    for _ in range(12):
        a = policy.get(pos, "·")
        if a == "·":
            return "The agent has not learned a route from the start yet."
        pos, _, done = mz_step(pos, a)
        path += 1
        if done:
            return ("Following the arrows from the start reaches the 🏆 treasure in %d steps." % path) if pos == MZ_GOAL else \
                   "Following the arrows from the start falls into the ⚠️ trap."
    return "Following the arrows from the start wanders without reaching the treasure."


def reset_guess():
    """Clear the previous answer when a new scenario is picked."""
    st.session_state["m1_guess"] = None


def scatter(df, x, y, xt, yt, color=None, ctitle=None, height=280):
    enc = dict(x=alt.X(x, title=xt, scale=alt.Scale(zero=False)), y=alt.Y(y, title=yt, scale=alt.Scale(zero=False)))
    if color:
        enc["color"] = alt.Color(color, title=ctitle, scale=alt.Scale(range=["#2563eb", "#f59e0b", "#16a34a", "#dc2626"]))
    return alt.Chart(df).mark_circle(size=55, opacity=0.8).encode(**enc).properties(height=height)

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

    st.markdown("#### How does Netflix know what you like? (Machine Learning)")
    n1, n2 = st.columns([3, 2])
    with n1:
        st.markdown("It didn't guess. It **learned from your actions**: (1) what you watched, (2) what you skipped, (3) what people similar to you watched. "
                    "Over time it found patterns and started predicting what you would enjoy.")
        definition("Machine Learning (ML)", "a subset of AI where machines are trained to learn from data, adapt over time and make decisions, "
                                            "almost like a self-improving student.")
    with n2:
        st.markdown("**Powers things like:**\n- Spotify's song recommendations\n- Google Translate improving over time\n- Self-driving cars learning to navigate roads")
    st.caption("AI is the big umbrella; machine learning is the brain that learns, and it is everywhere around us.")

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
            st.markdown("Trained on a dataset of **input → label** pairs. *Input* = the data we give (height, weight, age). "
                        "*Label* = the correct answer we expect (\"healthy\" / \"not healthy\"). It learns the pattern, then predicts labels for **new, unseen** inputs, "
                        "like solving new maths problems without help after enough practice.")
            st.markdown("**Regression** → a *number* that can go up or down smoothly. Inputs: size, bedrooms, location. Label: price in dollars.")
            st.markdown("**Classification** → a *category* from a fixed set. Inputs: words in the email, subject line, sender's address. Label: Spam / Not Spam.")
    with b:
        with st.container(border=True):
            st.markdown("### 🧭 Unsupervised")
            st.markdown("*Learning without a teacher*: no labels or predefined answers. Like walking into a room of jumbled objects and, with no instructions, "
                        "sorting toys into one corner and books into another because they look alike.")
            st.markdown("**Clustering** → *who belongs together?* Netflix groups users who binge sci-fi, comedy or thrillers, then builds \"Top Picks for You\".")
            st.markdown("**Dimensionality reduction** → *simplifying the mess.* 1000 recipe ingredients become a few themes (sweet, spicy, fresh, savory) "
                        "because North-Indian dishes, desserts and salads each reuse similar sets.")
    with c:
        with st.container(border=True):
            st.markdown("### 🐶 Reinforcement")
            st.markdown("An **agent** acts in an **environment**, gets **rewards or penalties**, and learns a **policy** (a mapping from *states* to *actions*) "
                        "that maximises the expected total reward over time. *Try → get feedback → improve → repeat*, like giving a dog a treat when it sits.")
            st.markdown("Maze game: find the shortest path to the treasure (reward) and avoid traps (penalties). Two ways to learn it:")
            st.markdown("**Model-based** → given or learns a model (how moves change state, what each action rewards), then simulates outcomes in its head and plans ahead.")
            st.markdown("**Model-free** → no map, just experience: act, get reward, update. Often slower, but works when the world is too complex to model.")

    st.markdown("#### 🔬 One worked example for each type")
    st.caption("The data below is made up (random but fixed), so you can see what each technique actually does.")
    rng = np.random.default_rng(7)
    ex = st.tabs(["Regression", "Classification", "Clustering", "Dimensionality reduction", "Model-based", "Model-free"])

    with ex[0]:
        st.markdown("**Question:** what will this house sell for? &nbsp; **Output:** a *number*.")
        size = rng.uniform(500, 2500, 60)
        price = 40 + 0.11 * size + rng.normal(0, 18, 60)
        slope, icpt = np.polyfit(size, price, 1)
        d = pd.DataFrame({"size": size, "price": price})
        line = pd.DataFrame({"size": [500, 2500], "price": [icpt + slope * 500, icpt + slope * 2500]})
        ch = scatter(d, "size", "price", "House size (square feet)", "Sale price ($ thousand)") + \
            alt.Chart(line).mark_line(color="#dc2626", strokeWidth=3).encode(x="size", y="price")
        st.altair_chart(ch, width="stretch")
        q = st.slider("Try it: size of a new house (sq ft)", 500, 2500, 1500, 50, key="m1_reg_q")
        st.success(f"Each dot is a past sale (the labelled training data). The red line is the learned pattern. "
                   f"A {q} sq ft house is predicted at about **${icpt + slope * q:,.0f}k**.")

    with ex[1]:
        st.markdown("**Question:** is this email spam? &nbsp; **Output:** a *category*.")
        n = 40
        ham = pd.DataFrame({"links": rng.normal(1.5, 0.8, n), "exclaims": rng.normal(1.0, 0.8, n), "label": "Not spam"})
        spam = pd.DataFrame({"links": rng.normal(5.5, 1.2, n), "exclaims": rng.normal(5.0, 1.3, n), "label": "Spam"})
        d = pd.concat([ham, spam], ignore_index=True)
        c1, c2 = st.columns(2)
        links = c1.slider("Links in the new email", 0.0, 9.0, 4.0, 0.5, key="m1_cls_l")
        exc = c2.slider("Exclamation marks in the new email", 0.0, 9.0, 4.0, 0.5, key="m1_cls_e")
        cen = {k: g[["links", "exclaims"]].mean().to_numpy() for k, g in d.groupby("label")}
        pt = np.array([links, exc])
        guess = min(cen, key=lambda k: np.linalg.norm(pt - cen[k]))
        new = pd.DataFrame({"links": [links], "exclaims": [exc], "label": ["NEW email"]})
        ch = alt.Chart(d).mark_circle(size=55, opacity=0.75).encode(
            x=alt.X("links", title="Links in the email"), y=alt.Y("exclaims", title="Exclamation marks in the email"),
            color=alt.Color("label", title="Known label", scale=alt.Scale(domain=["Not spam", "Spam", "NEW email"], range=["#2563eb", "#dc2626", "#000000"]))) + \
            alt.Chart(new).mark_point(size=260, shape="diamond", filled=True).encode(x="links", y="exclaims", color="label")
        st.altair_chart(ch.properties(height=280), width="stretch")
        st.success(f"Each dot is an email whose label we already know. The new email (black diamond) sits closest to the "
                   f"**{guess}** group, so the prediction is **{guess}**.")

    with ex[2]:
        st.markdown("**Question:** which customers behave alike? &nbsp; **Output:** groups nobody labelled for us.")
        pts = np.vstack([rng.normal([2, 15], [0.8, 4], (30, 2)), rng.normal([6, 55], [1, 8], (30, 2)), rng.normal([10, 30], [0.9, 6], (30, 2))])
        d = pd.DataFrame(pts, columns=["visits", "spend"])
        cent = pts[[0, 35, 70]].copy()
        for _ in range(15):                                   # a tiny k-means, k = 3
            lab = np.argmin(((pts[:, None] - cent[None]) ** 2).sum(-1), axis=1)
            cent = np.array([pts[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(3)])
        d["group"] = ["Group " + str(i + 1) for i in lab]
        c1, c2 = st.columns(2)
        with c1:
            st.caption("**Before:** raw data, no labels")
            st.altair_chart(scatter(d, "visits", "spend", "Store visits per month", "Spend per month ($)", height=260), width="stretch")
        with c2:
            st.caption("**After:** the algorithm found 3 groups on its own (k-means)")
            st.altair_chart(scatter(d, "visits", "spend", "Store visits per month", "Spend per month ($)", "group", "Found group", height=260), width="stretch")
        st.info("We only chose *how many* groups (3). The names are ours to invent afterwards, e.g. occasional, regular, big spenders.")

    with ex[3]:
        st.markdown("**Question:** can fewer numbers describe the same thing? &nbsp; **Output:** a simpler version of the data.")
        h = rng.normal(170, 9, 80)
        wt = 0.9 * (h - 170) + 68 + rng.normal(0, 3, 80)
        X = np.column_stack([h, wt]); mu = X.mean(0)
        _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        v = Vt[0]; share = S[0] ** 2 / (S ** 2).sum()
        d = pd.DataFrame({"height": h, "weight": wt})
        a, b = mu - 30 * v, mu + 30 * v
        axis = pd.DataFrame({"height": [a[0], b[0]], "weight": [a[1], b[1]]})
        ch = scatter(d, "height", "weight", "Height (cm)", "Weight (kg)") + \
            alt.Chart(axis).mark_line(color="#dc2626", strokeWidth=3).encode(x="height", y="weight")
        st.altair_chart(ch, width="stretch")
        st.success(f"Height and weight move together, so **2 numbers per person can become 1** (position along the red line, "
                   f"called the first principal component) and we still keep about **{share:.0%}** of the information (variance). "
                   "This is PCA. The same idea turns 500 survey questions into 5 themes.")

    with ex[4]:
        st.markdown("**Reinforcement learning with a model.** The agent first learns (or is given) a **map of how the world works**, then *plans* with it.")
        st.markdown("**Example:** a delivery robot is given the floor plan of a warehouse. Before moving it can *imagine* routes, "
                    "check each against the map, and pick the shortest. A chess engine that looks several moves ahead works the same way.")
        st.markdown("- ➕ Needs far fewer real attempts (it practises in its head)\n- ➕ Easy to explain a decision\n"
                    "- ➖ Only as good as the map: a wrong model gives confidently wrong plans\n- ➖ Building the model can be hard")
    with ex[5]:
        st.markdown("**Reinforcement learning without a model.** The agent has **no map**; it just tries actions and keeps what earned reward.")
        st.markdown("**Example:** a robot dropped into an unknown maze bumps into walls, gets a small penalty for each wall and a big reward at the exit. "
                    "After thousands of tries it knows *\"in this spot, turn left\"* without ever knowing the layout. Q-learning and the agents "
                    "that learned Atari games are model-free.")
        st.markdown("- ➕ Works when the world is too complex to model\n- ➕ Simple idea\n"
                    "- ➖ Needs a huge number of real attempts\n- ➖ Mistakes while learning can be costly")
    with st.container(border=True):
        st.markdown("##### 🧭 Reinforcement learning demo: the same maze solved model-based and model-free")
        st.caption("4×4 maze. 🚩 start top-left, 🏆 treasure bottom-right (+10), ⚠️ trap (−10), every step costs −1. Arrows show the best move the agent believes in each square.")
        mb, mf = st.columns(2)
        with mb:
            st.markdown("**Model-based** (knows the rules, plans in its head)")
            msg = mz_show(mz_model_based())
            st.caption("Real attempts used: **0**. " + msg)
        with mf:
            st.markdown("**Model-free** (no map, learns by trying)")
            eps = st.slider("Real attempts (episodes) so far", 0, 300, 5, 5, key="m1_mz_eps")
            pol, nsteps = mz_model_free(eps)
            msg = mz_show(pol)
            st.caption(f"Real attempts used: **{eps}** ({nsteps} moves). " + msg)
        st.info("Slide the attempts up: the model-free agent starts clueless (·) and only slowly discovers a route, while the model-based agent plans it at once. "
                "The price: the model-based agent needed the rules handed to it.")
    st.dataframe(pd.DataFrame({
        "": ["Has a map of the world?", "How it decides", "Learning speed (real attempts)", "Example"],
        "Model-based": ["Yes (given or learned)", "Plans ahead using the map", "Fewer attempts", "Warehouse robot with a floor plan"],
        "Model-free": ["No", "Repeats what earned reward", "Many attempts", "Robot learning an unknown maze"],
    }), hide_index=True, width="stretch")

    st.markdown("#### 🎮 Which type of learning is it?")
    scenarios = {
        "Predict tomorrow's temperature from past weather data": ("Supervised — Regression", "Output is a continuous number and we have past answers."),
        "Group customers by shopping behaviour, no labels given": ("Unsupervised — Clustering", "No labels; we look for natural groups."),
        "Teach a robot to walk by rewarding forward steps": ("Reinforcement", "Learning by trial, error and reward."),
        "Decide whether a tumour image is benign or malignant": ("Supervised — Classification", "Output is one of a fixed set of classes."),
        "Compress 500 survey questions into 5 key themes": ("Unsupervised — Dimensionality reduction", "Fewer features, same information."),
    }
    pick = st.selectbox("Pick a scenario", list(scenarios), key="m1_scenario", on_change=reset_guess)
    guess = st.radio("Your answer", ["Supervised — Regression", "Supervised — Classification",
                                     "Unsupervised — Clustering", "Unsupervised — Dimensionality reduction",
                                     "Reinforcement"], index=None, horizontal=True, key="m1_guess")
    if guess:
        ans, why = scenarios[pick]
        (st.success if guess == ans else st.error)(f"{'Correct!' if guess == ans else 'Not quite.'} **{ans}** — {why}")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Life cycle of a machine-learning project")
    st.markdown("Every ML project follows the same loop. The diagram shows the seven steps; **the orange box is the step explained below it**. "
                "Dashed arrows are the loops back that happen in real projects. The flow snakes from the top row down to the bottom row. Running example: **predicting house prices**.")
    if "ml_life_step" not in st.session_state:
        st.session_state["ml_life_step"] = 0
    i = st.session_state["ml_life_step"]
    st.graphviz_chart(lifecycle_dot(i), width="stretch")
    st.caption("Blue = done, orange = you are here, grey = still to come. Red dashed arrows: go back when results are poor. "
               "Green dashed arrow: the cycle never really ends, because the world changes.")
    name, icon, what, example = LIFECYCLE[i]
    st.progress((i + 1) / len(LIFECYCLE), text=f"Step {i + 1} of {len(LIFECYCLE)}")
    with st.container(border=True):
        st.markdown(f"### {icon} {i + 1}. {name}")
        st.markdown(what)
        st.info(f"**House-price example:** {example}", icon="🏠")
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("◀ Back", key="ml_life_back", disabled=i == 0, width="stretch"):
        st.session_state["ml_life_step"] -= 1
        st.rerun()
    if c2.button("Next ▶", key="ml_life_next", disabled=i == len(LIFECYCLE) - 1, width="stretch"):
        st.session_state["ml_life_step"] += 1
        st.rerun()
    with st.expander("Why does the cycle loop back?"):
        st.markdown("- **Evaluate → Prepare data:** if accuracy is poor the cause is usually the data, not the model.\n"
                    "- **Evaluate → Tune → Train:** change settings, retrain, test again until the result is good enough.\n"
                    "- **Predict → Collect data:** prices, tastes and fraud tricks change, so a model gets stale and must be refreshed.")

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
    with st.expander("The 8 steps in words", expanded=False):
        st.markdown(
            "1. **Data**: images, text, audio or any structured or unstructured input (cat photos, temperature readings, customer reviews).\n"
            "2. **Input layer**: receives the data; each neuron represents one feature (e.g. the brightness of a pixel).\n"
            "3. **Hidden layers**: neurons apply weighted transformations and activation functions to learn features step by step (in an image: edges → shapes → objects). "
            "*'Deep' means many hidden layers.*\n"
            "4. **Output layer**: produces the prediction: a class (\"cat\"), a number (a price) or a sequence (a translated sentence).\n"
            "5. **Compare (loss)**: the prediction is compared with the true answer; the loss function measures the error. Higher loss = more wrong.\n"
            "6. **Backpropagation**: the error is sent backwards through the network to work out how much each weight contributed to it.\n"
            "7. **Weight update**: each connection's weight is nudged slightly (using gradient descent) so the next prediction is better.\n"
            "8. **Trained model**: after many repetitions it predicts well on unseen data and is ready for real-world work.")

    st.markdown("#### 🧪 Mini experiment: one neuron learning by gradient descent")
    st.caption("Goal: learn `y = 2x`. The neuron has one **weight** `w` and predicts `w × x`. The right answer is `w = 2`. "
               "It starts with a wrong weight (0.2), measures how wrong it is (the **loss**), and nudges the weight to be less wrong. Then it repeats.")
    lr = st.slider("Learning rate (step size; try 1.4+ to see it overshoot and diverge)", 0.01, 1.5, 0.1, 0.01)
    epochs = st.slider("Training iterations (how many nudges)", 1, 40, 15)
    xs = [1, 2, 3, 4]
    w, rows = 0.2, []
    for ep in range(epochs + 1):                                  # point k = the state after k nudges
        loss = sum((w * x - 2 * x) ** 2 for x in xs) / len(xs)
        rows.append({"iteration": ep, "weight": w, "loss": loss})
        grad = sum(2 * (w * x - 2 * x) * x for x in xs) / len(xs)
        w -= lr * grad / 10
    df = pd.DataFrame(rows)
    shrink = abs(1 - 1.5 * lr)                                    # each nudge multiplies the error (w - 2) by (1 - 1.5*lr)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Loss: how wrong is the neuron?**")
        st.altair_chart(alt.Chart(df).mark_line(point=True, color="#2563eb").encode(
            x=alt.X("iteration:Q", title="Training iteration (0 = before any learning)"),
            y=alt.Y("loss:Q", title="Loss (0 = perfect)", scale=alt.Scale(zero=True)),
            tooltip=["iteration", alt.Tooltip("loss", format=".2f")]).properties(height=260), width="stretch")
    with c2:
        st.markdown("**Weight: what the neuron has learned**")
        base = alt.Chart(df).mark_line(point=True, color="#16a34a").encode(
            x=alt.X("iteration:Q", title="Training iteration (0 = before any learning)"),
            y=alt.Y("weight:Q", title="Weight w (target = 2.0)"),
            tooltip=["iteration", alt.Tooltip("weight", format=".3f")])
        target = alt.Chart(pd.DataFrame({"y": [2.0]})).mark_rule(color="#dc2626", strokeDash=[6, 4], strokeWidth=2).encode(y="y:Q")
        st.altair_chart((base + target).properties(height=260), width="stretch")
        st.caption("Red dashed line = the correct weight, 2.0.")
    final = rows[-1]
    st.markdown(f"After **{epochs}** iterations: weight = **{final['weight']:.3f}** (target 2.0), loss = **{final['loss']:.2f}**.")
    if lr > 4 / 3:
        st.error(f"Learning rate {lr:.2f} is too high: every step overshoots by more than it corrects (the error is multiplied by {shrink:.2f} each step, "
                 "which is bigger than 1), so the weight swings further and further from 2.0 and the loss grows.")
    elif lr > 0.9:
        st.warning(f"Learning rate {lr:.2f} is high: the weight jumps back and forth across 2.0 (the error flips sign each step) and only slowly settles "
                   f"(it shrinks to {shrink:.2f} of its size per step).")
    elif lr < 0.05:
        st.info(f"Learning rate {lr:.2f} is low: safe but slow. The weight creeps towards 2.0; try more iterations or a bigger rate.")
    else:
        st.success(f"Learning rate {lr:.2f} is a good size: the weight moves smoothly towards 2.0 and the loss falls towards 0.")
    with st.expander("How to read these two graphs"):
        st.markdown(
            "- **x axis (both graphs):** the training iteration. Iteration 0 is the start, before any learning. Each iteration the neuron "
            "looks at the 4 examples (x = 1, 2, 3, 4), measures its error, and nudges the weight.\n"
            "- **Left graph, y axis = loss:** the average of (prediction − correct answer)² over the 4 examples. 0 means perfect. "
            "At the start (w = 0.2) the loss is about 24.3. **Learning = this line going down.**\n"
            "- **Right graph, y axis = weight w:** the one number the neuron is learning. The red dashed line is the right answer, 2.0. "
            "**Learning = this line settling on the red line.**\n"
            "- **Why they can look different:** with a high rate such as 1.3 the weight jumps from below 2 to above 2 and back, "
            "but each jump lands a little closer, so the loss still falls slowly. Both graphs show the same story, in different units.\n"
            "- **Rule of thumb for this toy:** each step multiplies the distance to 2.0 by (1 − 1.5 × learning rate). "
            "Smaller than 1 in size means it converges; above a learning rate of about 1.33 it diverges."
        )
    with st.expander("See the numbers"):
        st.dataframe(df.round(3), hide_index=True, width="stretch")

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
        ("2014–17", "Autoencoders", "Learn to compress and rebuild data: the backbone of image and text generation and anomaly detection."),
        ("2014", "GANs", "Generator vs discriminator compete → realistic faces, art, deepfakes."),
        ("2017", "Transformer", "“Attention Is All You Need” — whole sequences at once, long-range context."),
        ("2018", "BERT", "Reads bidirectionally → deep language understanding (Google Search)."),
        ("2019–20", "GPT-2 / GPT-3", "Web-scale pre-training; essays, code, music."),
        ("2021", "Codex · DALL·E · CLIP", "Text → code and text → image."),
        ("2022–25", "ChatGPT · GPT-4 · Gemini · Claude · Mistral", "Multimodal, memory-enabled, reasoning partners."),
    ]
    st.markdown("**The six phases:** 1950s–70s *mimicking conversation without understanding* → 1980s–90s *building mental representations* → "
                "2000s *the idea of memory is born* → 2014–17 *from analysis to imagination* → 2017–19 *from generating to understanding* → "
                "2020–now *from a tool to a creative and cognitive partner*.")
    year = st.select_slider("Slide through history", [t[0] for t in timeline], value="2017")
    for y, name, desc in timeline:
        if y == year:
            st.success(f"**{y} — {name}**: {desc}")
    st.dataframe(pd.DataFrame(timeline, columns=["Year", "Milestone", "Why it mattered"]), hide_index=True, width="stretch")

    st.caption("\"It took 70 years for machines to move from mimicking language… to speaking it with understanding and purpose.\"")
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
    with st.expander("Examples from the notes: fill-in-the-blank, top-k and top-p"):
        st.markdown(
            "- **Temperature:** in *\"The cat sat on the ___\"*, always guessing \"mat\" is predictable and safe; guessing \"roof\", \"throne\" or \"spaceship\" is more creative, even if odd.\n"
            "- **Top-k (k = 5):** for *\"What is the capital of France?\"* the model samples only from the 5 most likely next words, e.g. Paris, London, Berlin, Rome, Madrid.\n"
            "- **Top-p (p = 0.9):** for *\"The stars danced across the midnight sky…\"* it samples from the words that together hold 90% of the probability, "
            "which may include poetic ones like twinkled, shimmered, waltzed, sighed, sparkled.\n"
            "- **Why it matters:** these methods balance creativity with coherence.")
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

    st.markdown("#### 📋 Example: the same prompt at different temperatures")
    st.caption("Sample outputs written for teaching (not a live call). Prompt: *\"Write a one-line tagline for a neighbourhood chai stall.\"*")
    samples = [
        ("0.0", "Safe and repeatable", ["Freshly brewed chai, served with a smile.", "Freshly brewed chai, served with a smile.", "Freshly brewed chai, served with a smile."]),
        ("0.7", "Balanced", ["Where every cup tastes like home.", "Your corner, your chai, your moment of calm.", "Warm cups, warmer conversations."]),
        ("1.5", "Adventurous", ["Steam, spice and stories: the street's favourite sunrise.", "Sip the monsoon, one cardamom cloud at a time.", "Chai so good, your Monday will file for a transfer."]),
    ]
    cols = st.columns(3)
    for col, (temp, mood, outs) in zip(cols, samples):
        with col:
            with st.container(border=True):
                st.markdown(f"**Temperature {temp}** · {mood}")
                for k, o in enumerate(outs, 1):
                    st.markdown(f"Run {k}: {o}")
    st.caption("Read across the runs: at T = 0 all three runs are identical; at 0.7 they vary but stay on-brand; at 1.5 they surprise you and may drift.")
    st.info("Note: newer Gemini models (such as Gemini 3.x) are tuned to run at their default temperature, and Google recommends leaving it alone. "
            "That is why this page shows saved examples instead of a live temperature test. The idea still matters for many other models and APIs.", icon="ℹ️")
    st.code(
        """from google import genai
from google.genai import types

client = genai.Client(api_key=API_KEY)
resp = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Write a one-line tagline for a chai stall.",
    # Teaching values. On Gemini 3.x leave temperature, top_p, top_k at their defaults.
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

with tabs[7]:
    takeaway([
        "AI ⊃ ML ⊃ DL; Generative AI is built on deep learning and *creates* new content.",
        "Supervised (labels), unsupervised (patterns) and reinforcement (rewards) are the three ways machines learn.",
        "Regression predicts a number, classification a category, clustering finds groups, dimensionality reduction simplifies the data.",
        "Reinforcement learning is model-based (plans with a map of the world) or model-free (learns by trial and error with no map).",
        "An ML project is a loop: collect data → prepare → choose model → train → evaluate → tune → predict, then refresh as the world changes.",
        "Evaluate on data the model has never seen; accuracy on the training data is misleading.",
        "Deep learning loops: predict → measure loss → backpropagate → update weights → repeat. A learning rate that is too high overshoots.",
        "Temperature controls randomness; top-k and top-p limit which tokens can be sampled.",
        "GenAI boosts productivity and creativity but brings bias, IP, privacy and over-reliance risks.",
    ])

    with st.expander("Conclusion from the course notes"):
        st.markdown(
            "1. **AI, ML and DL:** AI is the broad field of making machines mimic human intelligence; ML is a subset focused on learning from data; "
            "DL, a subset of ML, uses neural networks to model complex patterns, especially in unstructured data.\n"
            "2. **GenAI and its evolution:** it creates new text, images and audio by learning patterns from existing data, from rule-based systems (1950s) to transformer models like GPT and DALL·E.\n"
            "3. **Terminology:** temperature controls randomness; top-k and top-p balance creativity and coherence.\n"
            "4. **Benefits:** productivity, creativity, personalisation, education and data augmentation.\n"
            "5. **Challenges:** bias, misinformation, plagiarism, job displacement and security risks such as deepfakes. Responsible use and critical evaluation are essential.")
    st.markdown("**Resources:** [Google AI Studio](https://aistudio.google.com) · [Hugging Face hub](https://huggingface.co)")

with tabs[8]:
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
        {"q": "Grouping customers into segments with no labels given is…",
         "options": ["Regression", "Classification", "Clustering", "Model-free reinforcement learning"],
         "answer": 2, "why": "No labels plus finding natural groups is clustering (unsupervised)."},
        {"q": "Why do we keep a test set that the model never trains on?",
         "options": ["To make training faster", "To measure accuracy on unseen data", "To increase the training data", "To remove bias from the model"],
         "answer": 1, "why": "Accuracy on training data is unrealistically high; unseen data shows how it will really perform."},
        {"q": "A robot learns a maze purely by trial and error and never builds a map. This is…",
         "options": ["Model-based RL", "Model-free RL", "Supervised classification", "Dimensionality reduction"],
         "answer": 1, "why": "Model-free methods learn what to do directly from reward, without a model of the world."},
        {"q": "Which 2017 paper introduced the Transformer?",
         "options": ["Generative Adversarial Nets", "Attention Is All You Need", "BERT", "ELIZA"],
         "answer": 1, "why": "Transformers process whole sequences with attention, capturing long-range context."},
    ])
