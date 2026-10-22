import json
import re

import pandas as pd
import streamlit as st

from core import catalog
from core.codelab import code_lab
from core.diagrams import chain, raw, node
from core.llm import generate, run_live
from core.ui import analogy, cards, definition, diagram, mdtable, page_header, quiz, takeaway, warn

PROJ = catalog.DL / "projects" / "scamguard_ai"

page_header(
    "m10",
    "A complete mini-project: classify text messages as Scam / Not Scam / Uncertain with a clean 3-step pipeline, "
    "swap prompt strategies, validate the output and measure accuracy on a labelled dataset.",
    [
        "Explain the build → execute → parse pipeline and why each step is a separate class",
        "Compare four prompt strategies (ReAct, few-shot, strict JSON, simplified) on the same message",
        "Parse messy model text into JSON and validate it with Pydantic",
        "Compute accuracy, precision, recall and F1 and read a confusion matrix",
        "Run the project locally, in Colab, or as a Streamlit app — without exposing an API key",
    ],
)

STRATS = {
    "ReAct (Thought → Action → Observation)": "react.md",
    "Few-shot (3 worked examples)": "fewshot.md",
    "Strict JSON only": "strict_json.txt",
    "Simplified (quick)": "simplified.txt",
}


def template(fname: str) -> str:
    return (PROJ / "llm" / "prompts" / fname).read_text("utf-8").strip()


def build_prompt(msg: str, fname: str) -> str:
    """Exactly what llm/prompts.py does: template + the user's message."""
    return f"{template(fname)}\n\nUser Message:\n{msg.strip()}"


def extract_json(text: str) -> dict:
    """Same idea as utils.extract_json_from_text: find the first {...} block."""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}


def validate(d: dict):
    """Pydantic validation, as in llm/parser.py. Returns (ok, message)."""
    from pydantic import BaseModel, Field, ValidationError

    class ScamDetectionOutput(BaseModel):
        label: str = Field(...)
        reasoning: str = Field(...)
        intent: str = Field(...)
        risk_factors: list[str] = Field(...)

    try:
        ScamDetectionOutput(**d)
        return True, "valid"
    except ValidationError as e:
        return False, "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())


def metrics(actual, pred):
    """The project's evaluate.py logic, compact: accuracy + per-class precision/recall/F1 (in %)."""
    a = [x.lower().strip() for x in actual]
    p = [x.lower().strip() for x in pred]
    out = {"accuracy": round(100 * sum(x == y for x, y in zip(a, p)) / max(1, len(a)), 2), "classes": {}}
    for lab in sorted(set(a)):
        tp = sum(x == lab and y == lab for x, y in zip(a, p))
        fn = sum(x == lab and y != lab for x, y in zip(a, p))
        fp = sum(x != lab and y == lab for x, y in zip(a, p))
        pr = 100 * tp / (tp + fp) if tp + fp else 0
        rc = 100 * tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        out["classes"][lab] = {"precision": round(pr, 1), "recall": round(rc, 1), "f1": round(f1, 1)}
    return out


@st.cache_data(show_spinner=False)
def dataset() -> pd.DataFrame:
    return pd.read_csv(PROJ / "scam_detection_dataset.csv", encoding="utf-8-sig")


tabs = st.tabs(["📋 Problem statement", "📖 Project description", "🎯 The pipeline", "📝 Prompt strategies", "🟢 Try it live", "🧱 Parser & validator", "📊 Evaluation",
                "🗂️ Code walkthrough", "🔐 Keys & security", "⬇️ Downloads", "🧾 Key takeaways", "✅ Knowledge check"])

t_problem, t_desc, *tabs = tabs
SHOTS = PROJ / "assets" / "screenshots"

# ---------------------------------------------------------------------------
with t_problem:
    st.markdown("### ScamGuard AI: Problem Statement")
    st.markdown("**Overview.** In today's digitally connected world, businesses face a growing threat from sophisticated scam communications that exploit "
                "human psychology, impersonate trusted entities and bypass traditional security filters. These fraudulent messages erode user trust, damage brand "
                "reputation and create costly support and compliance burdens.")
    st.markdown("**ScamGuard AI** uses **Large Language Models (LLMs)** to analyse and classify text messages as **Scam, Not Scam or Uncertain**, while also "
                "identifying the **underlying manipulative intent** and **categorising the type of scam**. Unlike one-time detection tools, it is a **modular, scalable "
                "application** that combines several **prompt-engineering techniques** to support real-time analysis, explainable reasoning and easy feature expansion. "
                "It is both a practical business solution and a hands-on project showing how to build trustworthy, interpretable AI systems.")
    st.markdown("#### Problem context — common scam types")
    mdtable(["Scam type", "What it looks like"], [
        ("**OTP fraud**", "Messages asking you to share a one-time password"),
        ("**Phishing**", "Fake links that lead to malicious websites"),
        ("**Account suspension**", "False claims that a service is suspended and needs immediate action"),
        ("**Reward manipulation**", "Fake prizes and lottery wins"),
        ("**Fear tactics**", "Threats about account closure, SIM deactivation or legal action"),
        ("**Fake authority**", "Impersonating government agencies (RBI, police …)"),
        ("**Loan scams**", "Unsolicited “pre-approved” loan offers"),
        ("**Urgency**", "Messages that create false time pressure"),
    ])
    st.caption("These scams succeed because they use **context-aware, emotionally charged language** that static rules and keyword filters struggle to detect, "
               "and they **evolve faster than blacklists** can adapt — so the system must be adaptable and explainable.")
    st.markdown("#### Why it matters for businesses")
    st.markdown("Banks, telecoms, e-commerce, fintech and government services are prime targets. Scam messages that impersonate their brand cause "
                "**brand-reputation damage**, **higher support load**, **regulatory risk** and **loss of customer trust** — and many organisations lack a scalable, "
                "explainable, up-to-date way to flag them early.")
    st.markdown("#### Educational context — what you practise")
    cards([("Prompting techniques", "Zero-shot and few-shot classification · Chain-of-Thought for intent · ReAct for explainability and tool use (e.g. link safety checks) · dynamic few-shot examples for new scam types"),
           ("System design", "Modular components (classification, intent extraction, scam-type labelling) that evolve independently · structured **output validation** · step-by-step reasoning and **risk scoring**"),
           ("Product & quality", "A real-time interface with Streamlit · evaluating output for **accuracy, interpretability and reliability** · a system learners can extend")], cols=3)
    st.markdown("#### Extend it (stretch goals)")
    mdtable(["Extension", "How"], [
        ("**Multi-language support**", "Add a translation module that detects the language and converts the message to English before the LLM"),
        ("**Threat-database integration**", "Cross-check URLs or message patterns against known scam sources (e.g. RBI alerts, PhishTank) via an API or keyword lookup"),
        ("**Feedback learning loop**", "Capture human corrections of wrong labels (false positives) and feed them back as extra few-shot examples"),
    ])
    st.markdown("#### Technical requirements")
    st.markdown("Python 3.12+ · Google Generative AI (Gemini API) · Streamlit (web UI) · Pandas (data) · python-dotenv (environment) · Pydantic (validation)")
    st.markdown("#### Dataset")
    st.markdown("A labelled set of messages for training and testing. **Columns in the file you downloaded:** `message_text` (the message), `label` (`Scam` / `Not Scam`), "
                "`intent_type` (e.g. Service Reminder, Loan Scam, Account Suspension) and `flag_reason`. "
                "*(The original brief calls the text column `text` — the code accepts `text`, `message_text` or `message`.)*")
    ds = dataset()
    c1, c2 = st.columns([1, 1.3])
    c1.metric("Rows", f"{len(ds):,}")
    c1.dataframe(ds.label.value_counts().rename("rows"), width="stretch")
    c2.dataframe(ds.intent_type.value_counts().head(8).rename("rows"), width="stretch")
    st.caption("Balanced: 500 Scam / 500 Not Scam, so accuracy is meaningful here — but still check recall per class.")
    catalog.dl_button("projects/scamguard_ai/scam_detection_dataset.csv", "⬇️ Download the dataset (CSV)", key="m10_ds_problem")

# ---------------------------------------------------------------------------
with t_desc:
    st.markdown("### 🚨 ScamGuard AI: Protecting Trust with Generative Intelligence")
    st.markdown("ScamGuard AI is a modular, scalable system that detects scam messages using **Google Gemini**, **Streamlit** and prompt engineering. "
                "It classifies a message as **Scam**, **Not Scam** or **Uncertain** and returns **reasoning**, **intent** and **risk factors**.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📌 Features**")
        st.markdown("- **Single message analysis** — instant classification with reasoning\n- **Dataset evaluation** — upload a CSV and analyse many messages\n"
                    "- **Explainable output** — label, reasoning, intent, risk factors (JSON)\n- **Modular architecture** — builder · executor · parser · detector\n"
                    "- **Extensible** — multi-language, threat DB, feedback loops")
    with c2:
        st.markdown("**🛠 Tech stack**")
        st.markdown("- Python 3.10+\n- **Streamlit** — web interface\n- **Gemini API** — LLM reasoning\n- **Pandas** — dataset handling\n- **Custom parser** — raw LLM text → structured JSON")
    st.markdown("**🏗️ Project structure**")
    if (SHOTS / "FolderStructure.png").exists():
        st.image(str(SHOTS / "FolderStructure.png"), width=420)
    st.markdown("**🎥 Demo screenshots (the Streamlit app)**")
    d1, d2 = st.columns(2)
    for col, (fn, cap) in zip((d1, d2), [("Single_scam_message.png", "A scam message → label, intent, reasoning, risk factors"),
                                         ("Single_nonscam_message.png", "A legitimate message → Not Scam")]):
        if (SHOTS / fn).exists():
            col.image(str(SHOTS / fn), caption=cap)
    if (SHOTS / "Multi_message.png").exists():
        st.image(str(SHOTS / "Multi_message.png"), caption="Dataset evaluation tab: many messages at once", width=700)
    st.markdown("**▶ Run it**: see the *Code walkthrough* tab, or download everything in the *Downloads* tab (zip + Colab notebook).")

# ---------------------------------------------------------------------------
with tabs[0]:
    definition("ScamGuard AI", "a small system that reads a message and returns a structured verdict: "
               "`label`, `reasoning`, `intent` and a list of `risk_factors`.")
    analogy("A mail-room clerk: **one person** writes the question card (build), **one** phones the expert (execute) and **one** "
            "writes the answer into the register (parse). If the phone line is bad, only the middle person changes.")
    diagram(chain([("Message\\n“You won an iPhone…”", "grey"), ("1 · Builder\\nprompt template\\n+ message", "blue"),
                   ("2 · Executor\\nLLMClient: Gemini\\nretry + back-off", "orange"), ("3 · Parser\\nfind JSON,\\nfallback = Uncertain", "teal"),
                   ("Result\\nlabel · reasoning\\nintent · risk_factors", "green")]),
            "ScamDetector.detect() calls the three steps in order. detect_batch() repeats it and turns any failure into an “Uncertain” row.")
    st.markdown("#### Why split it into three classes?")
    mdtable(["Step", "File", "Changes when…", "Doesn't care about…"], [
        ("Build", "`pipeline/scam_detector/builder.py`", "you try a new prompt strategy", "which model or API is used"),
        ("Execute", "`pipeline/scam_detector/executor.py` + `llm/client.py`", "you change model, retries or provider", "what the prompt says"),
        ("Parse", "`pipeline/scam_detector/parser.py`", "the output format changes", "how the text was produced"),
    ])
    st.markdown("#### The output contract")
    st.code('''{
  "label": "Scam | Not Scam | Uncertain",
  "reasoning": "<step-by-step analysis>",
  "intent": "<short description of what the sender wants>",
  "risk_factors": ["urgency", "bank details request", "shortened link"]
}''', language="json")
    st.caption("“Uncertain” is deliberate: a classifier that must pick one of two labels will be confidently wrong on ambiguous messages. "
               "Giving the model an exit makes errors visible instead of hidden.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("Four ways to ask the same question. `llm/prompts.py` loads **one** template file and appends the message — "
                "switching strategy is a one-line change: `PROMPT = load_prompt(\"react.md\")`.")
    mdtable(["Strategy", "File", "Idea", "Good for", "Watch out"], [
        ("ReAct", "`react.md`", "Force a Thought → Action → Observation chain, then JSON", "Explainable verdicts, tricky messages", "Longer output, more tokens"),
        ("Few-shot", "`fewshot.md`", "Show 3 solved examples", "Consistent style/labels", "Examples bias the model toward their wording"),
        ("Strict JSON", "`strict_json.txt`", "“Respond with ONLY valid JSON” + schema", "Easy parsing, cheapest to parse", "Little reasoning, so harder to audit"),
        ("Simplified", "`simplified.txt`", "Short checklist + labels", "Fast, cheap first pass", "Less accurate on subtle cases"),
    ])
    pick = st.radio("Show the full prompt file:", list(STRATS), horizontal=True, key="m10_strat_view")
    st.code(template(STRATS[pick]), language="markdown", wrap_lines=True)
    st.info("At run time the project appends: `\\n\\nUser Message:\\n<the message>` — see “Try it live” to view the final prompt that is really sent.")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("Pick strategies, type a message, and compare. Each strategy = **one** call (paced to 4 calls/min, so pick 1–2 for a quick run).")
    examples = {
        "Prize bait": "Congratulations! You've won a free iPhone. Click this link and enter your bank details to claim.",
        "Real bill reminder": "Dear customer, your electricity bill of Rs 840 is due on the 5th. Pay on the official portal.",
        "Ambiguous parcel": "Your package from Amazon is delayed. Track it here: bit.ly/track123",
        "Bank threat": "URGENT: Your account will be suspended! Call 555-FAKE immediately or lose access forever!",
        "(write your own)": "",
    }
    ex = st.selectbox("Example message", list(examples), key="m10_ex")
    msg = st.text_area("Message to analyse", value=examples[ex] or "Hi, this is your bank. Share the OTP sent to your phone to avoid a penalty.",
                       key=f"m10_msg_{ex}", height=90)
    chosen = st.multiselect("Strategies to run", list(STRATS), default=list(STRATS)[:2], key="m10_live_strats")

    def _run():
        outs = []
        for s in chosen:
            txt = generate(build_prompt(msg, STRATS[s]))[0]
            d = extract_json(txt)
            ok, why = validate(d) if d else (False, "no JSON found → parser would return the fallback “Uncertain”")
            outs.append((s, txt, d, ok, why))
        return outs

    res = run_live(f"🔍 Analyse with {len(chosen)} strateg{'y' if len(chosen) == 1 else 'ies'}", _run, key="m10_live") if chosen else None
    if res:
        cols = st.columns(len(res))
        for col, (s, txt, d, ok, why) in zip(cols, res):
            with col.container(border=True):
                st.markdown(f"**{s.split(' (')[0]}**")
                lab = d.get("label", "—")
                st.markdown(f"### {'🚨' if lab == 'Scam' else '✅' if lab == 'Not Scam' else '❓'} {lab}")
                st.caption(d.get("intent", ""))
                if d.get("risk_factors"):
                    st.markdown("Risk factors: " + ", ".join(f"`{r}`" for r in d["risk_factors"]))
                st.markdown(("✅ schema valid" if ok else f"⚠️ {why}"))
                with st.expander("Reasoning / raw reply"):
                    st.write(d.get("reasoning", ""))
                    st.code(txt, language="text", wrap_lines=True)
    st.markdown("##### Or edit and run the whole pipeline yourself")
    code_lab("m10_pipeline", '''\
import json, re
from pydantic import BaseModel, Field, ValidationError

# ---- STEP 1 · BUILD the prompt -------------------------------------------------
TEMPLATE = """You are a safety-focused AI that detects scams in text messages.
Return ONLY JSON with keys: label ("Scam" | "Not Scam" | "Uncertain"), reasoning, intent, risk_factors (list).
Look at: urgency, money or OTP requests, impersonation, suspicious links, emotional pressure.
Be cautious when unsure; do not invent details."""
message = "Your SBI KYC expires today. Update now at http://sbi-kyc-update.xyz or the account will be blocked."
prompt = f"{TEMPLATE}\\n\\nUser Message:\\n{message}"

# ---- STEP 2 · EXECUTE ----------------------------------------------------------
raw = generate(prompt)[0]
print("RAW MODEL REPLY:\\n", raw, "\\n")

# ---- STEP 3 · PARSE + VALIDATE -------------------------------------------------
class Out(BaseModel):
    label: str
    reasoning: str
    intent: str
    risk_factors: list[str]

m = re.search(r"\\{.*\\}", raw, re.DOTALL)
try:
    result = Out(**json.loads(m.group()))
    print("VALID ->", result.label, "|", result.risk_factors)
except (AttributeError, json.JSONDecodeError, ValidationError) as e:
    print("FALLBACK -> Uncertain  (", type(e).__name__, ")")''', title="build → execute → parse", height=470)

# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("Models don't always return clean JSON. Two defences, both in the project:")
    cards([("`extract_json_from_text`", "Finds the first `{ … }` even when the model wraps it in prose or a ```json fence. Returns `{}` if there is none; "
                                        "the parser then returns the **fallback**: `label = Uncertain`."),
           ("`validate_output` (Pydantic)", "Checks that every key exists and has the right type. A missing `intent` or a string where a list belongs "
                                            "raises a clear `ValueError` instead of a silent bug later.")], cols=2)
    st.markdown("##### Try it — no API key needed")
    code_lab("m10_parse", '''\
import json, re
from pydantic import BaseModel, ValidationError

class ScamDetectionOutput(BaseModel):
    label: str
    reasoning: str
    intent: str
    risk_factors: list[str]

def extract_json(text):
    try:
        m = re.search(r"\\{.*\\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}

REPLIES = {
    "clean":        '{"label":"Scam","reasoning":"asks for OTP","intent":"steal OTP","risk_factors":["OTP request"]}',
    "wrapped":      'Sure! Here is the analysis:\\n```json\\n{"label":"Not Scam","reasoning":"bill","intent":"remind","risk_factors":[]}\\n```',
    "missing key":  '{"label":"Scam","reasoning":"link","risk_factors":["short url"]}',
    "wrong type":   '{"label":"Scam","reasoning":"x","intent":"y","risk_factors":"urgency"}',
    "no json":      "This looks suspicious to me, I would not click it.",
    "trailing comma": '{"label":"Scam","reasoning":"x","intent":"y","risk_factors":["a",],}',
}
for name, reply in REPLIES.items():
    d = extract_json(reply)
    if not d:
        print(f"{name:15} -> FALLBACK Uncertain (no parsable JSON)")
        continue
    try:
        ok = ScamDetectionOutput(**d)
        print(f"{name:15} -> OK       {ok.label}")
    except ValidationError as e:
        print(f"{name:15} -> INVALID  {e.errors()[0]['loc'][0]}: {e.errors()[0]['msg']}")''', title="parse and validate six messy replies", needs_key=False, height=520)
    st.caption("Add your own broken reply to `REPLIES` and see which defence catches it. A more robust fix for production: ask the API for JSON mode / a response schema, "
               "and retry once with the validation error appended to the prompt.")

# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("**Accuracy alone can mislead.** If 95 % of messages are safe, a bot that always says “Not Scam” scores 95 % and catches zero scams. "
                "So the project also reports per-class **recall** and **F1**.")
    mdtable(["Metric", "Question it answers", "Formula"], [
        ("Accuracy", "How often is the label right overall?", "correct ÷ total"),
        ("Precision (Scam)", "When it says Scam, is it right?", "TP ÷ (TP + FP)"),
        ("Recall (Scam)", "Of all real scams, how many did it catch?", "TP ÷ (TP + FN)"),
        ("F1", "One number balancing precision and recall", "2·P·R ÷ (P + R)"),
    ])
    st.markdown("##### Offline: score your own predictions")
    code_lab("m10_metrics", '''\
actual    = ["Scam","Scam","Scam","Scam","Not Scam","Not Scam","Not Scam","Not Scam","Not Scam","Not Scam"]
predicted = ["Scam","Scam","Not Scam","Uncertain","Not Scam","Not Scam","Scam","Not Scam","Not Scam","Not Scam"]
# try: make the bot always answer "Not Scam" and compare accuracy with recall

correct = sum(a == p for a, p in zip(actual, predicted))
print(f"accuracy: {100*correct/len(actual):.0f}%   ({correct}/{len(actual)})")
for label in sorted(set(actual)):
    tp = sum(a == label and p == label for a, p in zip(actual, predicted))
    fn = sum(a == label and p != label for a, p in zip(actual, predicted))
    fp = sum(a != label and p == label for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0
    recall    = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print(f"{label:9} precision {precision:.0%}  recall {recall:.0%}  F1 {f1:.0%}   (TP={tp} FN={fn} FP={fp})")''',
             title="accuracy, recall, F1", needs_key=False, height=330)

    st.markdown("##### 🟢 Live: evaluate on real rows of the course dataset")
    df = dataset()
    st.caption(f"The course file has {len(df):,} labelled messages ({df.label.value_counts().to_dict()}). To stay inside the free quota this demo sends "
               "**one batched prompt** (the Module 6 trick) instead of one call per message as `evaluate.py` does.")
    n = st.slider("How many messages", 6, 30, 12, 2, key="m10_n")
    seed = st.number_input("Random seed", 0, 999, 7, key="m10_seed")
    sample = pd.concat([df[df.label == "Scam"].sample(n // 2, random_state=int(seed)),
                        df[df.label == "Not Scam"].sample(n - n // 2, random_state=int(seed))]).sample(frac=1, random_state=int(seed)).reset_index(drop=True)

    def _eval():
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(sample.message_text))
        p = (template("simplified.txt") + "\n\nClassify EACH numbered message. Return ONLY a JSON list like "
             '[{"id": 1, "label": "Scam"}, {"id": 2, "label": "Not Scam"}] using only the labels Scam, Not Scam or Uncertain.\n\n' + numbered)
        txt = generate(p)[0]
        m = re.search(r"\[.*\]", txt, re.DOTALL)
        data = json.loads(m.group()) if m else []
        lab = {d["id"]: d["label"] for d in data if isinstance(d, dict) and "id" in d}
        return [lab.get(i + 1, "Uncertain") for i in range(len(sample))]

    preds = run_live(f"▶ Classify {n} messages in one call", _eval, key="m10_eval")
    if preds:
        out = sample.assign(predicted=preds)
        out["✓"] = (out.label.str.lower() == out.predicted.str.lower()).map({True: "✅", False: "❌"})
        m = metrics(out.label.tolist(), out.predicted.tolist())
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']}%")
        for c, (lab, v) in zip((c2, c3), m["classes"].items()):
            c.metric(f"{lab} · recall / F1", f"{v['recall']}% / {v['f1']}%")
        st.markdown("**Confusion matrix** (rows = truth, columns = prediction)")
        st.dataframe(pd.crosstab(out.label, out.predicted), width="content")
        st.dataframe(out[["message_text", "label", "predicted", "✓"]], hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("Every file of the project, with what it does. Pick one to read the code.")
    files = sorted(catalog.SCAM_FILES, key=lambda f: (f.count("/"), f))
    code_files = [f for f in files if not f.endswith((".csv", "LICENSE"))]
    mdtable(["File", "Purpose"], [(f"`{f}`", catalog.SCAM_FILES[f]) for f in files])
    sel = st.selectbox("Read a file", code_files, index=code_files.index("pipeline/scam_detector/detector.py"), key="m10_file")
    lang = {"py": "python", "md": "markdown", "txt": "text"}.get(sel.rsplit(".", 1)[-1], "text")
    st.code((PROJ / sel).read_text("utf-8"), language=lang, wrap_lines=True)
    st.markdown("#### Run it yourself")
    st.code("""# 1) unzip, then
cd scamguard_ai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env                                    # then edit .env and paste YOUR key
python main.py                                          # one message
python evaluate.py test_scam_dataset.csv --verbose      # tiny test set
python evaluate.py scam_detection_dataset.csv --limit 30 --batch-size 5
streamlit run streamlit/app.py                          # browser UI""", language="bash")

# ---------------------------------------------------------------------------
with tabs[6]:
    warn("**Security finding in the uploaded project.** `config.py` contained a real Gemini API key typed straight into the source "
         "(`GEMINI_API_KEY = \"AIza…\"`). Anyone who sees that file — including on GitHub — can use your quota. "
         "**Delete that key in Google AI Studio and create a new one.** The copy offered here reads the key from the environment instead.")
    st.markdown("##### Before → after")
    c1, c2 = st.columns(2)
    c1.markdown("❌ **Before** (never do this)")
    c1.code('GEMINI_API_KEY = "AIzaSy…paste-your-key-here…"', language="python")
    c2.markdown("✅ **After** (what the download uses)")
    c2.code('GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")', language="python")
    st.markdown("Full guide to getting and storing keys (local `.env`, Streamlit Secrets, Colab Secrets): last page **⬇️ Downloads & API keys**.")

# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("All files are **key-free**. Add your own key after downloading.")
    c1, c2, c3 = st.columns(3)
    with c1:
        catalog.zip_button("⬇️ Whole project (.zip)", "projects/scamguard_ai", "scamguard_ai.zip", key="m10_zip", use_container_width=True)
    with c2:
        catalog.dl_button("notebooks/P1_ScamGuard_AI.ipynb", "⬇️ Colab notebook (.ipynb)", key="m10_nb", use_container_width=True)
    with c3:
        st.link_button("▶ Open in Colab", catalog.colab_url("notebooks/P1_ScamGuard_AI.ipynb"), use_container_width=True)
    st.caption("“Open in Colab” works once the notebook is on your GitHub repo (public). Otherwise download the .ipynb and use Colab → File → Upload notebook.")
    st.markdown("**Datasets**")
    c1, c2 = st.columns(2)
    with c1:
        catalog.dl_button("projects/scamguard_ai/scam_detection_dataset.csv", "⬇️ scam_detection_dataset.csv (1,000 rows)", key="m10_ds1", use_container_width=True)
    with c2:
        catalog.dl_button("projects/scamguard_ai/test_scam_dataset.csv", "⬇️ test_scam_dataset.csv (7 rows)", key="m10_ds2", use_container_width=True)

# ---------------------------------------------------------------------------
with tabs[8]:
    takeaway([
        "Split the pipeline into build → execute → parse so each part can change on its own.",
        "Prompt strategy is a one-line switch: compare ReAct, few-shot, strict JSON and simplified on the same data.",
        "Give the model an “Uncertain” exit and a parser fallback so failures are visible, not silent.",
        "Validate model output with Pydantic; never trust free text.",
        "Judge a classifier with recall and F1 per class, not just accuracy.",
        "Keys belong in the environment (.env / Secrets), never in source code.",
    ])

with tabs[9]:
    quiz("m10", [
        {"q": "Why is `Uncertain` a valid label?", "options": ["To make the dataset bigger", "So ambiguous messages aren't forced into a confidently wrong answer",
         "Gemini requires three labels", "It speeds up the call"], "answer": 1, "why": "An exit option makes low-confidence cases visible."},
        {"q": "The parser finds no JSON in the reply. What should the pipeline do?", "options": ["Crash", "Return a fallback result labelled Uncertain",
         "Guess Scam", "Call the model forever"], "answer": 1, "why": "Fail safe and keep the batch running."},
        {"q": "A bot always answers 'Not Scam' on a 95 % safe dataset. Which statement is true?", "options": ["Accuracy 95 %, Scam recall 0 %", "Accuracy 0 %",
         "Recall 95 %", "F1 100 %"], "answer": 0, "why": "High accuracy can hide zero recall on the rare class."},
        {"q": "Which file do you edit to switch from ReAct to few-shot?", "options": ["executor.py", "llm/prompts.py (the PROMPT line)", "evaluate.py", "utils.py"],
         "answer": 1, "why": "`PROMPT = load_prompt(\"fewshot.md\")`."},
        {"q": "Where should the Gemini key live?", "options": ["In config.py", "In a .env file / Secrets that is git-ignored", "In the README", "In the dataset"],
         "answer": 1, "why": "Environment variables keep secrets out of source control."},
    ])

catalog.module_downloads("m10", "⬇️ More downloads for this project")
