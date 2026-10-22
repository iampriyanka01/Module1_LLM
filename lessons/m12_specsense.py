import importlib.util
import json
import re

import pandas as pd
import streamlit as st

from core import catalog
from core.codelab import code_lab
from core.diagrams import chain
from core.llm import generate, run_live
from core.ui import analogy, cards, definition, diagram, mdtable, page_header, quiz, takeaway, warn

PROJ = catalog.DL / "projects" / "specsense_ai"
SHOTS = PROJ / "assets" / "screenshots"

page_header(
    "m12",
    "A recommendation assistant: turn “lightweight laptop under ₹30k for coding” into an explainable match — and learn why it must be grounded in a real catalog.",
    [
        "Describe the SpecSense problem: free-text needs → hard specs + soft preferences → explainable recommendation",
        "Read the ReAct prompt and the build → execute → parse pipeline",
        "Show how recommending from model memory invents products and reviews",
        "Ground the answer in a company catalog: extract → match (pandas) → explain",
        "Detect gaps when no product fits and suggest the nearest alternatives",
        "Evaluate a recommendation (match quality, rationale correctness, explainability)",
    ],
)


@st.cache_resource(show_spinner=False)
def matcher():
    """Load catalog_matcher.py from the project folder (its pure-pandas helpers need no API key)."""
    spec = importlib.util.spec_from_file_location("specsense_catalog_matcher", PROJ / "catalog_matcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = matcher()
CAT = M.load_catalog()
REACT = (PROJ / "llm" / "prompts" / "react.md").read_text("utf-8").strip()


def extract_json(text: str) -> dict:
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}


def show_result(r: dict):
    st.success(r.get("recommended_product", "Unknown"))
    for title, icon, key in [("Meets", "✅", "meets"), ("Partially meets", "⚠️", "partially_meets"), ("Trade-offs", "❌", "trade_offs"),
                             ("Customer review", "💬", "review_snippets"), ("Alternatives", "🔄", "alternatives")]:
        items = r.get(key) or []
        if items:
            st.markdown(f"**{title}**")
            for it in items:
                st.write(f"{icon} {it}")
    if r.get("gap_note"):
        st.warning(f"Gap: {r['gap_note']}")


tabs = st.tabs(["📋 Problem statement", "📖 Project description", "🎯 Pipeline & prompt", "🟢 Try the original", "🧱 Grounded in a catalog",
                "📊 Evaluate it", "🗂️ Code walkthrough", "⬇️ Downloads", "🧾 Key takeaways", "✅ Knowledge check"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### SpecSense AI: Problem Statement")
    st.markdown("**Overview.** In digital commerce and service ecosystems, matching users to the right product drives engagement and retention. Traditional recommenders "
                "rely on rigid filters or past behaviour and often miss the **real-time context and intent** behind a request.")
    st.markdown("**SpecSense AI** uses **Generative AI** to process a user's natural-language preferences and match them to the most suitable product(s) in a "
                "**company-provided catalog**. Instead of overwhelming users with irrelevant choices, it gives **context-aware, explainable recommendations** backed by "
                "structured product features and customer-profile metadata. It is valuable for enterprises that already have **clean product and customer data** and want "
                "personalisation without redesigning their platform.")
    st.markdown("It builds a Generative-AI system that:")
    st.markdown("- understands a user's **free-form** product query,\n- matches it with **structured** company product data,\n- generates **explainable, insight-driven** recommendations.")
    st.markdown("#### Problem context — what most recommenders struggle with")
    mdtable(["Need", "Example"], [
        ("**Flexible input**", "“I want something light, with great battery and under ₹30k”"),
        ("**Transparent recommendations**", "Why a product was recommended is not made clear"),
        ("**Hybrid filtering**", "Combine hard specs with soft preferences (“good in low light”, “eco-friendly”)"),
    ])
    st.caption("Companies have the **data** but lack the **intelligence layer** to connect the dots with natural-language understanding, real-time context and personalised matching.")
    st.markdown("**Consequences:** customer indecision or drop-off · frustration from irrelevant picks · under-use of product diversity.")
    st.markdown("#### Why it matters for enterprises")
    st.markdown("- Human-like, conversational product matching\n- Faster, more trusted product discovery\n- Vague intent converted into concrete decisions\n- Lower cognitive load for customers")
    st.markdown("#### Data")
    mdtable(["Input", "Contents"], [("**Product catalog**", "Structured CSV/JSON: name, specs, category, description"),
                                    ("**User profile**", "Age, budget, context tags, product history")])
    st.markdown("#### What SpecSense AI does (7 capabilities)")
    st.markdown("""
1. **Understands** free-form preference queries (“lightweight laptop under ₹30k for travel and battery life”)
2. **Extracts** structured constraints (budget, weight, purpose) and soft preferences (long battery, travel-friendly)
3. **Matches** user intent with structured catalog data
4. **Generates explainable recommendations** — which criteria were met, where trade-offs exist, which soft preferences were only partly met
5. **Highlights gaps** if no perfect match exists and suggests the closest alternatives with rationale
6. **Enriches** results with real-world insight: review snippets, availability, current offers
7. **Evaluates** recommendation quality, explainability and user satisfaction for continuous improvement
""")
    st.markdown("#### User flow")
    diagram(chain([("1 · Query\\n(natural language)", "grey"), ("2 · Intent &\\nconstraint extraction", "blue"), ("3 · Matching engine\\nhard + soft", "orange"),
                   ("4 · Recommendation +\\ntransparency layer", "green"), ("5 · Gap detection\\n(if needed)", "yellow"), ("6 · Enrichment\\nreviews · offers", "teal"),
                   ("7 · Evaluation\\nlayer", "purple")]), "The seven steps of the user flow.", max_w=10, max_h=3.2)
    mdtable(["Step", "Details"], [
        ("2 · Intent & constraint extraction", "**Hard specs:** budget, weight, use-case keywords · **Soft needs:** “good for coding”, “battery backup”"),
        ("3 · Matching engine", "Match structured data (RAM, battery life) · **soft match** with the LLM (associate “good for coding” with CPU + RAM + reviews)"),
        ("4 · Transparency layer", "For each product show: **hard matches · soft matches · trade-offs · unmet needs**"),
        ("5 · Gap detection", "“No products meet all criteria. Showing best alternatives…”"),
        ("6 · Enrichment", "Customer review snippets; availability/offers when the data has them"),
        ("7 · Evaluation", "Track quality of match, correctness of the rationale and explainability"),
    ])
    st.markdown("#### Technical requirements")
    st.markdown("Python 3.12+ · Gemini API (or OpenAI, optional) · Pandas · Streamlit · python-dotenv · Pydantic")
    st.markdown("**By solving a real consumer problem you learn to:** design practical LLM pipelines · apply advanced prompting · evaluate outputs for interpretability and "
                "usability · build scalable, modular systems.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### 🔍 SpecSense AI — Intelligent Product Recommendations")
    st.markdown("A **free-text shopping query** goes in; an **explainable recommendation** comes out (what is met, partly met, trade-offs, a review snippet, alternatives). "
                "Built with **Gemini**, **Streamlit** and ReAct prompting.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📌 Features**")
        st.markdown("- Single query analysis → structured JSON\n- Explainable output (`meets`, `partially_meets`, `trade_offs`, `review_snippets`, `alternatives`)\n"
                    "- Modular pipeline: builder → executor → parser → `SpecSensor`\n- **Catalog grounding** (`catalog_matcher.py`, added here)")
    with c2:
        st.markdown("**🛠 Stack**")
        st.markdown("- Python · **Gemini API** (retry + back-off)\n- **Streamlit** UI\n- **Pandas** for the catalog\n- Custom parser: raw text → JSON")
    st.markdown("**Output example**")
    st.code(json.dumps({"recommended_product": "…", "meets": ["Strict budget under ₹30,000", "Lightweight (≈1.4 kg)"], "partially_meets": ["CPU fine for light coding"],
                        "trade_offs": ["RAM is soldered"], "review_snippets": ["Runs VS Code without lag"], "alternatives": ["…"]}, indent=2, ensure_ascii=False), language="json")
    st.markdown("**Screenshots from the deployed app**")
    if (SHOTS / "Laptop_Recommendation_streamlit.png").exists():
        st.image(str(SHOTS / "Laptop_Recommendation_streamlit.png"), caption="Streamlit output for “affordable lightweight laptop under ₹30k for coding students”", width=760)
    if (SHOTS / "FolderStructure.png").exists():
        st.image(str(SHOTS / "FolderStructure.png"), caption="Project structure", width=360)
    st.info("**What we changed in the shipped copy:** fixed the batch-error fallback (it used ScamGuard's fields), a log line that read a non-existent `label`, "
            "a debug `print`, an over-chatty log of the full response, and the README (which described a different project). Added `catalog_matcher.py` and a 12-product sample catalog.")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("Same three-step shape as ScamGuard — only the prompt and the output schema change. That is the point of the modular design.")
    diagram(chain([("User query", "grey"), ("1 · Builder\\nReAct template + query", "blue"), ("2 · Executor\\nLLMClient (Gemini)", "orange"),
                   ("3 · Parser\\nfind JSON → schema", "teal"), ("Recommendation", "green")]))
    mdtable(["ScamGuard", "SpecSense", "Changed?"], [
        ("`pipeline/scam_detector/*`", "`pipeline/spec_sensor/*`", "folder + class name (`ScamDetector` → `SpecSensor`)"),
        ("prompt: classify a message", "prompt: extract constraints and match products", "**yes — `react.md`**"),
        ("output: label · reasoning · intent · risk_factors", "output: recommended_product · meets · partially_meets · trade_offs · review_snippets · alternatives", "**yes — parser fields**"),
        ("`llm/client.py` (retry + back-off)", "identical", "no"),
    ])
    st.markdown("##### The prompt (`llm/prompts/react.md`)")
    st.code(REACT, language="markdown", wrap_lines=True)
    st.caption("`llm/prompts.py` appends `\\n\\nUser Message:\\n<the query>`. Notice what is **missing**: there is no product list anywhere in this prompt.")
    warn("The prompt tells the model to “match products against these requirements” and “include review snippets” — but never gives it any products or reviews. "
         "The model can only answer from memory. See the next tabs.")

# ---------------------------------------------------------------------------
with tabs[3]:
    st.markdown("Run the **original** pipeline: ReAct prompt + your query → Gemini → parsed JSON. Then we check the answer against a real catalog.")
    q = st.text_area("Product query", "Find an affordable, lightweight laptop under rupees 30k suitable for coding and programming students.", key="m12_q1", height=80)

    def _orig():
        txt = generate(f"{REACT}\n\nUser Message:\n{q.strip()}")[0]
        return txt, extract_json(txt)

    res = run_live("🔍 Recommend (original pipeline)", _orig, key="m12_orig")
    if res:
        txt, d = res
        if d:
            show_result(d)
            name = str(d.get("recommended_product", ""))
            in_cat = any(name.lower() == n.lower() or n.lower() in name.lower() for n in CAT["name"])
            if in_cat:
                st.success("✅ The recommended product exists in the sample catalog.")
            else:
                st.error(f"❌ **“{name}” is not in the company catalog.** It came from the model's memory — the specs, price and review quotes above are unverified "
                         "and may be invented. A shop can't sell what isn't in its catalog.")
        else:
            st.warning("No JSON found — the parser would return the fallback `Unknown` result.")
            st.code(txt, language="text", wrap_lines=True)
    st.markdown("##### Or edit and run the three steps yourself")
    code_lab("m12_orig_code", '''\
import json, re
REACT = """You are a transparency-focused AI that recommends products matching hard specifications and soft preferences.
1. Thought: analyse budget, category and preferences.
2. Action: extract HARD constraints (price, RAM, weight) and SOFT preferences (portability, battery).
3. Observation: match products and note where they fit or fall short.
4. Final Answer: ONLY JSON: {"recommended_product": str, "meets": [str], "partially_meets": [str], "trade_offs": [str],
   "review_snippets": [str], "alternatives": [str]}
Do not output Thought/Action/Observation. If no perfect match, suggest the nearest alternatives."""
query = "Need a lightweight laptop under 30000 rupees for coding with good battery life"

raw = generate(f"{REACT}\\n\\nUser Message:\\n{query}")[0]
m = re.search(r"\\{.*\\}", raw, re.DOTALL)
data = json.loads(m.group()) if m else {}
print("Recommended:", data.get("recommended_product"))
print("Review snippets the model 'found':", data.get("review_snippets"))
print("\\nQuestion: where did those reviews come from? (no catalog was given to the model)")''', title="original pipeline in 15 lines", height=420)

# ---------------------------------------------------------------------------
with tabs[4]:
    definition("Grounding", "making the model answer **from your data** instead of its memory: retrieve the right rows first, then let the LLM explain **only** those rows.")
    analogy("A shop assistant who **looks at the shelf** (catalog) before answering, versus one who answers from what they vaguely remember about laptops in general.")
    diagram(chain([("Query", "grey"), ("1 · EXTRACT\\nLLM → hard constraints\\n+ soft preferences", "blue"), ("2 · MATCH\\npandas filters the catalog\\n(exact, testable)", "green"),
                   ("Gap?\\nnearest alternatives", "yellow"), ("3 · EXPLAIN\\nLLM uses ONLY\\nthe retrieved rows", "orange"), ("Recommendation\\n+ gap note", "teal")]), max_w=10, max_h=3.2)
    st.markdown("##### The catalog (12 fictional laptops — replace with your company's data)")
    st.dataframe(CAT.drop(columns=["description"]), hide_index=True, width="stretch")
    st.markdown("##### Step 2 offline: hard-constraint matching and gap detection (no API key)")
    st.caption("Edit the constraints. The matcher is plain pandas, so it is exact and unit-testable — the LLM is not involved.")
    c1, c2, c3 = st.columns(3)
    max_price = c1.number_input("Max price (₹)", 0, 200000, 30000, 1000, key="m12_p")
    max_w = c2.number_input("Max weight (kg, 0 = any)", 0.0, 5.0, 1.5, 0.1, key="m12_w")
    min_ram = c3.number_input("Min RAM (GB, 0 = any)", 0, 64, 0, 4, key="m12_r")
    c4, c5 = st.columns(2)
    min_bat = c4.number_input("Min battery (hours, 0 = any)", 0, 24, 8, 1, key="m12_b")
    min_sto = c5.number_input("Min storage (GB, 0 = any)", 0, 2048, 0, 128, key="m12_s")
    cons = {"max_price_inr": max_price or None, "max_weight_kg": max_w or None, "min_ram_gb": min_ram or None,
            "min_battery_hours": min_bat or None, "min_storage_gb": min_sto or None}
    rows, exact = M.match(CAT, cons)
    if exact:
        st.success(f"✅ {len(rows)} in-stock product(s) meet **every** hard constraint (cheapest first).")
    else:
        st.warning("⚠️ **No product meets all criteria.** Showing the nearest alternatives (fewest broken constraints, then smallest shortfall).")
    show = rows[["name", "price_inr", "weight_kg", "ram_gb", "storage_gb", "battery_hours", "violations"]].copy()
    show["violations"] = show["violations"].map(lambda v: "; ".join(v) if v else "—")
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Try: price 30000 + RAM 16 → no exact match (gap detection). Out-of-stock items (L05, L12) are never offered.")

    st.markdown("##### 🟢 Full grounded pipeline (3 calls → 2 here: extract, explain)")
    q2 = st.text_area("Query", "Lightweight laptop under ₹30k for coding students, good battery backup", key="m12_q2", height=70)

    def _grounded():
        c = extract_json(generate(M.EXTRACT_PROMPT + q2)[0]) or {}
        r, ex = M.match(CAT, c)
        payload = r.drop(columns=["n_viol", "shortfall"]).to_dict(orient="records")
        prompt = (f"{M.EXPLAIN_PROMPT}\nUser query: {q2}\nExtracted constraints: {json.dumps(c)}\nExact match found: {ex}\n"
                  f"Catalog rows (the ONLY products you may mention):\n{json.dumps(payload, ensure_ascii=False)}")
        return c, ex, r["name"].tolist(), extract_json(generate(prompt)[0])

    g = run_live("🧱 Recommend from the catalog", _grounded, key="m12_grounded")
    if g:
        c, ex, cands, d = g
        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.markdown("**Extracted constraints**")
            st.json(c)
            st.caption(f"Exact match: {ex} · candidates sent to the model: {', '.join(cands)}")
        with c2:
            show_result(d)
            name = str(d.get("recommended_product", ""))
            if any(name.lower() == n.lower() for n in CAT["name"]):
                st.success("✅ Product is a real catalog row.")
            else:
                st.error("❌ Name not found in the catalog — flag for review.")
    st.markdown("##### Edit the matcher")
    code_lab("m12_matcher", '''\
import pandas as pd
cat = pd.DataFrame([
    {"name": "Nimbus Air 14 Lite", "price_inr": 27990, "weight_kg": 1.30, "ram_gb": 8,  "battery_hours": 9},
    {"name": "Vertex Student 15",  "price_inr": 24990, "weight_kg": 1.75, "ram_gb": 8,  "battery_hours": 6},
    {"name": "Orbit Go 13",        "price_inr": 29490, "weight_kg": 1.10, "ram_gb": 8,  "battery_hours": 11},
    {"name": "Nimbus Air 14 Plus", "price_inr": 32990, "weight_kg": 1.30, "ram_gb": 16, "battery_hours": 10},
])
want = {"price_inr": ("<=", 30000), "weight_kg": ("<=", 1.5), "battery_hours": (">=", 8)}   # <- change me

def broken(row):
    return [f"{c} {row[c]} (wanted {op} {v})" for c, (op, v) in want.items() if (row[c] > v if op == "<=" else row[c] < v)]

cat["broken"] = cat.apply(broken, axis=1)
cat["n_broken"] = cat["broken"].map(len)
print(cat.sort_values(["n_broken", "price_inr"])[["name", "price_inr", "n_broken", "broken"]].to_string(index=False))
print("\\nEXACT MATCHES:", cat[cat.n_broken == 0].name.tolist() or "none -> show nearest + a gap note")''',
             title="hard-constraint filter + gap detection", needs_key=False, height=470)

# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("The problem statement's last layer: **track quality of match, correctness of rationale, and explainability.** Some of this needs no LLM at all.")
    mdtable(["Criterion", "Question", "How to check"], [
        ("**Quality of match**", "Does the product really satisfy the hard constraints?", "**Code:** compare price/weight/RAM from the catalog row with the query's limits"),
        ("**Correctness of rationale**", "Are the claims in `meets` and the review quote true?", "**Code:** the product exists; the review text is a substring of the catalog's review"),
        ("**Explainability**", "Does it list meets / partial / trade-offs / alternatives clearly?", "**Code** for presence of fields; **LLM judge** or humans for clarity"),
        ("**User satisfaction**", "Would the shopper accept it?", "Human ratings, click-through, returns (production)"),
    ])
    st.markdown("##### Offline: automatic checks (no API key)")
    code_lab("m12_eval", '''\
import pandas as pd
catalog = pd.DataFrame([
    {"name": "Nimbus Air 14 Lite", "price_inr": 27990, "weight_kg": 1.30, "review": "Light enough to carry all day and VS Code runs fine, but the screen could be sharper."},
    {"name": "Orbit Go 13", "price_inr": 29490, "weight_kg": 1.10, "review": "Barely notice it in my bag and the battery goes on forever; only 256 GB storage."},
])
limits = {"price_inr": 30000, "weight_kg": 1.5}

answers = {   # try editing: make the model 'hallucinate'
    "grounded":     {"recommended_product": "Orbit Go 13", "review_snippets": ["Barely notice it in my bag and the battery goes on forever"],
                     "meets": ["Price under 30000", "Lightweight"], "partially_meets": ["Storage"], "trade_offs": ["256 GB"], "alternatives": ["Nimbus Air 14 Lite"]},
    "hallucinated": {"recommended_product": "Lenovo IdeaPad Slim 1", "review_snippets": ["Great budget laptop for CS students."],
                     "meets": ["Price under 30000"], "partially_meets": [], "trade_offs": [], "alternatives": []},
}
for label, a in answers.items():
    row = catalog[catalog.name == a["recommended_product"]]
    exists = len(row) == 1
    price_ok = exists and row.price_inr.iloc[0] <= limits["price_inr"] and row.weight_kg.iloc[0] <= limits["weight_kg"]
    review_ok = exists and all(s.lower() in row.review.iloc[0].lower() for s in a["review_snippets"])
    fields = sum(bool(a[k]) for k in ["meets", "partially_meets", "trade_offs", "review_snippets", "alternatives"])
    print(f"{label:13} product in catalog: {exists!s:5} | hard constraints met: {price_ok!s:5} | review quote verbatim: {review_ok!s:5} | explainability fields filled: {fields}/5")''',
             title="three automatic checks", needs_key=False, height=520)
    st.markdown("##### 🟢 LLM-as-judge for the parts code can't judge")
    q3 = "Lightweight laptop under 30000 for coding, good battery"
    ans = st.text_area("Recommendation JSON to judge (paste one from the earlier tabs, or keep this example)",
                       json.dumps({"recommended_product": "Orbit Go 13", "meets": ["Price ₹29,490 under ₹30,000", "1.1 kg lightweight", "11 h battery"],
                                   "partially_meets": ["Only 256 GB storage for coding projects"], "trade_offs": ["Core i3-class CPU is modest for heavy builds"],
                                   "review_snippets": ["Barely notice it in my bag and the battery goes on forever"], "alternatives": ["Nimbus Air 14 Lite — slightly heavier, cheaper"]},
                                  indent=1, ensure_ascii=False), height=210, key="m12_judge_in")

    def _judge():
        p = ("You are a strict, independent evaluator of product recommendations. You did NOT make this one.\n"
             f"User query: {q3}\nRecommendation JSON:\n{ans}\n\nScore 1-5 each and be critical: match_quality, rationale_correctness, explainability. "
             'Return ONLY JSON: {"match_quality": n, "rationale_correctness": n, "explainability": n, "feedback": "one sentence"}')
        return extract_json(generate(p)[0])

    j = run_live("🧑‍⚖️ Judge this recommendation", _judge, key="m12_judge")
    if j:
        c = st.columns(3)
        for col, k in zip(c, ["match_quality", "rationale_correctness", "explainability"]):
            col.metric(k.replace("_", " ").title(), f"{j.get(k, '—')} / 5")
        st.info(j.get("feedback", ""))

# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown("Every file of the project, with its purpose.")
    FILES = catalog.SPEC_FILES
    mdtable(["File", "Purpose"], [(f"`{f}`", d) for f, d in FILES.items()])
    readable = [f for f in FILES if f.endswith((".py", ".md", ".csv"))]
    sel = st.selectbox("Read a file", readable, index=readable.index("pipeline/spec_sensor/sensor.py"), key="m12_file")
    lang = {"py": "python", "md": "markdown", "csv": "text"}[sel.rsplit(".", 1)[-1]]
    st.code((PROJ / sel).read_text("utf-8"), language=lang, wrap_lines=True)
    st.code("""cd specsense_ai
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env                                      # paste YOUR key into .env
python main.py                                            # original pipeline
python catalog_matcher.py "Lightweight laptop under 30000 for coding"
streamlit run streamlit/app.py""", language="bash")

# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("All files are **key-free**. Add your own key after downloading (see the last page, *Downloads & API keys*).")
    c1, c2, c3 = st.columns(3)
    with c1:
        catalog.zip_button("⬇️ Whole project (.zip)", "projects/specsense_ai", "specsense_ai.zip", key="m12_zip", use_container_width=True)
    with c2:
        catalog.dl_button("notebooks/P3_SpecSense_AI.ipynb", "⬇️ Colab notebook (.ipynb)", key="m12_nb", use_container_width=True)
    with c3:
        st.link_button("▶ Open in Colab", catalog.colab_url("notebooks/P3_SpecSense_AI.ipynb"), use_container_width=True)
    catalog.dl_button("projects/specsense_ai/data/products_sample.csv", "⬇️ Sample product catalog (CSV)", key="m12_csv")
    st.caption("“Open in Colab” works once the repo is on GitHub (public). Otherwise download the .ipynb → colab.research.google.com → File ▸ Upload notebook.")

with tabs[8]:
    takeaway([
        "SpecSense = free-text need → hard specs + soft preferences → explainable recommendation.",
        "Without a catalog in the prompt the model answers from memory and can invent products, specs and review quotes.",
        "Ground it: let the LLM extract constraints, let **code** (pandas) match the catalog, then let the LLM explain only the retrieved rows.",
        "When nothing fits, say so (gap note) and show the nearest alternatives.",
        "Evaluate with cheap code checks first (exists in catalog? constraints met? quote verbatim?), then an LLM judge for clarity.",
        "Same modular pipeline as ScamGuard: only the prompt and the output schema changed.",
    ])

with tabs[9]:
    quiz("m12", [
        {"q": "The original SpecSense prompt contains no product list. What can go wrong?", "options": ["Nothing, Gemini knows every shop's catalog", "The model may invent products, specs and reviews",
         "The JSON will be invalid", "Streamlit will crash"], "answer": 1, "why": "Without data in the prompt the model answers from memory."},
        {"q": "Which part of the grounded pipeline should be plain code, not the LLM?", "options": ["Filtering the catalog by price and weight", "Writing the explanation", "Extracting soft preferences", "Greeting the user"],
         "answer": 0, "why": "Exact numeric filtering is deterministic and testable."},
        {"q": "No product satisfies every constraint. What should SpecSense do?", "options": ["Return an empty answer", "Invent a product that fits", "Say so and show the nearest alternatives with rationale", "Ignore the budget"],
         "answer": 2, "why": "That is the “highlights gaps” capability."},
        {"q": "How can you automatically check that a review snippet is not made up?", "options": ["Ask the model if it is sure", "Check it is a substring of the catalog row's review", "Count its words", "You can't"],
         "answer": 1, "why": "A verbatim-match check against your data is cheap and exact."},
        {"q": "A hard constraint vs a soft preference — which is a soft one?", "options": ["Under ₹30,000", "Weight below 1.5 kg", "“Good for coding”", "At least 8 GB RAM"], "answer": 2,
         "why": "Soft preferences are subjective and need judgement (CPU, RAM, reviews)."},
    ])

catalog.module_downloads("m12", "⬇️ More downloads for this project")
