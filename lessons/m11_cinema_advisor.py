import json
import re

import streamlit as st

from core import catalog
from core.codelab import code_lab
from core.diagrams import chain
from core.llm import generate, run_live
from core.ui import analogy, cards, definition, diagram, mdtable, page_header, quiz, takeaway, warn

PROJ = catalog.DL / "projects" / "movie_advisor"

page_header(
    "m11",
    "A movie-recommendation chat app that returns validated, structured answers, enriches them with posters and ratings, and scores its own recommendations with an independent judge.",
    [
        "Describe the Cinema Advisor problem and its pipeline",
        "Use Pydantic schemas to get validated structured output from an LLM",
        "Explain why search grounding and structured output need two calls",
        "Build and read an independent LLM-as-judge evaluator (and clamp its scores)",
        "Trace each step of a request for debugging",
        "Get and store the Gemini and OMDb keys safely",
    ],
)

tabs = st.tabs(["📋 Problem statement", "📖 Project description", "🧠 Structured output", "🧑‍⚖️ The evaluator", "🔍 OMDb & tracing",
                "🗂️ Code walkthrough", "🔑 Keys", "⬇️ Downloads", "🧾 Key takeaways", "✅ Knowledge check"])


def extract_json(text: str):
    try:
        m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Gemini Cinema Advisor: Problem Statement")
    st.markdown("**Overview.** Streaming services offer thousands of titles, yet people still spend longer choosing a film than watching one. Rating-based and "
                "“people also watched” recommenders ignore what the user actually says — *“slow-burn sci-fi like Arrival, nothing with jump-scares”* — and they cannot "
                "explain **why** a title fits.")
    st.markdown("**Gemini Cinema Advisor** is a conversational assistant that understands those preferences, remembers what the user loves and hates across the chat, "
                "recommends **real** films with a **specific reason** for each, enriches every pick with **poster, IMDb rating, genre and runtime**, and finally "
                "**evaluates its own recommendations** with an independent judge so quality is measurable, not assumed.")
    st.markdown("#### Problem context")
    mdtable(["Challenge", "Example"], [
        ("**Free-form taste**", "“Something like Arrival but funnier, and I hate horror”"),
        ("**Memory across turns**", "A dislike stated in turn 1 must still be respected in turn 5"),
        ("**Freshness**", "“What's new in 2026?” is after the model's training data"),
        ("**Trust**", "LLMs invent film titles; the app must recommend only real films and show proof (posters, ratings)"),
        ("**Measurable quality**", "How do you know a recommendation list is good? Someone must score it — honestly"),
    ])
    st.markdown("#### What the system does")
    st.markdown("""
1. **Understands** the request and the running chat history (likes, dislikes, moods)
2. **Researches** current titles with Google Search when the request is about new releases
3. **Returns structured output** — a short reply, up to 5 movies (title, year, reason, genres) and an optional follow-up question — validated by Pydantic
4. **Enriches** each movie from OMDb (poster, IMDb rating, runtime, genre) and checks that the poster URL really loads
5. **Judges** the result with a separate critic call: a 1–10 score per movie and for the set, with one-sentence justifications
6. **Traces** every step with timings so failures are easy to debug
""")
    st.markdown("#### User flow")
    diagram(chain([("1 · Message\\n+ chat history", "grey"), ("2 · Research\\n(if “new/latest/2026”)", "yellow"), ("3 · Structured call\\nPydantic schema", "orange"),
                   ("4 · OMDb lookup\\nposter · rating", "blue"), ("5 · Judge call\\nscores 1-10", "purple"), ("6 · Cards + debug trace", "green")]),
            "The six steps of one chat turn.", max_w=10, max_h=3.2)
    st.markdown("#### Data")
    mdtable(["Source", "Contents"], [("**Chat history**", "`(role, text)` pairs replayed into the prompt each turn"),
                                     ("**OMDb API**", "Title, year, genre, plot, poster, IMDb rating, director, actors, runtime"),
                                     ("**Google Search (optional)**", "Up-to-date titles for recent/upcoming releases")])
    st.markdown("#### Technical requirements")
    st.markdown("Python 3.10+ · Streamlit · LangChain + `langchain-google-genai` · Gemini API · Pydantic · requests · python-dotenv · an OMDb key")
    st.markdown("#### Extend it")
    mdtable(["Extension", "How"], [("**Seen-it memory**", "Store watched/liked titles (SQLite or a vector store) so picks never repeat across sessions"),
                                   ("**Filters**", "Era, runtime and streaming availability as sidebar widgets folded into the prompt"),
                                   ("**Cache**", "`st.cache_data` around OMDb lookups"),
                                   ("**Judge-driven retry**", "If the overall score < 6, regenerate with the judge's feedback appended")])

# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### 🍿 Gemini Cinema Advisor")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📌 Features**")
        st.markdown("- Chat UI with memory of likes and dislikes\n- **Pydantic-validated** structured recommendations\n- Poster / rating / runtime cards from OMDb\n"
                    "- **Independent LLM judge** scoring every movie and the set\n- Optional Google-Search research for new releases\n- Debug **trace** with timings")
    with c2:
        st.markdown("**🛠 Stack**")
        st.markdown("- **Streamlit** — chat UI\n- **LangChain + Gemini** — prompt → model → structured output\n- **Pydantic** — schemas and validation\n- **OMDb API** — movie facts and posters")
    st.markdown("**Project layout**")
    st.code("""movie_advisor/
├── app.py          # Streamlit UI + chat loop + movie cards
├── recommender.py  # LangChain chain; optional Google-Search research call
├── models.py       # Pydantic: RecommendationResponse, MovieEvaluation, OmdbMovie
├── evaluator.py    # independent judge call -> EvaluationResponse
├── omdb_client.py  # OMDb lookups + poster reachability check
├── tracing.py      # tiny tracer: log(), timed(), get_trace_text()
└── requirements.txt""", language="text")
    st.markdown("**How one turn works**")
    st.markdown("""
1. `recommender.get_recommendation()` decides whether the text needs current info (`_needs_research`: words like *new, latest, 2026*).
2. If so, a **grounded research call** (Google Search tool, plain text) collects real current titles.
3. A **structured call** (`with_structured_output(RecommendationResponse)`) formats the answer — using the research notes when present.
4. For each movie `OmdbClient.get_movie()` fetches the facts and verifies the poster URL.
5. `evaluator.get_evaluation()` makes a **fresh** call with a strict critic prompt and returns scores.
6. `tracing` records each step; the UI shows it in a debug expander.
""")
    st.info("**Why this project over a tool-calling agent?** Every step is predictable and testable, it needs only two keys (Gemini + OMDb), and it contains a dedicated **evaluator** — "
            "the piece that turns a demo into something you can measure.")

# ---------------------------------------------------------------------------
with tabs[2]:
    definition("Structured output", "asking the model to return data that fits a schema (a Pydantic class), so the app gets typed fields instead of prose.")
    st.markdown("##### The schema (from `models.py`)")
    st.code('''class MovieRecommendation(BaseModel):
    title: str   = Field(description="The exact movie title, as it would appear on IMDb/OMDb.")
    year: Optional[int] = None
    reason: str  = Field(description="Short, specific reason this fits the user's request.")
    genres: List[str] = []

class RecommendationResponse(BaseModel):
    reply: str                               # short conversational text
    movies: List[MovieRecommendation] = []   # 0-5 films (a validator caps it at 5)
    follow_up_question: Optional[str] = None

llm.with_structured_output(RecommendationResponse)   # LangChain does the schema + parsing''', language="python")
    st.markdown("##### Two calls, not one")
    cards([("Grounded research", "For words like *new, latest, 2026* a first call runs with **Google Search** and returns real current titles as plain text."),
           ("Structured formatting", "A second call turns those notes (or the model's own knowledge) into the validated object. "
                                     "Gemini can't use search grounding **and** a response schema in the same request — hence two calls.")], cols=2)
    st.markdown("##### 🟢 Try it: recommend → validate")
    code_lab("m11_struct", '''\
from pydantic import BaseModel, Field
from typing import List, Optional

class Movie(BaseModel):
    title: str
    year: Optional[int] = None
    reason: str = Field(description="why it fits THIS request")

class Recs(BaseModel):
    reply: str
    movies: List[Movie]

request = "I love slow-burn sci-fi like Arrival and hate jump-scare horror. Suggest 3 films."
r = client.models.generate_content(model=MODEL, contents=request,
        config=types.GenerateContentConfig(system_instruction="You are an enthusiastic film critic. Recommend only real films.",
                                           response_mime_type="application/json", response_schema=Recs))
recs = Recs.model_validate_json(r.text)          # raises if the shape is wrong
print(recs.reply, "\\n")
for m in recs.movies:
    print(f"- {m.title} ({m.year}): {m.reason}")''', title="structured recommendation", height=420)

# ---------------------------------------------------------------------------
with tabs[3]:
    definition("LLM-as-judge", "a **separate** model call, with a strict critic prompt, that scores another call's output. Fresh context means it can't grade its own homework.")
    st.markdown("`evaluator.py` makes **one structured call per turn** and returns an `EvaluationResponse`:")
    st.code('''class MovieEvaluation(BaseModel):
    title: str
    relevance_score: int      # 1-10: how well THIS movie fits the user's request
    justification: str        # one specific, critical sentence

class EvaluationResponse(BaseModel):
    overall_score: int        # 1-10 for the whole set
    overall_feedback: str     # honest 1-2 sentence critique
    movie_evaluations: List[MovieEvaluation]''', language="python")
    st.markdown("##### The judge's instructions (from `evaluator.py`)")
    src = (PROJ / "evaluator.py").read_text("utf-8")
    m = re.search(r'EVALUATOR_SYSTEM_PROMPT = """\\\n(.*?)"""', src, re.DOTALL)
    st.code((m.group(1) if m else "see evaluator.py").replace("\\\n", "\n").strip(), language="text", wrap_lines=True)
    cards([("Be critical", "“A generic *good movie* that ignores their specific ask should score low.” — otherwise judges drift toward 9/10 for everything."),
           ("Fresh call", "The judge never sees the recommender's reasoning, only the request and the list."),
           ("Clamp in Python", "`max(1, min(10, v))` — the API doesn't reliably enforce min/max, and a stray 0 or 11 shouldn't fail the whole call."),
           ("Never block the user", "If the evaluator fails, `get_evaluation` returns `(None, error)`; the recommendation still shows.")], cols=4)
    st.markdown("##### Offline: why clamp?")
    code_lab("m11_clamp", '''\
from pydantic import BaseModel, field_validator

class Eval(BaseModel):
    relevance_score: int
    @field_validator("relevance_score")
    @classmethod
    def clamp(cls, v): return max(1, min(10, v))

for raw_score in [0, 7, 11, -3]:
    print(raw_score, "->", Eval(relevance_score=raw_score).relevance_score)''', title="validator clamps a bad score", needs_key=False, height=250)
    st.markdown("##### 🟢 Judge a recommendation list yourself")
    st.caption("Edit the request or the list — try a list that ignores the request (e.g. a horror film for “no horror”) and see whether the judge notices.")
    req = st.text_input("User request", "I love slow-burn sci-fi like Arrival and hate jump-scare horror.", key="m11_req")
    lst = st.text_area("Recommended movies", "- Annihilation (2018): eerie, but with disturbing body-horror scenes\n- Arrival (2016): the film the user already named\n"
                                              "- The Conjuring (2013): a classic scary movie", height=110, key="m11_lst")

    def _judge():
        p = ("You are a strict, independent film-recommendation critic. You did NOT make these recommendations.\n"
             "Score each movie 1-10 on how well it fits the user's SPECIFIC request, and the set overall 1-10. Be genuinely critical; "
             "a generic good movie that ignores the request scores low. One short specific sentence per justification.\n"
             f"User request: {req}\nRecommended movies:\n{lst}\n\n"
             'Return ONLY JSON: {"overall_score": n, "overall_feedback": "...", "movie_evaluations": [{"title": "...", "relevance_score": n, "justification": "..."}]}')
        return extract_json(generate(p)[0])

    j = run_live("🧑‍⚖️ Judge it", _judge, key="m11_judge")
    if j:
        clamp = lambda v: max(1, min(10, int(v)))
        st.metric("Overall", f"{clamp(j.get('overall_score', 1))} / 10")
        st.info(j.get("overall_feedback", ""))
        st.dataframe([{"title": e.get("title"), "score": clamp(e.get("relevance_score", 1)), "why": e.get("justification")} for e in j.get("movie_evaluations", [])],
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("##### OMDb enrichment (`omdb_client.py`)")
    st.markdown("- `get_movie(title, year)` → typed `OmdbMovie` (poster, IMDb rating, runtime, genre …); `Response: False` becomes `found=False` instead of an exception.\n"
                "- **Poster check:** OMDb sometimes returns dead poster URLs (recent releases). `_poster_is_reachable()` sends a HEAD request and drops the URL if it fails, "
                "so the UI shows a clean fallback instead of a broken-image icon.\n- Network errors are caught and returned as `OmdbMovie(found=False, error=…)` — the UI never crashes.")
    st.markdown("##### Tracing (`tracing.py`)")
    st.markdown("Every step calls `tracing.log()` or wraps work in `with tracing.timed(\"label\"):`. At the end the UI prints `get_trace_text()` in a debug expander — "
                "an ordered, timestamped story of the turn, without hunting through a terminal.")
    code_lab("m11_trace", '''\
import time

records = []
def log(msg, level="INFO"):
    records.append(f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}")

class timed:                                   # `with timed("label"): ...` logs start, end and duration
    def __init__(self, label): self.label = label
    def __enter__(self):
        self.t0 = time.perf_counter(); log(f"-> starting: {self.label}"); return self
    def __exit__(self, exc_type, exc, tb):
        ms = (time.perf_counter() - self.t0) * 1000
        log(f"x failed: {self.label} after {ms:.0f}ms - {exc_type.__name__}: {exc}" if exc_type else f"ok finished: {self.label} in {ms:.0f}ms",
            "ERROR" if exc_type else "INFO")
        return False                           # never swallow the exception

with timed("omdb lookup"):
    time.sleep(0.2)
try:
    with timed("judge call"):
        raise TimeoutError("simulated timeout")
except TimeoutError:
    pass
print("\\n".join(records))''', title="a 20-line tracer", needs_key=False, height=520)
    warn("The tracer uses one module-level buffer, so it assumes **one request at a time** — fine for a single-user local app, not for concurrent users (use one trace object per session).")

# ---------------------------------------------------------------------------
with tabs[5]:
    files = catalog.ADVISOR_FILES
    mdtable(["File", "Purpose"], [(f"`{f}`", d) for f, d in files.items()])
    readable = [f for f in files if f.endswith((".py", ".md"))]
    sel = st.selectbox("Read a file", readable, index=readable.index("evaluator.py"), key="m11_file")
    st.code((PROJ / sel).read_text("utf-8"), language="python" if sel.endswith(".py") else "markdown", wrap_lines=True)
    st.code("""cd movie_advisor
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env        # then paste YOUR keys into .env
streamlit run app.py""", language="bash")

# ---------------------------------------------------------------------------
with tabs[6]:
    warn("**The uploaded `.env` files contained live keys** (Gemini, OMDb and others). They were **not** copied into this app, the notebooks or the downloads. "
         "If those files were ever shared, uploaded or committed, **delete and recreate each key**.")
    mdtable(["Key", "Used for", "Get it at", "Notes"], [
        ("`GEMINI_API_KEY`", "Recommender and evaluator", "https://aistudio.google.com/apikey", "Free tier available."),
        ("`OMDB_API_KEY`", "Posters and ratings", "https://www.omdbapi.com/apikey.aspx", "Free: 1,000 requests/day. The key is e-mailed — **click the activation link** in that mail."),
    ])
    st.markdown("Where to store them (`.env`, Streamlit Secrets, Colab Secrets): last page, **⬇️ Downloads & API keys**. The app's sidebar also accepts both keys for the session only.")

# ---------------------------------------------------------------------------
with tabs[7]:
    st.markdown("All files are **key-free**.")
    c1, c2, c3 = st.columns(3)
    with c1:
        catalog.zip_button("⬇️ Whole project (.zip)", "projects/movie_advisor", "movie_advisor.zip", key="m11_zip", use_container_width=True)
    with c2:
        catalog.dl_button("notebooks/P2_Cinema_Advisor_LangChain_OMDb.ipynb", "⬇️ Colab notebook", key="m11_nb", use_container_width=True)
    with c3:
        st.link_button("▶ Open in Colab", catalog.colab_url("notebooks/P2_Cinema_Advisor_LangChain_OMDb.ipynb"), use_container_width=True)
    st.caption("“Open in Colab” works once the repo is on GitHub (public). Otherwise download the .ipynb → colab.research.google.com → File ▸ Upload notebook.")

with tabs[8]:
    takeaway([
        "Structured output + Pydantic gives typed, validated results instead of text you must parse.",
        "Search grounding and response schemas can't share one call — research first, then format.",
        "Score output with a separate, critical judge call; clamp its numbers in Python and never let it block the user.",
        "Verify what the model claims (OMDb facts, poster URLs) instead of trusting it.",
        "Trace every step with timings — debugging an LLM app without a trace is guesswork.",
        "Keys live in `.env` / Secrets, never in code.",
    ])

with tabs[9]:
    quiz("m11", [
        {"q": "Why does the Advisor use two LLM calls for “latest 2026 movies”?", "options": ["To double the cost", "Search grounding and a response schema can't be used in one call",
         "OMDb needs it", "LangChain needs it"], "answer": 1, "why": "Research first, then structure it."},
        {"q": "Why is the judge a separate call?", "options": ["It's cheaper", "A fresh call can't grade its own homework (no shared context or bias)", "Pydantic requires it", "Streamlit requires it"],
         "answer": 1, "why": "Independent context gives a more honest score."},
        {"q": "Why clamp `relevance_score` in Python?", "options": ["The API doesn't reliably enforce min/max", "Pydantic can't validate ints", "Streamlit needs it", "To lower the score"],
         "answer": 0, "why": "Otherwise a 0 or 11 could fail the whole call."},
        {"q": "OMDb returns a poster URL. What does the app do before showing it?", "options": ["Nothing", "Checks with a HEAD request that it really loads", "Downloads it to disk", "Asks the LLM"],
         "answer": 1, "why": "Dead URLs are common for new releases; the check avoids broken images."},
        {"q": "What should happen if the evaluator call fails?", "options": ["Hide the recommendations", "Show the recommendations and surface the evaluator error", "Retry forever", "Crash the app"],
         "answer": 1, "why": "A broken judge must never take down the main flow."},
    ])

catalog.module_downloads("m11", "⬇️ Scripts and notebooks for this project")
