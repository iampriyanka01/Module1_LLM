import streamlit as st

from core import catalog
from core.codelab import code_lab
from core.diagrams import chain, raw, node
from core.ui import analogy, cards, definition, diagram, mdtable, page_header, quiz, takeaway, warn

DL = catalog.DL / "projects"

page_header(
    "m11",
    "Two movie assistants built from the course ideas: CineMate (the LLM decides which tools to call) and Cinema Advisor "
    "(structured output + OMDb + an independent judge).",
    [
        "Trace the tool-calling agent loop in CineMate (model → tool call → result → model)",
        "Explain why per-session state (ProfileManager) beats a module-level global in a web app",
        "Use Pydantic schemas to get validated structured output from an LLM",
        "Explain why search grounding and structured output need two calls",
        "Use a separate LLM-as-judge call to score recommendations",
        "Get and store the Groq, TMDB and OMDb keys safely",
    ],
)

tabs = st.tabs(["🎬 Two projects", "🛠️ CineMate: tool loop", "🧠 Advisor: structured output", "🗂️ Code walkthrough",
                "🔑 Keys for these projects", "⬇️ Downloads", "🧾 Key takeaways", "✅ Knowledge check"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.markdown("Same goal — *recommend movies you will like* — two different architectures. Comparing them shows when to use which pattern.")
    mdtable(["", "CineMate", "Gemini Cinema Advisor"], [
        ("LLM", "Groq · `openai/gpt-oss-20b`", "Google Gemini via LangChain"),
        ("Data source", "**TMDB** (search, discover, details, genres)", "**OMDb** (poster, IMDb rating, runtime) + optional Google Search"),
        ("Who decides what to fetch?", "**The model**, by calling tools (function calling)", "**Your code**: fixed steps in a fixed order"),
        ("Output", "Free-text chat + tool results rendered as posters", "A **Pydantic-validated** object: reply, ≤5 movies, follow-up question"),
        ("Memory", "`ProfileManager`: name, favourite genres/actors, watched movies…", "Chat history replayed into the prompt each turn"),
        ("Quality check", "`run_evaluator` judge in the Streamlit app", "`evaluator.py` judge scores every movie 1–10"),
        ("Pattern from the course", "Module 8 · function calling", "Module 6 · structured output + Module 9 · LLM-as-judge"),
    ])
    diagram(raw(
        node("u", "User message", "grey") + node("m", "LLM", "orange") + node("t", "Tools\\nTMDB · profile", "blue") + node("a", "Answer", "green") +
        "u -> m; m -> t [label=\" tool call\"]; t -> m [label=\" result\" style=dashed color=\"#dc2626\" fontcolor=\"#dc2626\"]; m -> a;", "LR"),
        "CineMate — the model is in charge and loops until it has what it needs.")
    diagram(chain([("User message", "grey"), ("Research call\\n(Google Search)", "yellow"), ("Structured call\\nPydantic schema", "orange"), ("OMDb lookup\\nposter · rating", "blue"),
                   ("Judge call\\nscores 1–10", "purple"), ("Show cards", "green")]),
            "Cinema Advisor — your code is in charge; the model fills in a typed form at each step.")
    analogy("CineMate is a **travel agent** who phones airlines and hotels as needed until the trip is planned. "
            "The Advisor is a **form-based kiosk**: every step is fixed, and the model only fills in the boxes.")

# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown("##### The six tools the model can call")
    mdtable(["Tool", "What it does", "Arguments"], [
        ("`update_user_profile`", "Saves what the user reveals (name, loved/watched movies, genres, languages, actors, directors)", "all optional lists / strings"),
        ("`get_user_profile`", "Reads the saved profile back", "none"),
        ("`search_movies`", "TMDB search by title", "`query`"),
        ("`get_movie_genres`", "TMDB genre names → numeric IDs", "none"),
        ("`discover_movies`", "TMDB discover with filters", "`genre_ids`, `year`, `min_rating`, `language`, `sort_by`"),
        ("`get_movie_details`", "Cast, director, runtime…", "`movie_id`"),
    ])
    st.markdown("##### The agent loop (from `main.py`)")
    st.code('''def run_chat(messages):
    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice="auto", temperature=0.7)
        msg = response.choices[0].message
        messages.append(msg)                       # keep the model's own turn in the history

        if not msg.tool_calls:                     # no tool wanted -> this is the final answer
            return msg.content

        for call in msg.tool_calls:                # the model asked for one or more tools
            args = json.loads(call.function.arguments)
            try:
                result = available_functions[call.function.name](**args)
            except Exception as error:             # tell the model the tool failed; never crash
                result = {"success": False, "error": str(error)}
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
        # loop again: the model now sees the results and decides what to do next''', language="python")
    st.caption("Same shape as the Gemini loop in Module 8 — only the message format differs (`role: tool` + `tool_call_id` instead of `function_response`).")

    st.markdown("##### Why a `ProfileManager` class instead of a global dict?")
    compare_c1, compare_c2 = st.columns(2)
    compare_c1.markdown("❌ **CLI version** — one module-level dict")
    compare_c1.code("user_profile = {'name': None, 'favorite_genres': []}\n\ndef update_user_profile(...):\n    user_profile['name'] = name", language="python")
    compare_c2.markdown("✅ **Streamlit version** — one object per browser session")
    compare_c2.code("class ProfileManager:\n    def __init__(self):\n        self.profile = {'name': None, 'favorite_genres': []}\n\n"
                    "st.session_state.profile_manager = ProfileManager()", language="python")
    st.caption("A Streamlit server is **one Python process shared by every visitor**. A module-level global would leak one user's taste into another user's chat.")

    st.markdown("##### 🧪 Run the loop with a fake model — no key, no network")
    st.caption("A scripted “LLM” first asks for a tool, then answers. Add a second tool call, make a tool crash, or call an unknown tool and watch the loop cope.")
    code_lab("m11_loop", '''\
import json

# --- the real functions (here: tiny fakes) -----------------------------------
PROFILE = {"favorite_genres": []}
def update_user_profile(favorite_genres=None):
    PROFILE["favorite_genres"] += favorite_genres or []
    return {"success": True, "profile": PROFILE}
def search_movies(query):
    return {"results": [{"id": 157336, "title": "Interstellar", "rating": 8.4}]}
available_functions = {"update_user_profile": update_user_profile, "search_movies": search_movies}

# --- a scripted fake LLM: turn 1 = tool calls, turn 2 = final text --------------
SCRIPT = [
    {"tool_calls": [("update_user_profile", {"favorite_genres": ["Sci-Fi"]}), ("search_movies", {"query": "Interstellar"})]},
    {"content": "Saved Sci-Fi to your profile. I found Interstellar (8.4) - want similar films?"},
]
def fake_llm(messages, turn):
    return SCRIPT[turn]

# --- the loop (same logic as main.py) ----------------------------------------
messages = [{"role": "user", "content": "I love sci-fi like Interstellar"}]
turn = 0
while True:
    reply = fake_llm(messages, turn); turn += 1
    if "tool_calls" not in reply:
        print("FINAL ANSWER:", reply["content"]); break
    for name, args in reply["tool_calls"]:
        try:
            result = available_functions[name](**args)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        print(f"tool {name}({args}) -> {result}")
        messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
print("\\nmessages the model has seen by the end:", len(messages))''', title="agent loop with a fake LLM", needs_key=False, height=560)

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
    st.markdown("##### Independent judge")
    st.markdown("`evaluator.py` makes a **fresh** call with a strict critic prompt that scores each movie 1–10 and the set overall. "
                "A separate call can't “grade its own homework”. Scores are **clamped in Python** (`max(1, min(10, v))`) because the API doesn't reliably enforce `ge/le` limits.")
    st.markdown("##### 🟢 Try it: recommend → validate → judge")
    st.caption("Uses your Gemini key from the sidebar (no OMDb needed). Change the request or the schema.")
    code_lab("m11_struct", '''\
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class Movie(BaseModel):
    title: str
    year: Optional[int] = None
    reason: str = Field(description="why it fits THIS request")

class Recs(BaseModel):
    reply: str
    movies: List[Movie]

class Verdict(BaseModel):
    overall_score: int = Field(description="1-10")
    feedback: str
    @field_validator("overall_score")
    @classmethod
    def clamp(cls, v): return max(1, min(10, v))      # APIs don't reliably enforce min/max

request = "I love slow-burn sci-fi like Arrival and hate jump-scare horror. Suggest 3 films."

# CALL 1 - recommender (structured)
r1 = client.models.generate_content(model=MODEL, contents=request,
        config=types.GenerateContentConfig(system_instruction="You are an enthusiastic film critic. Recommend only real films.",
                                           response_mime_type="application/json", response_schema=Recs))
recs = Recs.model_validate_json(r1.text)
print(recs.reply, "\\n")
for m in recs.movies:
    print(f"- {m.title} ({m.year}): {m.reason}")

# CALL 2 - an independent judge (fresh call, strict prompt)
judge_prompt = ("You are a strict film-recommendation critic. You did NOT make these picks. Be critical.\\n"
                f"User request: {request}\\nRecommended: " + "; ".join(f"{m.title} ({m.year}) - {m.reason}" for m in recs.movies))
r2 = client.models.generate_content(model=MODEL, contents=judge_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Verdict))
v = Verdict.model_validate_json(r2.text)
print(f"\\nJUDGE: {v.overall_score}/10 - {v.feedback}")''', title="structured recommendation + judge", height=640)
    st.markdown("##### Offline: why clamping matters")
    code_lab("m11_clamp", '''\
from pydantic import BaseModel, field_validator

class Eval(BaseModel):
    relevance_score: int
    @field_validator("relevance_score")
    @classmethod
    def clamp(cls, v): return max(1, min(10, v))

for raw_score in [0, 7, 11, -3]:
    print(raw_score, "->", Eval(relevance_score=raw_score).relevance_score)''', title="validator clamps a bad score", needs_key=False, height=250)

# ---------------------------------------------------------------------------
with tabs[3]:
    proj = st.radio("Project", ["CineMate (Streamlit)", "CineMate (CLI, doubt session)", "Gemini Cinema Advisor"], horizontal=True, key="m11_proj")
    key, files = {"CineMate (Streamlit)": ("cinemate", catalog.CINE_FILES),
                  "CineMate (CLI, doubt session)": ("cinemate_cli_doubt_session", catalog.CINE_CLI_FILES),
                  "Gemini Cinema Advisor": ("movie_advisor", catalog.ADVISOR_FILES)}[proj]
    mdtable(["File", "Purpose"], [(f"`{f}`", d) for f, d in files.items()])
    readable = [f for f in files if f.endswith((".py", ".md", ".txt"))]
    sel = st.selectbox("Read a file", readable, key=f"m11_file_{key}")
    lang = {"py": "python", "md": "markdown", "txt": "text"}[sel.rsplit(".", 1)[-1]]
    st.code((DL / key / sel).read_text("utf-8"), language=lang, wrap_lines=True)
    st.markdown("**Run locally**")
    st.code(f"""cd {key}
pip install -r requirements.txt{'  streamlit' if key == 'cinemate' else ''}
cp .env.example .env        # then paste YOUR keys into .env
{'python main.py' if key == 'cinemate_cli_doubt_session' else 'streamlit run app.py'}""", language="bash")

# ---------------------------------------------------------------------------
with tabs[4]:
    warn("**The uploaded `.env` files contained live keys** (Groq, TMDB, Gemini, OMDb, and the Session 6 `.env` files). They were **not** copied into this app, "
         "the notebooks or the downloads. If those files were ever shared, uploaded or committed, **delete and recreate each key** (steps below).")
    mdtable(["Key", "Used by", "Get it at", "Notes"], [
        ("`GROQ_API_KEY`", "CineMate", "https://console.groq.com/keys", "Free tier has rate limits; if you see 429 errors wait a minute."),
        ("`TMDB_API_KEY`", "CineMate", "https://www.themoviedb.org/settings/api", "Use the **API Read Access Token (v4)** — the long `eyJ…` string. The short v3 key will give 401."),
        ("`GEMINI_API_KEY`", "Cinema Advisor, ScamGuard", "https://aistudio.google.com/apikey", "Free tier available."),
        ("`OMDB_API_KEY`", "Cinema Advisor", "https://www.omdbapi.com/apikey.aspx", "Free: 1,000 requests/day. The key is e-mailed; **click the activation link** in that mail."),
    ])
    st.markdown("Where to store them: see the last page, **⬇️ Downloads & API keys**.")

# ---------------------------------------------------------------------------
with tabs[5]:
    for title, key, nb in [("CineMate (Streamlit)", "cinemate", "P2_CineMate_Groq_TMDB.ipynb"),
                           ("CineMate (CLI, doubt session)", "cinemate_cli_doubt_session", None),
                           ("Gemini Cinema Advisor", "movie_advisor", "P3_Cinema_Advisor_LangChain_OMDb.ipynb")]:
        st.markdown(f"**{title}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            catalog.zip_button("⬇️ Project (.zip)", f"projects/{key}", f"{key}.zip", key=f"m11_zip_{key}", use_container_width=True)
        if nb:
            with c2:
                catalog.dl_button(f"notebooks/{nb}", "⬇️ Colab notebook", key=f"m11_nb_{key}", use_container_width=True)
            with c3:
                st.link_button("▶ Open in Colab", catalog.colab_url(f"notebooks/{nb}"), use_container_width=True)
        st.divider()

with tabs[6]:
    takeaway([
        "Function calling: the model *requests* a tool, your code runs it, you return the result — and loop until there is no tool call.",
        "Return tool errors to the model as data; never let a tool crash the loop.",
        "Per-user state must live in the session (`st.session_state`), never in module globals.",
        "Structured output + Pydantic gives typed, validated results; clamp what the API can't enforce.",
        "Search grounding and response schemas can't share one call — split into research then format.",
        "Use a separate judge call to score output; keep keys in `.env` / Secrets.",
    ])

with tabs[7]:
    quiz("m11", [
        {"q": "In CineMate, who decides to call `search_movies`?", "options": ["The Streamlit UI", "The LLM, by returning a tool call", "TMDB", "The user's browser"],
         "answer": 1, "why": "Function calling: the model chooses; your code executes."},
        {"q": "Why does CineMate-Streamlit use a `ProfileManager` object per session?", "options": ["It is faster", "A global dict would be shared between all visitors",
         "Groq requires it", "TMDB requires it"], "answer": 1, "why": "One server process serves everyone."},
        {"q": "Why does the Advisor use two LLM calls for 'latest 2026 movies'?", "options": ["To double the cost", "Search grounding and a response schema can't be used in one call",
         "OMDb needs it", "LangChain needs it"], "answer": 1, "why": "Research first, then structure it."},
        {"q": "Why clamp `relevance_score` in Python?", "options": ["The API doesn't reliably enforce min/max", "Pydantic can't validate ints", "Streamlit needs it", "To lower the score"],
         "answer": 0, "why": "Otherwise a 0 or 11 could fail the whole call."},
        {"q": "Which TMDB credential does CineMate need?", "options": ["v3 API key", "v4 API Read Access Token", "Your TMDB password", "An OMDb key"],
         "answer": 1, "why": "It is sent as `Authorization: Bearer …`."},
    ])

catalog.module_downloads("m11", "⬇️ Scripts and notebooks for these projects")
