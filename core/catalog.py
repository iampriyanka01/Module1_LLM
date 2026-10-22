"""Catalogue of every downloadable file (code, notebooks, notes) + helpers that render download buttons.

All files live in `downloads/` (key-free; see tools/build_assets.py). Descriptions are written here so
the last page of the app and the per-module "Download" boxes stay in sync.
"""
from __future__ import annotations

import io
import mimetypes
import re
import zipfile
from pathlib import Path

import streamlit as st

DL = Path(__file__).resolve().parent.parent / "downloads"
REPO = "iampriyanka01/Module1_LLM"
BRANCH = "main"

KEY_HELP = {"GEMINI_API_KEY": "Gemini", "GOOGLE_API_KEY": "Gemini", "OPENWEATHER_API_KEY": "OpenWeather", "GROQ_API_KEY": "Groq",
            "TMDB_API_KEY": "TMDB", "OMDB_API_KEY": "OMDb", "API_KEY": "Gemini"}


def colab_url(rel: str) -> str:
    return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/downloads/{rel}"


# (relative path, title, what it is / what to learn, module)
NOTEBOOKS = [
    ("notebooks/M6_response_handling.ipynb", "M6 · Response handling notebook",
     "Blocking vs streaming (time-to-first-word), async batching with `asyncio.gather`, many inputs in one prompt, Pydantic validation, "
     "LangChain `PydanticOutputParser` + `abatch`, and the customer-feedback mini project.", "m6"),
    ("notebooks/M7_conversational_context.ipynb", "M7 · Conversations & context notebook",
     "Stateless calls, manual history, `chats.create`, a summarise-when-long strategy with editable `SUMMARY_THRESHOLD`, and multi-turn chat through the OpenAI library.", "m7"),
    ("notebooks/M7_auto_summary_chatbot.ipynb", "M7 · Auto-summary chatbot notebook",
     "Your chatbot as a Colab notebook: summary every N turns, summary + last-K memory mode, tokens-sent table, a scripted demo and a live `input()` chat.", "m7"),
    ("notebooks/M8_function_calling.ipynb", "M8 · Function-calling notebook",
     "Weather tool (OpenWeatherMap key from Colab Secrets), the full request→run→reply loop, multi-tool calls, automatic function calling, tool modes AUTO/ANY/NONE, and LangChain tools.", "m8"),
    ("notebooks/M9_llm_as_judge.ipynb", "M9 · LLM-as-judge notebook",
     "Study Buddy target model + a separate evaluator model scoring answers against a rubric, then a batch comparison table.", "m9"),
    ("notebooks/P1_ScamGuard_AI.ipynb", "Project 1 · ScamGuard AI notebook",
     "Writes the whole project into the Colab VM (`%%writefile`), runs the detector on sample messages, shows the exact prompt and evaluates accuracy/recall/F1 on an embedded dataset sample.", "m10"),
    ("notebooks/P2_CineMate_Groq_TMDB.ipynb", "Project 2 · CineMate notebook",
     "The tool-calling movie chatbot (Groq LLM + TMDB data) as a runnable notebook; keys come from Colab Secrets.", "m11"),
    ("notebooks/P3_Cinema_Advisor_LangChain_OMDb.ipynb", "Project 3 · Cinema Advisor notebook",
     "LangChain + Gemini structured output, OMDb enrichment and an independent judge call — no Streamlit needed.", "m11"),
]

SCRIPTS = [
    ("scripts/m6_no_stream.py", "Normal (blocking) call", "Waits for the full answer before printing — the baseline for streaming.", "m6"),
    ("scripts/m6_stream.py", "Streaming with LangChain", "Prints tokens as they arrive; shows `streaming=True` and chunk handling.", "m6"),
    ("scripts/m6_async_stream.py", "Async streaming", "`astream` inside `asyncio` — the pattern for web back-ends.", "m6"),
    ("scripts/m6_no_batch_input.py", "One request at a time", "Loop over inputs sequentially (slow) — baseline for batching.", "m6"),
    ("scripts/m6_batch_input.py", "Batch input", "`llm.batch([...])` — many prompts, one call.", "m6"),
    ("scripts/m6_async_batch_simple.py", "Async batch", "`await llm.abatch([...])` — concurrent requests.", "m6"),
    ("scripts/m6_batch_stream_sync.py", "Batch + stream (sync)", "Combines batching and streaming without asyncio.", "m6"),
    ("scripts/m6_batch_stream_async.py", "Batch + stream (async)", "Combines `abatch` and `astream`.", "m6"),
    ("scripts/m6_pydantic_without_validation.py", "Structured output, no validation", "Shows what goes wrong when you trust raw model text.", "m6"),
    ("scripts/m6_pydantic_with_parser.py", "Pydantic + `PydanticOutputParser`", "Schema, format instructions and validation (incl. a bad e-mail test).", "m6"),
    ("scripts/m6_mini_project_resume_parser.py", "Mini project · resume parser", "Batch + streaming + Pydantic with a malformed-input case.", "m6"),
    ("scripts/m6_customer_feedback_core_python.py", "Customer feedback (plain Python)", "Classify feedback and draft a reply using only the Gemini SDK.", "m6"),
    ("scripts/m6_customer_feedback_langchain.py", "Customer feedback (LangChain)", "Same task with prompt templates and output parsers.", "m6"),
    ("scripts/m6_simple_langchain_chatbot.py", "Simple LangChain chatbot", "Smallest possible chat loop.", "m6"),
    ("scripts/m7_no_context_management.py", "Chat with no context management", "Sends every message alone, so the bot forgets — the problem to solve.", "m7"),
    ("scripts/m7_context_management_summary.py", "Context management (SDK chat)", "`client.chats.create` keeps history; prints the stored conversation.", "m7"),
    ("scripts/m7_context_management_openai.py", "Context management (OpenAI library)", "Multi-turn chat via the OpenAI-compatible Gemini endpoint.", "m7"),
    ("scripts/m7_auto_summary_chatbot_app.py", "Auto-summary chatbot (Streamlit app)", "Run locally with `streamlit run`: sidebar settings, summary memory mode and a tokens-per-turn table.", "m7"),
    ("scripts/m8_single_function_call.py", "Single function call", "One weather tool: model returns a call, you run it.", "m8"),
    ("scripts/m8_openweather_tool.py", "OpenWeather tool", "Same idea, kept as the reference weather script (needs `OPENWEATHER_API_KEY`).", "m8"),
    ("scripts/m8_multi_tool_call.py", "Multiple tools", "`add` and `multiply` with a manual execute-and-return loop.", "m8"),
    ("scripts/m8_multi_tool_langchain.py", "Multiple tools (LangChain agent)", "`@tool` + `create_agent` does the loop for you.", "m8"),
    ("scripts/m9_study_buddy_chatbot.py", "Study Buddy chatbot", "The target model: friendly tutor with a safety/scope system prompt.", "m9"),
    ("scripts/m9_evaluator_model.py", "Evaluator model", "A second model scores Study Buddy's answer with a rubric (LLM-as-judge).", "m9"),
]

# ----- project files -------------------------------------------------------
SCAM_FILES = {
    "README.md": "Project overview and run instructions.",
    "LICENSE": "Licence of the original project.",
    ".env.example": "Template for your key. Copy to `.env` (never commit it).",
    ".gitignore": "Keeps `.env`, caches and outputs out of Git.",
    "requirements.txt": "Python packages: google-genai, pandas, pydantic, python-dotenv, streamlit, tqdm.",
    "config.py": "Settings: model name, retries, dataset paths. **Reads the key from the environment** (the original had it hard-coded — fixed).",
    "utils.py": "Logger, `extract_json_from_text` (finds `{...}` in free text) and `load_file`.",
    "main.py": "Command-line demo: type a message, get the verdict.",
    "evaluate.py": "Runs the detector on a labelled CSV and prints accuracy, recall and F1 per class.",
    "scam_detection_dataset.csv": "≈1,000 labelled messages (Scam / Not Scam) with intent type and reason.",
    "test_scam_dataset.csv": "Tiny 7-message set for quick tests.",
    "llm/client.py": "`LLMClient`: Gemini call with retry + exponential back-off.",
    "llm/prompts.py": "Loads a prompt template file and appends the user's message. Change `PROMPT = load_prompt(...)` to switch strategy.",
    "llm/parser.py": "Pydantic model `ScamDetectionOutput` + `validate_output`.",
    "llm/validator.py": "Same schema (kept as the validation entry point).",
    "llm/__init__.py": "Package marker.",
    "llm/prompts/react.md": "ReAct-style prompt: Thought → Action → Observation → Final JSON (default).",
    "llm/prompts/fewshot.md": "Few-shot prompt with three worked examples in Markdown.",
    "llm/prompts/few_shot_examples.txt": "Four plain-text examples (Scam / Not Scam / Uncertain).",
    "llm/prompts/strict_json.txt": "Strict “JSON only” prompt with schema and criteria.",
    "llm/prompts/simplified.txt": "Short, quick-classification prompt.",
    "pipeline/__init__.py": "Package marker.",
    "pipeline/scam_detector/__init__.py": "Package marker (exports commented out).",
    "pipeline/scam_detector/builder.py": "Step 1 — build the prompt for a chosen strategy.",
    "pipeline/scam_detector/executor.py": "Step 2 — send the prompt through `LLMClient`.",
    "pipeline/scam_detector/parser.py": "Step 3 — pull JSON out of the reply; fall back to “Uncertain”.",
    "pipeline/scam_detector/detector.py": "`ScamDetector`: orchestrates the 3 steps; `detect` and `detect_batch`.",
    "streamlit/app.py": "Browser UI for the detector.",
}
CINE_FILES = {
    "read.md": "Notes on the Streamlit port of the CLI bot.",
    ".env.example": "Template for `GROQ_API_KEY` and `TMDB_API_KEY`.",
    ".gitignore": "Keeps `.env` out of Git.",
    "requirements.txt": "groq, python-dotenv, requests (add streamlit for `app.py`).",
    "tools.py": "TMDB helpers + `ProfileManager` (one profile per browser session) and the four TMDB tools.",
    "main.py": "Command-line agent: system prompt, tool schemas, the tool-calling loop.",
    "app.py": "Streamlit chat UI with profile sidebar, poster gallery and a live tool-status box.",
}
CINE_CLI_FILES = {
    "tools.py": "Original CLI tools (profile kept in a module-level dict — fine for one terminal user).",
    "main.py": "Original CLI agent loop from the doubt session.",
    "requirements.txt": "Dependencies.", ".env.example": "Key template.", ".gitignore": "Keeps `.env` out of Git.",
}
ADVISOR_FILES = {
    "README.md": "Setup and how it works.",
    ".env.example": "Template for `GEMINI_API_KEY` and `OMDB_API_KEY`.",
    ".gitignore": "Keeps `.env` out of Git.",
    "requirements.txt": "streamlit, langchain, langchain-google-genai, pydantic, requests, python-dotenv.",
    "app.py": "Streamlit chat UI with movie cards, posters, judge scores and a debug trace.",
    "recommender.py": "LangChain prompt → Gemini → `RecommendationResponse`; optional Google-Search research call for current titles.",
    "models.py": "Pydantic schemas: recommendations, evaluations (with score clamping) and the OMDb movie.",
    "omdb_client.py": "OMDb lookups; checks that the poster URL really loads.",
    "evaluator.py": "Independent LLM judge that scores each movie and the whole set.",
    "tracing.py": "Tiny execution tracer (`log`, `timed`) shown in the UI's debug panel.",
    "test_omdb_key.py": "One-off script to check that your OMDb key works.",
}

PROJECTS = [
    ("scamguard_ai", "ScamGuard AI", "m10", SCAM_FILES,
     "Scam-message detector: prompt strategies → Gemini → parser → validator → evaluation.", "notebooks/P1_ScamGuard_AI.ipynb"),
    ("cinemate", "CineMate (Streamlit)", "m11", CINE_FILES,
     "Tool-calling movie assistant (Groq + TMDB) with a Streamlit UI.", "notebooks/P2_CineMate_Groq_TMDB.ipynb"),
    ("cinemate_cli_doubt_session", "CineMate (CLI, doubt session)", "m11", CINE_CLI_FILES,
     "The original command-line version of CineMate.", "notebooks/P2_CineMate_Groq_TMDB.ipynb"),
    ("movie_advisor", "Gemini Cinema Advisor", "m11", ADVISOR_FILES,
     "LangChain structured output + OMDb + LLM judge.", "notebooks/P3_Cinema_Advisor_LangChain_OMDb.ipynb"),
]

NOTES = [
    ("notes/5_api_integration_with_llms.docx", "Module 5 · API integration notes (Word)", "m52"),
    ("notes/6_advanced_handling_notes.pdf", "Module 6 · Advanced handling of LLM responses (PDF)", "m6"),
    ("notes/7_conversational_interfaces_notes.pdf", "Module 7 · Conversational interfaces (PDF)", "m7"),
    ("notes/8_function_calling_notes.pdf", "Module 8 · Function calling & tools (PDF)", "m8"),
    ("notes/9_evaluating_outputs_notes.pdf", "Module 9 · Evaluating GenAI outputs (PDF)", "m9"),
]


# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read(path: str, mtime: float) -> bytes:
    return Path(path).read_bytes()


def file_bytes(rel: str) -> bytes:
    p = DL / rel
    return _read(str(p), p.stat().st_mtime)


def exists(rel: str) -> bool:
    return (DL / rel).is_file()


@st.cache_data(show_spinner=False)
def _zip(rels: tuple[str, ...], strip: str, mtime: float) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in rels:
            arc = rel[len(strip):] if strip and rel.startswith(strip) else rel
            z.write(DL / rel, arc)
    return buf.getvalue()


def tree_files(rel_dir: str) -> list[str]:
    base = DL / rel_dir
    return sorted(str(p.relative_to(DL)).replace("\\", "/") for p in base.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.name != ".env")


def zip_bytes(rel_dir: str, strip_parent: bool = True) -> bytes:
    files = tree_files(rel_dir)
    strip = rel_dir.rstrip("/").rsplit("/", 1)[0] + "/" if strip_parent and "/" in rel_dir.rstrip("/") else ""
    return _zip(tuple(files), strip, max((DL / f).stat().st_mtime for f in files))


def zip_many(rel_dirs: list[str]) -> bytes:
    files = [f for d in rel_dirs for f in tree_files(d)]
    return _zip(tuple(files), "", max((DL / f).stat().st_mtime for f in files))


def _mime(name: str) -> str:
    if name.endswith(".ipynb"):
        return "application/x-ipynb+json"
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def dl_button(rel: str, label: str | None = None, key: str | None = None, use_container_width: bool = False):
    """Download button for one file under downloads/."""
    if not exists(rel):
        st.caption(f"(missing: {rel})")
        return
    name = rel.rsplit("/", 1)[-1]
    st.download_button(label or f"⬇️ {name}", data=file_bytes(rel), file_name=name, mime=_mime(name),
                       key=key or f"dl_{rel}", use_container_width=use_container_width)


def zip_button(label: str, rel_dir: str, file_name: str, key: str, use_container_width: bool = False):
    st.download_button(label, data=zip_bytes(rel_dir), file_name=file_name, mime="application/zip", key=key,
                       use_container_width=use_container_width)


def env_vars(rel: str) -> list[str]:
    """Which key names does this file read? (found by scanning the source)"""
    p = DL / rel
    if not p.is_file() or p.suffix not in {".py", ".ipynb"}:
        return []
    t = p.read_text("utf-8", errors="ignore")
    found = set(re.findall(r'(?:getenv|environ(?:\.get|\.setdefault)?|get_secret)\(\s*\\?["\']([A-Z][A-Z0-9_]*_KEY|API_KEY)\\?["\']', t))
    found |= set(re.findall(r'environ\[\s*\\?["\']([A-Z][A-Z0-9_]*_KEY)\\?["\']\s*\]', t))
    return sorted(found)


def module_downloads(module_key: str, title: str | None = None):
    """Compact 'Download code & Colab notebook' box for a lesson page."""
    nbs = [e for e in NOTEBOOKS if e[3] == module_key]
    scripts = [e for e in SCRIPTS if e[3] == module_key]
    projects = [p for p in PROJECTS if p[2] == module_key]
    notes = [n for n in NOTES if n[2] == module_key]
    if not (nbs or scripts or projects or notes):
        return
    st.divider()
    with st.expander(title or "⬇️ Download the code for this module (scripts, Colab notebook, notes)", expanded=False):
        st.caption("Files contain **no API keys**. Add your own: see the last page, “Downloads & API keys”.")
        for rel, t, d, _ in nbs:
            c1, c2, c3 = st.columns([3.2, 1.3, 1.3])
            c1.markdown(f"**📓 {t}**  \n{d}")
            with c2:
                dl_button(rel, "⬇️ .ipynb", key=f"mod_{module_key}_{rel}")
            c3.link_button("▶ Open in Colab", colab_url(rel))
        for key, name, _, files, d, nb in projects:
            c1, c2 = st.columns([4.5, 1.3])
            c1.markdown(f"**📁 {name}** — {d}")
            with c2:
                zip_button("⬇️ Project .zip", f"projects/{key}", f"{key}.zip", key=f"mod_{module_key}_zip_{key}")
        if scripts:
            st.markdown("**🐍 Python scripts**")
            for rel, t, d, _ in scripts:
                c1, c2 = st.columns([4.5, 1.3])
                vars_ = env_vars(rel)
                c1.markdown(f"`{rel.split('/')[-1]}` — **{t}**. {d}" + (f"  \n<small>needs: {', '.join(vars_)}</small>" if vars_ else ""),
                            unsafe_allow_html=True)
                with c2:
                    dl_button(rel, "⬇️ .py", key=f"mod_{module_key}_{rel}")
        for rel, t, _ in notes:
            dl_button(rel, f"⬇️ {t}", key=f"mod_{module_key}_{rel}")


def run_elsewhere_box(nb_rel: str, key: str, script_rel: str | None = None, text: str = "Want to run this outside the app?"):
    """Prominent 'download the code / run it in Google Colab' strip."""
    with st.container(border=True):
        st.markdown(f"**⬇️ {text}**  \nDownload it, or open it in Google Colab (keys are read from **Colab Secrets** — nothing is hard-coded).")
        c1, c2, c3 = st.columns(3)
        with c1:
            dl_button(nb_rel, "⬇️ Colab notebook (.ipynb)", key=f"{key}_nb", use_container_width=True)
        with c2:
            if script_rel:
                dl_button(script_rel, "⬇️ Streamlit app (.py)", key=f"{key}_py", use_container_width=True)
        with c3:
            st.link_button("▶ Open in Colab", colab_url(nb_rel), use_container_width=True)
        st.caption("“Open in Colab” works once the repo is on GitHub (public). Otherwise download the .ipynb → colab.research.google.com → File ▸ Upload notebook. "
                   "Keys: see the last page, “Downloads & API keys”.")
