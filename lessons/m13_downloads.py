import io
import re
import zipfile
from pathlib import Path

import streamlit as st

from core import catalog
from core.ui import mdtable, page_header, warn

ROOT = Path(__file__).resolve().parent.parent

page_header(
    "m13",
    "Every file in this course, what it does, and a download button for it — plus a complete guide to API keys and secrets.",
    [
        "Find and download any script, notebook, project or set of notes",
        "Run the notebooks in Google Colab with keys held in Colab Secrets",
        "Get the Gemini, OpenWeather and OMDb keys",
        "Store keys safely: .env, environment variables, Streamlit Secrets, Colab Secrets",
        "Know what to do if a key leaks",
    ],
)

tabs = st.tabs(["📦 Overview & bundles", "📓 Colab notebooks", "🐍 Scripts", "📁 Projects", "📚 Notes", "🧩 This app's files", "🔑 API keys & secrets"])

# ---------------------------------------------------------------------------
with tabs[0]:
    n_nb, n_sc = len(catalog.NOTEBOOKS), len(catalog.SCRIPTS)
    n_pr, n_no = len(catalog.PROJECTS), len(catalog.NOTES)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Colab notebooks", n_nb)
    c2.metric("Python scripts", n_sc)
    c3.metric("Projects", n_pr)
    c4.metric("Notes (PDF / Word)", n_no)
    st.info("🔐 **Every file here is key-free.** You add your own keys after downloading — see the **🔑 API keys & secrets** tab.")
    st.markdown("#### One-click bundles")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        catalog.zip_button("⬇️ All notebooks", "notebooks", "colab_notebooks.zip", key="b_nb", use_container_width=True)
    with b2:
        catalog.zip_button("⬇️ All scripts", "scripts", "course_scripts.zip", key="b_sc", use_container_width=True)
    with b3:
        catalog.zip_button("⬇️ All projects", "projects", "course_projects.zip", key="b_pr", use_container_width=True)
    with b4:
        catalog.zip_button("⬇️ Notes (PDF/Word)", "notes", "course_notes.zip", key="b_no", use_container_width=True)
    st.download_button("⬇️ EVERYTHING (notebooks + scripts + projects + notes)", data=catalog.zip_many(["notebooks", "scripts", "projects", "notes"]),
                       file_name="genai_course_downloads.zip", mime="application/zip", key="b_all", type="primary", use_container_width=True)
    st.markdown("#### Where each module's files are")
    mdtable(["Module", "Notebook", "Scripts", "Project / notes"], [
        ("5 · API integration", "—", "—", "Notes (Word)"),
        ("6 · Handling responses", "M6_response_handling", "14 scripts (stream, batch, Pydantic, mini project)", "Notes (PDF)"),
        ("7 · Conversations", "M7_conversational_context", "3 scripts (no context / SDK chat / OpenAI library)", "Notes (PDF)"),
        ("8 · Function calling", "M8_function_calling", "4 scripts (single, multi, OpenWeather, LangChain)", "Notes (PDF)"),
        ("9 · Evaluating", "M9_llm_as_judge", "2 scripts (Study Buddy, evaluator)", "Notes (PDF)"),
        ("10 · ScamGuard AI", "P1_ScamGuard_AI", "—", "Full project + 2 datasets + screenshots"),
        ("11 · Cinema Advisor", "P2_Cinema_Advisor_LangChain_OMDb", "—", "Full project (with LLM judge)"),
        ("12 · SpecSense AI", "P3_SpecSense_AI", "—", "Full project + sample product catalog"),
    ])

# ---------------------------------------------------------------------------
with tabs[1]:
    st.markdown(f"{len(catalog.NOTEBOOKS)} notebooks, ready for **Google Colab**. Each one: installs its packages (`!pip install`), reads keys from **Colab Secrets** "
                "(or asks once with hidden input) and contains no key.")
    st.markdown("##### How to run a notebook in Colab")
    st.markdown("""
1. Download the `.ipynb` below → go to **colab.research.google.com** → **File ▸ Upload notebook**. *(Or click “Open in Colab” once the repo is on GitHub and public.)*
2. Click the **🔑 Secrets** icon in the left sidebar → **Add new secret** → name `GEMINI_API_KEY`, paste your key, switch **Notebook access** on.
3. **Runtime ▸ Run all** (or run cell by cell and edit the prompts as you go).
""")
    for rel, title, desc, mod in catalog.NOTEBOOKS:
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 1.3, 1.3])
            keys = {"M6": "GEMINI_API_KEY", "M7": "GEMINI_API_KEY", "M8": "GEMINI_API_KEY, OPENWEATHER_API_KEY", "M9": "GEMINI_API_KEY",
                    "P1": "GEMINI_API_KEY", "P2": "GEMINI_API_KEY, OMDB_API_KEY", "P3": "GEMINI_API_KEY"}[rel.split("/")[-1][:2]]
            c1.markdown(f"**📓 {title}**  \n`{rel.split('/')[-1]}`  \n{desc}  \n🔑 Secrets needed: `{keys}`")
            with c2:
                catalog.dl_button(rel, "⬇️ .ipynb", key=f"nbtab_{rel}", use_container_width=True)
            c3.link_button("▶ Open in Colab", catalog.colab_url(rel), use_container_width=True)
    st.caption(f"“Open in Colab” opens `github.com/{catalog.REPO}` → `downloads/notebooks/…` and works after you push this repo (public repo or Colab connected to your GitHub).")
    st.markdown("**Model name:** the notebooks use `gemini-3.5-flash-lite`. If your key can't access it, change the `MODEL = …` line in the first code cell to `gemini-2.5-flash`.")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.markdown("The original class scripts (names tidied, **keys removed**). Each reads its key from an environment variable — the “needs” column shows which.")
    mod_titles = {"m6": "Module 6 · Handling responses", "m7": "Module 7 · Conversations", "m8": "Module 8 · Function calling", "m9": "Module 9 · Evaluating outputs"}
    for mod, title in mod_titles.items():
        with st.expander(f"{title}  ({sum(1 for s in catalog.SCRIPTS if s[3] == mod)} files)", expanded=(mod == "m6")):
            for rel, t, d, _ in [s for s in catalog.SCRIPTS if s[3] == mod]:
                c1, c2 = st.columns([5, 1.2])
                needs = catalog.env_vars(rel)
                c1.markdown(f"**{t}** — `{rel.split('/')[-1]}`  \n{d}" + (f"  \n🔑 needs `{'`, `'.join(needs)}`" if needs else ""))
                with c2:
                    catalog.dl_button(rel, "⬇️ .py", key=f"sc_{rel}", use_container_width=True)
    st.warning("Heads-up: the class scripts don't all use the same variable name (`GOOGLE_API_KEY`, `GEMINI_API_KEY`, `API_KEY`). "
               "Easiest fix: put **both** `GEMINI_API_KEY` and `GOOGLE_API_KEY` (same value) in your `.env`.")

# ---------------------------------------------------------------------------
with tabs[3]:
    for key, name, mod, files, desc, nb in catalog.PROJECTS:
        with st.expander(f"📁 {name} — {len(files)} files", expanded=(key == "scamguard_ai")):
            st.markdown(desc)
            c1, c2, c3 = st.columns(3)
            with c1:
                catalog.zip_button("⬇️ Whole project (.zip)", f"projects/{key}", f"{key}.zip", key=f"pj_zip_{key}", use_container_width=True)
            with c2:
                catalog.dl_button(nb, "⬇️ Colab notebook", key=f"pj_nb_{key}", use_container_width=True)
            with c3:
                st.link_button("▶ Open in Colab", catalog.colab_url(nb), use_container_width=True)
            st.markdown("**Every file**")
            for f, d in files.items():
                c1, c2 = st.columns([5, 1.1])
                c1.markdown(f"`{f}` — {d}")
                with c2:
                    catalog.dl_button(f"projects/{key}/{f}", "⬇️", key=f"pj_{key}_{f}")

# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("Class notes for each session.")
    for rel, title, mod in catalog.NOTES:
        c1, c2 = st.columns([4, 1.5])
        c1.markdown(f"**{title}**")
        with c2:
            catalog.dl_button(rel, "⬇️ Download", key=f"nt_{rel}", use_container_width=True)

# ---------------------------------------------------------------------------
APP_FILES = [
    ("app.py", "Entry point: sidebar (Gemini key + model), navigation to every lesson page."),
    ("requirements.txt", "Python packages the app needs."),
    ("README.md", "How to run and deploy the app."),
    (".streamlit/config.toml", "Theme and server settings."),
    ("core/llm.py", "Shared Gemini client, 4-calls/min rate limiter, retries, `generate`, `stream`, `run_live` (button + exact-prompt panel), usage/cost helpers."),
    ("core/ui.py", "Page header, cards, steppers, quiz, `diagram` and `mdtable` helpers."),
    ("core/diagrams.py", "All Graphviz teaching diagrams (`show_dg`)."),
    ("core/codelab.py", "`code_lab`: editable, runnable code boxes (▶ Run, ↺ Reset, live output). Turn off with `CODE_LAB_DISABLED=1`."),
    ("core/catalog.py", "This page's catalogue: file descriptions and download-button helpers."),
    ("core/course_prompts.py", "Study Buddy + evaluator prompts and `safe_parse_json` (Module 9)."),
    ("core/sim.py", "Offline simulators used by several lessons."),
    ("lessons/home.py", "Home page and course map."),
    ("lessons/m1_foundations.py", "Module 1 · Foundations (sampling-parameter lab)."),
    ("lessons/m2_platforms.py", "Module 2 · Platforms & LLM landscape."),
    ("lessons/m3_prompt_basics.py", "Module 3 · Prompt basics, dynamic prompting."),
    ("lessons/m4_advanced_prompting.py", "Module 4 · CoT, Auto-CoT, self-consistency, ToT, ReAct."),
    ("lessons/m51_api_basics.py", "Module 5.1 · How APIs work."),
    ("lessons/m52_api_integration.py", "Module 5.2 · LLM API integration, tokens, caching, retries."),
    ("lessons/m6_response_handling.py", "Module 6 · Streaming, batching, Pydantic, guardrails."),
    ("lessons/m7_conversational.py", "Module 7 · Chat, memory, context strategies."),
    ("lessons/m8_function_calling.py", "Module 8 · Tools and function calling (weather key box)."),
    ("lessons/m9_evaluation.py", "Module 9 · Hallucinations, metrics, LLM-as-judge."),
    ("lessons/m10_scamguard.py", "Project 10 · ScamGuard AI: problem statement, description, pipeline, prompts, metrics."),
    ("lessons/m11_cinema_advisor.py", "Project 11 · Cinema Advisor: problem statement, structured output, LLM judge, tracing."),
    ("lessons/m12_specsense.py", "Project 12 · SpecSense AI: problem statement, original vs catalog-grounded recommender, evaluation."),
    ("lessons/m13_downloads.py", "This page."),
    ("downloads/…", "The key-free files offered on this page (notebooks, scripts, projects, notes)."),
    ("tools/build_assets.py", "Rebuilds `downloads/` from the original course folder, removing keys."),
    ("tools/build_notebooks.py", "Regenerates the Colab notebooks."),
]


@st.cache_data(show_spinner=False)
def _app_zip() -> bytes:
    skip_dirs = {".git", "__pycache__", ".venv", "downloads"}
    skip_names = {".env", "secrets.toml", ".DS_Store"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT)
            if not p.is_file() or set(rel.parts) & skip_dirs or p.name in skip_names or p.suffix == ".pyc":
                continue
            z.write(p, str(rel))
    return buf.getvalue()


with tabs[5]:
    st.markdown("The Streamlit app itself.")
    mdtable(["File", "What it does"], [(f"`{f}`", d) for f, d in APP_FILES])
    st.download_button("⬇️ App source code (.zip, no keys)", data=_app_zip(), file_name="genai_learning_hub_source.zip", mime="application/zip", key="app_zip")
    st.caption("The zip skips `.env`, `secrets.toml`, caches and the `downloads/` folder (get that above).")

# ---------------------------------------------------------------------------
with tabs[6]:
    warn("**Treat an API key like a password.** Anyone who has it can spend your quota (or money) and act as you. "
         "Never put it in code, a notebook output, a screenshot, a chat, a README or Git.")

    st.markdown("### 1 · What is each key and where do I get it?")
    mdtable(["Key (env var)", "Used for", "How to get it", "Free tier / notes"], [
        ("`GEMINI_API_KEY` (alias `GOOGLE_API_KEY`)", "All Gemini calls: every module, ScamGuard, Cinema Advisor",
         "[aistudio.google.com/apikey](https://aistudio.google.com/apikey) → sign in → **Create API key** → copy", "Free tier with rate limits. This app paces calls at 4/min."),
        ("`OPENWEATHER_API_KEY` (also accepted: `WEATHER_API_KEY`)", "Weather tool in Module 8", "[home.openweathermap.org/api_keys](https://home.openweathermap.org/api_keys) → sign up → **API keys** tab → copy the *Default* key (or generate one)",
         "Free plan; a **new key needs ~10 min to 2 h** to activate (401 until then). Endpoint: `api.openweathermap.org/data/2.5/weather`."),
        ("`OMDB_API_KEY`", "Posters and ratings in Cinema Advisor", "[omdbapi.com/apikey.aspx](https://www.omdbapi.com/apikey.aspx) → choose **Free** → enter your e-mail → click the **activation link** in the mail",
         "Free: 1,000 requests/day."),
    ])

    st.markdown("### 2 · Where do I store keys?")
    where = st.tabs(["💻 Local `.env`", "🖥️ Environment variable", "☁️ Streamlit Cloud Secrets", "🔬 Google Colab Secrets", "🧪 This app's sidebar"])
    with where[0]:
        st.markdown("Best for running on your own computer. A `.env` file sits next to your code and is loaded by `python-dotenv`.")
        st.code("""# .env   (plain text, one KEY=value per line, no quotes, no spaces around =)
GEMINI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here""", language="bash")
        st.code("""# .gitignore  - MUST contain these lines BEFORE your first commit
.env
.env.*
!.env.example
.streamlit/secrets.toml
__pycache__/""", language="bash")
        st.code("""from dotenv import load_dotenv
import os
load_dotenv()                                   # reads .env into the environment
key = os.getenv("GEMINI_API_KEY")               # None if missing
if not key:
    raise SystemExit("GEMINI_API_KEY is not set - add it to .env")""", language="python")
        st.caption("Share a **`.env.example`** (same names, fake values) so others know what to create. Never share `.env`.")
    with where[1]:
        st.code("""# macOS / Linux  (current terminal only)
export GEMINI_API_KEY="your_key_here"
# Windows PowerShell
$env:GEMINI_API_KEY = "your_key_here"
# Windows (permanent, new terminals)
setx GEMINI_API_KEY "your_key_here\"""", language="bash")
    with where[2]:
        st.markdown("""For the deployed app (share.streamlit.io): **App ▸ ⋮ ▸ Settings ▸ Secrets**, paste TOML, save.""")
        st.code('''GEMINI_API_KEY = "your_key_here"
OPENWEATHER_API_KEY = "your_key_here"
# optional: switch the editable/runnable code boxes OFF on a public app
CODE_LAB_DISABLED = "1"''', language="toml")
        st.markdown("Top-level secrets are also exposed as environment variables, so this app's `os.getenv(\"GEMINI_API_KEY\")` finds them. "
                    "Locally you can put the same text in `.streamlit/secrets.toml` — **and git-ignore that file**.")
        st.warning("On a **public** deployment with a Gemini key in Secrets, everyone who opens the app uses *your* quota, and the editable code boxes run code on *your* server. "
                   "Set `CODE_LAB_DISABLED = \"1\"` or keep the app private, and leave the secret unset so visitors paste their own key in the sidebar.")
    with where[3]:
        st.markdown("""
1. In Colab click the **🔑** icon in the left sidebar (**Secrets**).
2. **+ Add new secret** → *Name* `GEMINI_API_KEY` → *Value* paste your key.
3. Switch **Notebook access** **on** (per notebook).
4. In code: `from google.colab import userdata; key = userdata.get("GEMINI_API_KEY")` — the notebooks here already do this, and fall back to a hidden `getpass` prompt.
""")
        st.code('''from google.colab import userdata
import os
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")   # never print it''', language="python")
        st.caption("Secrets belong to your Google account, not the notebook file, so sharing the notebook doesn't share the key. "
                   "Still: clear outputs that might show a key (Edit ▸ Clear all outputs) before sharing.")
    with where[4]:
        st.markdown("The sidebar's **Gemini API key** box keeps the key in your browser session only (never written to disk). "
                    "If `GEMINI_API_KEY` / `GOOGLE_API_KEY` is already in the environment the box offers an *override*.")

    st.markdown("### 3 · Golden rules")
    st.markdown("""
- ✅ One key per project, so you can revoke one without breaking the others.
- ✅ Keep keys in `.env` / Secrets, and `.env` in `.gitignore` **before the first commit**.
- ✅ Give keys the least power (restrict by API/IP in the provider console when offered) and watch the usage page.
- ❌ Never type a key in source, a notebook cell, a README, an issue, a screenshot or a chat message.
- ❌ Never upload a `.env` file or a project folder that contains one (zip files too!).
""")

    st.markdown("### 4 · A key leaked — what now?")
    st.markdown("""
1. **Revoke it immediately** in the provider's console (AI Studio, OpenWeather, OMDb) and **create a new one**. This is the only step that really fixes it.
2. Put the new key in `.env` / Secrets, not in code.
3. Remove it from the code, then from Git history if it was committed (`git filter-repo` or the BFG tool). Deleting the file in a new commit is **not enough** — old commits still contain it.
4. Check the provider's usage page for calls you didn't make.
""")
    st.code("""# Quick pre-commit check: anything key-shaped in what you're about to commit?
git diff --cached | grep -nE "AIza[0-9A-Za-z_-]{20,}|gsk_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{20,}\\." && echo "STOP: key found" || echo "clean"
git ls-files | grep -E "(^|/)\\.env$" && echo "STOP: .env is tracked" || echo ".env not tracked\"""", language="bash")

    st.markdown("### 5 · Tools")
    t1, t2 = st.tabs(["🔎 Does my key look right?", "📝 Make a `.env.example`"])
    with t1:
        st.caption("Checks only the *shape* of the text, in your browser session. Nothing is sent anywhere or stored.")
        provider = st.selectbox("Provider", ["Gemini", "OpenWeather", "OMDb"], key="kc_prov")
        val = st.text_input("Paste a key to check its format", type="password", key="kc_val")
        rules = {"Gemini": (r"AIza[0-9A-Za-z_\-]{35}", "starts with `AIza`, 39 characters"),
                 "OpenWeather": (r"[0-9a-f]{32}", "32 lowercase hex characters"),
                 "OMDb": (r"[0-9a-f]{8}", "8 hex characters")}
        pat, desc = rules[provider]
        if val:
            v = val.strip()
            if v != val:
                st.warning("There is a space or newline at the start/end — remove it, it causes 401 errors.")
            if re.fullmatch(pat, v):
                st.success(f"Looks like a {provider} key ({desc}). That only checks the shape, not that it's active.")
            else:
                st.error(f"Doesn't match the usual {provider} format: {desc}.  Check you copied the whole key with no spaces.")
    with t2:
        opts = {"GEMINI_API_KEY": "your_gemini_key_here", "GOOGLE_API_KEY": "same_gemini_key_here", "OPENWEATHER_API_KEY": "your_openweather_key_here",
                "OMDB_API_KEY": "your_omdb_key_here"}
        sel = st.multiselect("Keys your project needs", list(opts), default=["GEMINI_API_KEY"], key="env_sel")
        text = "# Copy to .env and fill in YOUR values. Never commit .env.\n" + "\n".join(f"{k}={opts[k]}" for k in sel) + "\n"
        st.code(text, language="bash")
        st.download_button("⬇️ .env.example", data=text, file_name=".env.example", mime="text/plain", key="env_dl")
        gi = ".env\n.env.*\n!.env.example\n.streamlit/secrets.toml\n__pycache__/\n*.pyc\n.venv/\n"
        st.download_button("⬇️ .gitignore (secrets-safe)", data=gi, file_name=".gitignore", mime="text/plain", key="gi_dl")
