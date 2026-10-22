"""Build the key-free `downloads/` folder from the course code zip.

    python tools/build_assets.py "/path/to/Module1_Code"        # extracted folder
    python tools/build_assets.py "/path/to/Module1_Code" --notebooks-only

What it does
  * copies the sessions' scripts and the three projects into downloads/
  * NEVER copies .env files, __pycache__, .DS_Store, logs or outputs
  * replaces any hard-coded key with an environment-variable lookup / placeholder
  * normalises awkward file names ("codewithbatchstreamasync .py" -> code_with_batch_stream_async.py)
  * fails loudly if anything that looks like a real key is left in the output
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "downloads"

KEY_PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "YOUR_GEMINI_API_KEY"),
    (re.compile(r"gsk_[A-Za-z0-9_]{20,}"), "YOUR_GROQ_API_KEY"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}"), "YOUR_TMDB_READ_ACCESS_TOKEN"),
    (re.compile(r"sk-[A-Za-z0-9]{32,}"), "YOUR_OPENAI_API_KEY"),
]
SKIP_NAMES = {".env", ".DS_Store", "__pycache__", "logs", "outputs", ".git"}


def scrub(text: str) -> str:
    for pat, repl in KEY_PATTERNS:
        text = pat.sub(repl, text)
    return text


def fix_script(text: str) -> str:
    """Hard-coded placeholder keys -> environment lookups (so the scripts run once the env var is set)."""
    def repl(m):
        name = m.group(1)
        alt = ' or os.getenv("GOOGLE_API_KEY")' if name == "GEMINI_API_KEY" else ""
        return f'{name} = os.getenv("{name}"){alt}  # set in .env / environment - never hard-code'
    text = re.sub(r'^(GEMINI_API_KEY|OPENWEATHER_API_KEY)\s*=\s*"(?:YOUR_[A-Z_]+|Place you api key here)"[^\n]*', repl, text, flags=re.M)
    if "os.getenv(" in text and not re.search(r"^\s*import os\b|^\s*from os\b", text, flags=re.M):
        text = "import os\n" + text
    return text


def copy_tree(src: Path, dst: Path, fix=None):
    for p in sorted(src.rglob("*")):
        if any(part in SKIP_NAMES for part in p.relative_to(src).parts) or p.suffix == ".pyc":
            continue
        target = dst / p.relative_to(src)
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in {".py", ".md", ".txt", ".csv", ".json", ".ipynb", ""} and p.name != "LICENSE":
            raw = p.read_bytes()
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                try:
                    text = raw.decode("utf-16")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1")
            text = scrub(text)
            if fix:
                text = fix(p.name, text)
            target.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(p, target)


def fix_scamguard(name: str, text: str) -> str:
    if name == "config.py":
        text = re.sub(r'GEMINI_API_KEY\s*=\s*"[^"]*"',
                      'GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")  # from .env / Colab Secrets - never hard-code',
                      text)
        text = text.replace('DEFAULT_MODEL = "gemini-2.5-flash"',
                            'DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # e.g. gemini-3.5-flash-lite')
    if name == "requirements.txt":  # the original was saved as UTF-16 with pins; make it plain and loose
        text = "google-genai>=1.20\npandas>=2.0\npydantic>=2.5\npython-dotenv>=1.0\nstreamlit>=1.40\ntqdm>=4.66\n"
    return text


def build(src: Path, notebooks_only: bool = False):
    if not notebooks_only:
        if OUT.exists():
            shutil.rmtree(OUT)
        (OUT / "scripts").mkdir(parents=True)
        (OUT / "notes").mkdir()
        (OUT / "projects").mkdir()

        # ---- session scripts ------------------------------------------------
        s6 = next(src.glob("6.Advanced*/Module 1 Session 6/Session 6 _ Advanced Handling of LLM Responses"))
        names6 = {
            "abatchcodesimple.py": "m6_async_batch_simple.py",
            "code_with_batchinput.py": "m6_batch_input.py",
            "codewithastream.py": "m6_async_stream.py",
            "codewithbatchstreamasync .py": "m6_batch_stream_async.py",
            "codewithbatchstreamsync .py": "m6_batch_stream_sync.py",
            "codewithoutbatchinput.py": "m6_no_batch_input.py",
            "codewithoutstream.py": "m6_no_stream.py",
            "codewithstream.py": "m6_stream.py",
            "customer_feedback_system_using _core_python.py": "m6_customer_feedback_core_python.py",
            "customer_feedback_system_using _langchain.py": "m6_customer_feedback_langchain.py",
            "miniprojectcodefile.py": "m6_mini_project_resume_parser.py",
            "pydanticcode.py": "m6_pydantic_with_parser.py",
            "pydanticwithoutcode.py": "m6_pydantic_without_validation.py",
            "simplelangchainchatbot.py": "m6_simple_langchain_chatbot.py",
        }
        for old, new in names6.items():
            (OUT / "scripts" / new).write_text(fix_script(scrub((s6 / old).read_text("utf-8-sig", errors="replace"))), "utf-8")
        s7 = next(src.glob("7. Conv*/Module 1 Session 7"))
        for old, new in {"no_context_management.py": "m7_no_context_management.py",
                         "context_management.py": "m7_context_management_summary.py",
                         "context_management_openai.py": "m7_context_management_openai.py"}.items():
            (OUT / "scripts" / new).write_text(fix_script(scrub((s7 / old).read_text("utf-8-sig", errors="replace"))), "utf-8")
        s8 = next(src.glob("8. Function*/Module 1 Session 8"))
        for old, new in {"SingleFunctionCall.py": "m8_single_function_call.py", "OWM.py": "m8_openweather_tool.py",
                         "MultiToolCall.py": "m8_multi_tool_call.py",
                         "MultiToolCallUsingLangchain.py": "m8_multi_tool_langchain.py"}.items():
            (OUT / "scripts" / new).write_text(fix_script(scrub((s8 / old).read_text("utf-8-sig", errors="replace"))), "utf-8")
        s9 = next(src.glob("9. Under*/Module 1 Session 9"))
        for old, new in {"evaluatormodel.py": "m9_evaluator_model.py", "smartbuddychatbot.py": "m9_study_buddy_chatbot.py"}.items():
            (OUT / "scripts" / new).write_text(fix_script(scrub((s9 / old).read_text("utf-8-sig", errors="replace"))), "utf-8")

        for extra in (ROOT / "tools" / "extra").glob("*.py"):   # scripts written for this app (no source folder)
            shutil.copy2(extra, OUT / "scripts" / extra.name)

        # ---- notes (course PDFs / docx; contain no keys) ---------------------
        notes = {
            "5.API Integration with LLMs.docx": "5_api_integration_with_llms.docx",
        }
        for pat, new in [("6. Cont*/**/Adv Handling Notes.pdf", "6_advanced_handling_notes.pdf"),
                         ("7. Conv*/**/Session 7 Notes.pdf", "7_conversational_interfaces_notes.pdf"),
                         ("8. Function*/**/Module 1 Session 8.pdf", "8_function_calling_notes.pdf"),
                         ("9. Under*/**/Session 9.pdf", "9_evaluating_outputs_notes.pdf")]:
            notes[str(next(src.glob(pat)).relative_to(src))] = new
        for old, new in notes.items():
            shutil.copy2(src / old, OUT / "notes" / new)

        # ---- projects --------------------------------------------------------
        copy_tree(src / "10 In-Class Project ScamGuard AI/Project-1-Scam_Detection-main", OUT / "projects/scamguard_ai", fix_scamguard)
        copy_tree(src / "11. CineMate", OUT / "projects/cinemate")
        copy_tree(src / "11. Doubt session/CineMate", OUT / "projects/cinemate_cli_doubt_session")
        copy_tree(src / "11. movie_advisor", OUT / "projects/movie_advisor")
        (OUT / "projects/movie_advisor/test omdb key.py").rename(OUT / "projects/movie_advisor/test_omdb_key.py")
        envs = {
            "scamguard_ai": "GEMINI_API_KEY=your_gemini_key_here\n# GEMINI_MODEL=gemini-2.5-flash\n",
            "cinemate": "GROQ_API_KEY=your_groq_key_here\nTMDB_API_KEY=your_tmdb_v4_read_access_token_here\n",
            "cinemate_cli_doubt_session": "GROQ_API_KEY=your_groq_key_here\nTMDB_API_KEY=your_tmdb_v4_read_access_token_here\n",
            "movie_advisor": "GEMINI_API_KEY=your_gemini_key_here\nOMDB_API_KEY=your_omdb_key_here\n",
        }
        for proj, body in envs.items():
            d = OUT / "projects" / proj
            (d / ".env.example").write_text(
                "# Copy this file to `.env` and fill in your own keys. NEVER commit `.env`.\n" + body, "utf-8")
            gi = d / ".gitignore"
            existing = gi.read_text("utf-8") if gi.exists() else ""
            if ".env" not in existing.split():
                gi.write_text(existing.rstrip() + "\n.env\n__pycache__/\n*.pyc\n", "utf-8")
        (OUT / "README.md").write_text(
            "# Downloads\n\nKey-free copies of the course code. Add your own keys (see the app's last page: "
            "**Downloads & API keys**). Never commit `.env` or `secrets.toml`.\n", "utf-8")

    # notebooks are built by tools/build_notebooks.py
    leaks = []
    for p in OUT.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".txt", ".csv", ".ipynb", ".example", ""}:
            t = p.read_text("utf-8", errors="ignore")
            if any(pat.search(t) for pat, _ in KEY_PATTERNS):
                leaks.append(p)
    if leaks:
        raise SystemExit(f"KEY-LIKE STRINGS STILL PRESENT: {leaks}")
    print("OK - downloads/ built, no key-like strings found.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    build(Path(args[0]), notebooks_only="--notebooks-only" in sys.argv)
