import os
import json
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from tools import (
    ProfileManager,
    search_movies,
    get_movie_genres,
    discover_movies,
    get_movie_details,
)


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

MODEL = "openai/gpt-oss-20b"

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w200"

st.set_page_config(
    page_title="CineBot",
    page_icon="🎬",
    layout="wide",
)


# =========================================================
# SYSTEM PROMPT (unchanged from the CLI version)
# =========================================================

SYSTEM_PROMPT = """
You are CineBot, an intelligent and friendly personal
movie recommendation assistant.

Your job is to understand the user's movie taste and
recommend real movies that they are likely to enjoy.

You have access to:

1. User profile tools
2. TMDB movie search
3. TMDB movie discovery
4. TMDB movie details
5. TMDB genre information

=========================================================
USER PROFILE
=========================================================

Gradually learn the user's movie preferences.

Whenever the user reveals useful information about:

- Their name
- Movies they love
- Movies they have watched
- Genres they enjoy
- Genres they dislike
- Preferred languages
- Favorite actors
- Favorite directors
- Whether they like hidden gems

call update_user_profile.

IMPORTANT: preferred_languages must always be full language
names in English, e.g. "Hindi", "English", "Korean",
"Japanese", "French" — never two-letter ISO codes like
"hi" or "en". ISO codes are only used internally as the
`language` argument to discover_movies, never inside the
user profile.

Do NOT ask the user to explicitly tell you to remember
something.

If they say "I love Interstellar", update their profile.
If they say "I don't like horror", update their profile.
If they say "I've already seen Inception", update
watched_movies.

=========================================================
CONVERSATION
=========================================================

During onboarding:
- Ask ONE question at a time.
- Do not interrogate the user.
- Keep the conversation natural.
- Don't ask for information already known.
- Try to understand their taste through conversation.

=========================================================
MOVIE DATA
=========================================================

TMDB is the source of truth for movie information.
Whenever possible, use TMDB tools instead of relying on
your own knowledge for titles, dates, ratings,
descriptions, cast, directors, genres, or discovery.
Never invent movie information.

=========================================================
RECOMMENDATIONS
=========================================================

When the user asks for movie recommendations:
1. Check their user profile.
2. Understand their preferences.
3. Use TMDB to find candidate movies.
4. Avoid movies they have already watched.
5. Avoid genres they dislike.
6. Prefer movies matching their favorite genres, actors,
   directors, languages and preferences.
7. Explain WHY each movie matches their taste.

Try to recommend 3-5 movies.

=========================================================
TMDB TOOLS
=========================================================

Use search_movies when the user mentions a specific movie
or you need to find one by title.

Use get_movie_details when you need detailed info,
cast/director, or more detail on a movie already found.

Use get_movie_genres when you need TMDB genre IDs.

Use discover_movies when the user asks for
recommendations or describes a genre, year, rating,
language, or other preference.

=========================================================
IMPORTANT
=========================================================

Do not recommend a movie simply because you personally
know it. Use TMDB to obtain real candidates, then use the
user's profile and reasoning to pick the best fit.

Keep answers conversational rather than returning raw JSON.
"""


# =========================================================
# AI EVALUATOR SYSTEM PROMPT
# =========================================================
#
# A second, independent LLM call that grades CineBot's own
# reply after each turn (an "LLM-as-judge" pattern). This is
# NOT part of the user-facing conversation — it never sees
# the tool machinery, only the user's message and CineBot's
# final reply, and it scores that reply against a rubric.
# =========================================================

EVAL_SYSTEM_PROMPT = """
You are a strict, impartial evaluator grading a movie
recommendation chatbot named CineBot.

You will be shown:
1. The user's message
2. A snapshot of what CineBot currently knows about the
   user (their profile)
3. CineBot's reply

Score the REPLY (not the user's message) on these criteria,
each from 1 (poor) to 5 (excellent):

- relevance: does the reply directly address what the user
  said or asked, without going off-topic?
- personalization: does it make use of the user's known
  preferences (genres, actors, languages, watched movies)
  where relevant, instead of generic suggestions?
- groundedness: does it avoid inventing movie facts (titles,
  ratings, cast, release years)? A reply that only makes
  claims consistent with real, verifiable movie data scores
  high. A reply that seems to fabricate specific facts scores
  low.
- clarity: is it clear, concise, and conversational — not
  raw JSON, not a wall of text, not robotic?

Then give an "overall" score from 1-5 (your holistic judgment,
not simply the average).

Respond with ONLY a single JSON object, no prose before or
after it, in exactly this shape:

{
  "relevance": <int 1-5>,
  "personalization": <int 1-5>,
  "groundedness": <int 1-5>,
  "clarity": <int 1-5>,
  "overall": <int 1-5>,
  "verdict": "<one short sentence summarizing the judgment>",
  "issues": ["<short flag>", "..."]
}

"issues" should be an empty array if there are no problems.
Be honest and critical — do not default to high scores.
"""


# =========================================================
# TOOL SCHEMAS (same shape as the CLI version)
# =========================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": (
                "Update the user's movie preference profile. "
                "Call this whenever the user reveals information "
                "about their movie preferences."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "User's name"},
                    "favorite_movies": {"type": "array", "items": {"type": "string"}, "description": "Movies the user loves"},
                    "favorite_genres": {"type": "array", "items": {"type": "string"}, "description": "Genres the user likes"},
                    "disliked_genres": {"type": "array", "items": {"type": "string"}, "description": "Genres the user dislikes"},
                    "preferred_languages": {"type": "array", "items": {"type": "string"}, "description": "Preferred movie languages"},
                    "favorite_actors": {"type": "array", "items": {"type": "string"}, "description": "Favorite actors"},
                    "favorite_directors": {"type": "array", "items": {"type": "string"}, "description": "Favorite directors"},
                    "watched_movies": {"type": "array", "items": {"type": "string"}, "description": "Movies the user has watched"},
                    "likes_hidden_gems": {"type": "boolean", "description": "Whether the user likes underrated or lesser-known movies"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "Get the user's current movie preference profile. Use before making personalized recommendations.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_movies",
            "description": "Search TMDB for movies by title or keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Movie title or search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_genres",
            "description": "Get the list of movie genres and their TMDB IDs.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discover_movies",
            "description": (
                "Discover movies from TMDB using filters. Use for "
                "recommendations. genre_ids should contain TMDB "
                "genre IDs. sort_by can be popularity.desc, "
                "vote_average.desc, or primary_release_date.desc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "genre_ids": {"type": "array", "items": {"type": "integer"}, "description": "TMDB genre IDs"},
                    "year": {"type": "integer", "description": "Release year"},
                    "min_rating": {"type": "number", "description": "Minimum TMDB rating"},
                    "language": {"type": "string", "description": "Original language ISO code"},
                    "sort_by": {"type": "string", "description": "TMDB sorting method"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": "Get detailed information about a movie from TMDB (cast, director, genres, runtime, etc).",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {"type": "integer", "description": "TMDB movie ID"},
                },
                "required": ["movie_id"],
            },
        },
    },
]


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    st.session_state.display_messages = []

if "profile_manager" not in st.session_state:
    st.session_state.profile_manager = ProfileManager()

if "last_movies" not in st.session_state:
    st.session_state.last_movies = []

if "trace_log" not in st.session_state:
    # Each entry: {id, timestamp, turn, function, arguments, result,
    # success, duration_ms, raw_arguments, error}
    st.session_state.trace_log = []

if "trace_options" not in st.session_state:
    st.session_state.trace_options = {
        "enabled": True,
        "show_raw_json": False,
        "auto_expand_latest": True,
    }

if "eval_log" not in st.session_state:
    # Each entry: {message_index, timestamp, user_message, reply,
    # scores: {relevance, personalization, groundedness, clarity, overall},
    # verdict, issues, success, error}
    st.session_state.eval_log = []

if "eval_options" not in st.session_state:
    st.session_state.eval_options = {
        "enabled": True,
        "auto_expand_latest": True,
    }


# =========================================================
# SIDEBAR — API KEYS + PROFILE
# =========================================================

with st.sidebar:
    st.title("🎬 CineBot")
    st.caption("Your personal movie recommendation assistant")

    with st.expander("🔑 API keys", expanded=not (os.getenv("GROQ_API_KEY") and os.getenv("TMDB_API_KEY"))):
        groq_key_input = st.text_input(
            "GROQ_API_KEY",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
        )
        tmdb_key_input = st.text_input(
            "TMDB_API_KEY (v4 read access token)",
            value=os.getenv("TMDB_API_KEY", ""),
            type="password",
            help="Must be the long v4 'Read Access Token' (JWT-style string) from TMDB Settings → API, not the short v3 API key.",
        )

        key_col1, key_col2 = st.columns(2)

        with key_col1:
            if st.button("Save keys", use_container_width=True):
                # .strip() guards against a trailing newline/space
                # sneaking in from copy-paste, which silently causes
                # a 401 that looks identical to a wrong key.
                groq_clean = groq_key_input.strip()
                tmdb_clean = tmdb_key_input.strip()

                os.environ["GROQ_API_KEY"] = groq_clean
                os.environ["TMDB_API_KEY"] = tmdb_clean

                # tools.py read the key at import time, so patch it live too
                import tools as _tools_module
                _tools_module.TMDB_API_KEY = tmdb_clean
                _tools_module.TMDB_HEADERS = {
                    "Authorization": f"Bearer {tmdb_clean}",
                    "accept": "application/json",
                }
                st.success("Saved for this session.")

        with key_col2:
            if st.button("Test TMDB key", use_container_width=True):
                import tools as _tools_module
                try:
                    _tools_module.tmdb_request("/genre/movie/list", {"language": "en"})
                    st.success("✅ TMDB key works.")
                except Exception as test_error:
                    err_text = str(test_error)
                    if "401" in err_text:
                        st.error(
                            "❌ 401 Unauthorized. This almost always means "
                            "you pasted the v3 API key instead of the v4 "
                            "Read Access Token — grab the long JWT-style "
                            "token from TMDB Settings → API instead."
                        )
                    else:
                        st.error(f"❌ {err_text}")

    st.divider()

    st.subheader("🧠 Your profile")

    profile = st.session_state.profile_manager.get_user_profile()

    if profile.get("name"):
        st.markdown(f"**Name:** {profile['name']}")

    def _chips(label, items):
        if items:
            st.markdown(f"**{label}:** " + ", ".join(items))

    # ISO 639-1 -> full name, in case the model stores a code
    # instead of a language name in preferred_languages (its
    # instructions say not to, but this keeps the sidebar
    # readable even if it slips up).
    _ISO_LANGUAGE_NAMES = {
        "en": "English", "hi": "Hindi", "fr": "French", "es": "Spanish",
        "de": "German", "it": "Italian", "ja": "Japanese", "ko": "Korean",
        "zh": "Chinese", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
        "ml": "Malayalam", "mr": "Marathi", "bn": "Bengali", "pa": "Punjabi",
        "gu": "Gujarati", "ur": "Urdu", "ar": "Arabic", "ru": "Russian",
        "pt": "Portuguese", "tr": "Turkish", "th": "Thai", "vi": "Vietnamese",
    }

    def _friendly_language(lang):
        return _ISO_LANGUAGE_NAMES.get(lang.lower(), lang) if len(lang) <= 2 else lang

    _chips("Favorite movies", profile["favorite_movies"])
    _chips("Favorite genres", profile["favorite_genres"])
    _chips("Disliked genres", profile["disliked_genres"])
    _chips(
        "Preferred languages",
        [_friendly_language(l) for l in profile["preferred_languages"]],
    )
    _chips("Favorite actors", profile["favorite_actors"])
    _chips("Favorite directors", profile["favorite_directors"])
    _chips("Watched", profile["watched_movies"])

    if profile.get("likes_hidden_gems") is not None:
        st.markdown(
            f"**Likes hidden gems:** "
            f"{'Yes' if profile['likes_hidden_gems'] else 'No'}"
        )

    if not any([
        profile.get("name"),
        profile["favorite_movies"],
        profile["favorite_genres"],
    ]):
        st.caption("Chat with CineBot and it'll fill this in automatically.")

    with st.expander("Raw profile JSON"):
        st.json(profile)

    st.divider()

    # -----------------------------------------------------
    # TRACING
    # -----------------------------------------------------

    st.subheader("🔍 Tracing")

    st.session_state.trace_options["enabled"] = st.toggle(
        "Enable tracing",
        value=st.session_state.trace_options["enabled"],
        help="Record every tool/function call the agent makes so you can inspect arguments, results, and timing.",
    )

    with st.expander("Trace options", expanded=False):
        st.session_state.trace_options["show_raw_json"] = st.checkbox(
            "Show raw tool-call JSON (unparsed model output)",
            value=st.session_state.trace_options["show_raw_json"],
        )
        st.session_state.trace_options["auto_expand_latest"] = st.checkbox(
            "Auto-expand most recent call",
            value=st.session_state.trace_options["auto_expand_latest"],
        )

    trace_log = st.session_state.trace_log

    if trace_log:

        # ---- summary metrics ----
        total_calls = len(trace_log)
        failed_calls = sum(1 for t in trace_log if not t["success"])
        avg_ms = sum(t["duration_ms"] for t in trace_log) / total_calls

        m1, m2, m3 = st.columns(3)
        m1.metric("Calls", total_calls)
        m2.metric("Failed", failed_calls)
        m3.metric("Avg ms", f"{avg_ms:.0f}")

        # ---- per-function call counts ----
        counts = {}
        for t in trace_log:
            counts[t["function"]] = counts.get(t["function"], 0) + 1

        with st.expander("Calls by function", expanded=False):
            for fn_name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
                st.markdown(f"- `{fn_name}` — {count}")

        # ---- filter ----
        available_fns = sorted(counts.keys())
        selected_fns = st.multiselect(
            "Filter by function",
            options=available_fns,
            default=[],
            placeholder="All functions",
        )

        filtered = [
            t for t in trace_log
            if not selected_fns or t["function"] in selected_fns
        ]

        st.caption(f"Showing {len(filtered)} of {total_calls} call(s), most recent first.")

        for i, entry in enumerate(reversed(filtered)):
            status_icon = "✅" if entry["success"] else "❌"
            label = (
                f"{status_icon} `{entry['function']}` "
                f"· {entry['duration_ms']:.0f}ms · "
                f"turn {entry['turn']} · {entry['timestamp']}"
            )
            expanded = st.session_state.trace_options["auto_expand_latest"] and i == 0
            with st.expander(label, expanded=expanded):
                st.markdown("**Arguments**")
                st.json(entry["arguments"])

                if st.session_state.trace_options["show_raw_json"]:
                    st.markdown("**Raw model output (arguments)**")
                    st.code(entry["raw_arguments"], language="json")

                st.markdown("**Result**" if entry["success"] else "**Error**")
                if entry["success"]:
                    st.json(entry["result"])
                else:
                    st.error(entry["result"])

        if st.button("🧹 Clear trace log"):
            st.session_state.trace_log = []
            st.rerun()

    else:
        st.caption("No tool calls yet — chat with CineBot to see traces here.")

    st.divider()

    # -----------------------------------------------------
    # AI EVALUATOR
    # -----------------------------------------------------

    st.subheader("🧪 AI evaluator")

    st.session_state.eval_options["enabled"] = st.toggle(
        "Enable AI evaluation",
        value=st.session_state.eval_options["enabled"],
        help="After each reply, a second LLM call grades it for relevance, personalization, groundedness, and clarity.",
    )

    with st.expander("Evaluator options", expanded=False):
        st.session_state.eval_options["auto_expand_latest"] = st.checkbox(
            "Auto-expand most recent evaluation",
            value=st.session_state.eval_options["auto_expand_latest"],
        )

    eval_log = st.session_state.eval_log

    if eval_log:

        successful = [e for e in eval_log if e["success"]]
        failed = len(eval_log) - len(successful)

        if successful:
            avg = lambda key: sum(e["scores"][key] for e in successful) / len(successful)

            m1, m2 = st.columns(2)
            m1.metric("Avg overall", f"{avg('overall'):.1f}/5")
            m2.metric("Evaluated", f"{len(successful)}/{len(eval_log)}")

            with st.expander("Score breakdown (avg)", expanded=False):
                st.markdown(f"- Relevance: **{avg('relevance'):.1f}/5**")
                st.markdown(f"- Personalization: **{avg('personalization'):.1f}/5**")
                st.markdown(f"- Groundedness: **{avg('groundedness'):.1f}/5**")
                st.markdown(f"- Clarity: **{avg('clarity'):.1f}/5**")

        if failed:
            st.caption(f"⚠️ {failed} evaluation(s) failed to run — see details below.")

        for i, entry in enumerate(reversed(eval_log)):
            if entry["success"]:
                overall = entry["scores"]["overall"]
                score_icon = "🟢" if overall >= 4 else ("🟡" if overall >= 3 else "🔴")
                label = f"{score_icon} {overall}/5 · {entry['timestamp']} · {entry['user_message'][:30]}"
            else:
                label = f"❌ evaluation failed · {entry['timestamp']}"

            expanded = st.session_state.eval_options["auto_expand_latest"] and i == 0
            with st.expander(label, expanded=expanded):
                if entry["success"]:
                    s = entry["scores"]
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Relevance:** {s['relevance']}/5")
                    c1.markdown(f"**Personalization:** {s['personalization']}/5")
                    c2.markdown(f"**Groundedness:** {s['groundedness']}/5")
                    c2.markdown(f"**Clarity:** {s['clarity']}/5")
                    if entry["verdict"]:
                        st.markdown(f"**Verdict:** {entry['verdict']}")
                    if entry["issues"]:
                        st.markdown("**Issues flagged:**")
                        for issue in entry["issues"]:
                            st.markdown(f"- {issue}")
                else:
                    st.error(entry["error"] or "Unknown evaluator error")

                st.markdown("**User message**")
                st.caption(entry["user_message"])
                st.markdown("**Reply evaluated**")
                st.caption(entry["reply"])

        if st.button("🧹 Clear evaluation log"):
            st.session_state.eval_log = []
            st.rerun()

    else:
        st.caption("No evaluations yet — CineBot's replies get graded here automatically.")

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 New chat"):
            st.session_state.messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            st.session_state.display_messages = []
            st.session_state.last_movies = []
            st.session_state.trace_log = []
            st.session_state.eval_log = []
            st.rerun()
    with col_b:
        if st.button("🗑️ Clear profile"):
            st.session_state.profile_manager.reset()
            st.rerun()


# =========================================================
# TOOL DISPATCH (bound to this session's profile manager)
# =========================================================

def get_available_functions():
    pm = st.session_state.profile_manager
    return {
        "update_user_profile": pm.update_user_profile,
        "get_user_profile": pm.get_user_profile,
        "search_movies": search_movies,
        "get_movie_genres": get_movie_genres,
        "discover_movies": discover_movies,
        "get_movie_details": get_movie_details,
    }


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Please add your GROQ_API_KEY in the sidebar.")
        st.stop()
    return Groq(api_key=api_key)


def run_chat(status_box):
    """
    Same agent loop as the CLI version's run_chat(), adapted to
    update a Streamlit status box with tool-call progress, record
    a trace entry for every tool call (visible in the sidebar),
    and collect any movies surfaced by TMDB tool calls so they can
    be rendered as a poster gallery.
    """

    client = get_groq_client()
    available_functions = get_available_functions()
    turn_movies = []
    max_retries_per_turn = 2
    llm_turn = 0  # counts round-trips to the LLM within this run_chat() call

    while True:

        llm_turn += 1

        # ------------------------------------------------------
        # Groq occasionally rejects a completion outright with a
        # 400 "tool_use_failed" error when the model emits
        # malformed JSON for a tool call's arguments (this shows
        # up as failed_generation in the error body). This is a
        # model-side generation glitch, not something we caused,
        # so retry the request a couple of times before giving up
        # instead of failing the whole turn.
        # ------------------------------------------------------

        for attempt in range(max_retries_per_turn + 1):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=st.session_state.messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.7,
                )
                break
            except Exception as error:
                is_tool_use_failed = "tool_use_failed" in str(error)
                if is_tool_use_failed and attempt < max_retries_per_turn:
                    status_box.write(
                        f"⚠️ Model returned malformed tool call JSON "
                        f"(attempt {attempt + 1}/{max_retries_per_turn + 1}) "
                        f"— retrying..."
                    )
                    continue
                raise

        response_message = response.choices[0].message
        st.session_state.messages.append(response_message)

        if not response_message.tool_calls:
            return response_message.content, turn_movies

        for tool_call in response_message.tool_calls:

            function_name = tool_call.function.name
            status_box.write(f"🔧 Calling `{function_name}`...")

            raw_args = tool_call.function.arguments or "{}"
            function_args = {}
            start_time = time.perf_counter()
            success = True

            try:
                try:
                    function_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    # Some open-weight models occasionally emit malformed
                    # JSON for zero-argument tool calls (e.g. `{""}"}`
                    # instead of `{}`). If the schema for this function
                    # takes no required arguments, just fall back to an
                    # empty call instead of failing the whole turn.
                    function_args = {}
                    status_box.write(
                        f"⚠️ `{function_name}` sent malformed arguments "
                        f"({raw_args!r}) — retrying with no arguments."
                    )

                function_to_call = available_functions.get(function_name)

                if function_to_call is None:
                    success = False
                    tool_result = {
                        "success": False,
                        "error": f"Unknown function: {function_name}",
                    }
                else:
                    tool_result = function_to_call(**function_args)

                if isinstance(tool_result, dict) and "results" in tool_result:
                    turn_movies.extend(tool_result["results"])

            except TypeError as error:
                # function_args parsed as valid JSON but had unexpected
                # keys/types for this function's signature.
                success = False
                tool_result = f"Invalid arguments for {function_name}: {error}"
                status_box.write(f"✗ `{function_name}` failed: {error}")

            except Exception as error:
                success = False
                tool_result = str(error)
                status_box.write(f"✗ `{function_name}` failed: {error}")

            duration_ms = (time.perf_counter() - start_time) * 1000

            if st.session_state.trace_options["enabled"]:
                st.session_state.trace_log.append({
                    "id": tool_call.id,
                    "turn": llm_turn,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "function": function_name,
                    "arguments": function_args,
                    "raw_arguments": raw_args,
                    "result": tool_result,
                    "success": success,
                    "duration_ms": duration_ms,
                })

            st.session_state.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    tool_result if success else {"success": False, "error": tool_result},
                    default=str,
                ),
            })


# =========================================================
# AI EVALUATOR
# =========================================================

def run_evaluator(user_message, reply, message_index):
    """
    Runs a second, independent LLM call ("LLM-as-judge") that
    grades CineBot's reply against a small rubric, and appends
    the result to st.session_state.eval_log. Failures here never
    interrupt the main chat — they're recorded as a failed entry
    so the sidebar can show what went wrong.
    """

    entry = {
        "message_index": message_index,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "user_message": user_message,
        "reply": reply,
        "success": False,
        "scores": {},
        "verdict": "",
        "issues": [],
        "error": None,
    }

    try:
        client = get_groq_client()
        profile_snapshot = st.session_state.profile_manager.get_user_profile()

        eval_user_content = (
            f"USER PROFILE:\n{json.dumps(profile_snapshot, default=str)}\n\n"
            f"USER MESSAGE:\n{user_message}\n\n"
            f"CINEBOT REPLY:\n{reply}"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": eval_user_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Some models/routes may reject response_format — retry
            # without it and rely on prompt instructions alone.
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": eval_user_content},
                ],
                temperature=0.0,
            )

        raw_content = response.choices[0].message.content or "{}"

        # Judges occasionally wrap JSON in ```json fences despite
        # instructions — strip those before parsing.
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        parsed = json.loads(cleaned)

        scores = {
            "relevance": int(parsed.get("relevance", 0)),
            "personalization": int(parsed.get("personalization", 0)),
            "groundedness": int(parsed.get("groundedness", 0)),
            "clarity": int(parsed.get("clarity", 0)),
            "overall": int(parsed.get("overall", 0)),
        }

        entry.update({
            "success": True,
            "scores": scores,
            "verdict": parsed.get("verdict", ""),
            "issues": parsed.get("issues", []) or [],
        })

    except Exception as error:
        entry["error"] = str(error)

    if st.session_state.eval_options["enabled"]:
        st.session_state.eval_log.append(entry)

    return entry


# =========================================================
# MOVIE POSTER GALLERY
# =========================================================

def render_movie_gallery(movies):
    if not movies:
        return

    seen = set()
    unique_movies = []
    for m in movies:
        if m.get("id") not in seen:
            seen.add(m.get("id"))
            unique_movies.append(m)

    cols = st.columns(min(len(unique_movies), 5) or 1)

    for i, movie in enumerate(unique_movies[:10]):
        col = cols[i % len(cols)]
        with col:
            poster = movie.get("poster_path")
            if poster:
                st.image(f"{TMDB_IMAGE_BASE}{poster}", use_container_width=True)
            title = movie.get("title") or "Untitled"
            year = (movie.get("release_date") or "")[:4]
            st.caption(f"**{title}** ({year})" if year else f"**{title}**")
            rating = movie.get("rating")
            if rating:
                st.caption(f"⭐ {rating:.1f}")


# =========================================================
# MAIN CHAT UI
# =========================================================

st.title("🎬 CineBot")
st.caption("Tell me about movies you love — I'll learn your taste and recommend real films from TMDB.")

profile = st.session_state.profile_manager.get_user_profile()

# -----------------------------------------------------
# NAME GATE — name is mandatory before chatting starts
# -----------------------------------------------------

if not profile.get("name"):

    st.info("👋 Before we start, what should CineBot call you?")

    with st.form("name_gate_form", clear_on_submit=False):
        name_input = st.text_input(
            "Your name",
            placeholder="e.g. Priyanka",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Start chatting", use_container_width=True)

        if submitted:
            if name_input.strip():
                st.session_state.profile_manager.update_user_profile(
                    name=name_input.strip()
                )
                st.rerun()
            else:
                st.error("Please enter your name to continue — it's required.")

    st.stop()


def _score_badge(eval_entry):
    if not eval_entry or not eval_entry.get("success"):
        return None
    overall = eval_entry["scores"]["overall"]
    icon = "🟢" if overall >= 4 else ("🟡" if overall >= 3 else "🔴")
    return f"{icon} Eval: {overall}/5"


if not st.session_state.display_messages:
    with st.chat_message("assistant"):
        st.markdown(
            f"Hey {profile['name']}! 👋 Let's get to know your movie taste.\n\n"
            "What's one movie you absolutely loved?"
        )

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("movies"):
            render_movie_gallery(msg["movies"])
        badge = _score_badge(msg.get("eval"))
        if badge:
            st.caption(badge)

user_input = st.chat_input("Type a message...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.display_messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_box = st.status("Thinking...", expanded=False)
        try:
            reply, movies = run_chat(status_box)
            status_box.update(label="Done", state="complete")
        except Exception as e:
            status_box.update(label="Error", state="error")
            reply, movies = f"Something went wrong: {e}", []

        st.markdown(reply)
        render_movie_gallery(movies)

        eval_entry = None
        if st.session_state.eval_options["enabled"]:
            with st.spinner("Grading response..."):
                eval_entry = run_evaluator(
                    user_message=user_input,
                    reply=reply,
                    message_index=len(st.session_state.display_messages),
                )
            badge = _score_badge(eval_entry)
            if badge:
                st.caption(badge)

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": reply,
        "movies": movies,
        "eval": eval_entry,
    })