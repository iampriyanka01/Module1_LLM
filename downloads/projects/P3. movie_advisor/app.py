"""
Gemini Cinema Advisor - a Streamlit chat app that recommends movies using
Gemini (via LangChain, with Pydantic-structured output) and enriches each
recommendation with poster/rating data from OMDb.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from models import RecommendationResponse
from omdb_client import OmdbClient
from recommender import get_recommendation
from evaluator import get_evaluation
import tracing

load_dotenv()

st.set_page_config(page_title="Gemini Cinema Advisor", page_icon="🍿", layout="centered")


# --------------------------------------------------------------------------- #
# Sidebar: API keys + settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    gemini_key = st.text_input(
        "Gemini API key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Get one at https://aistudio.google.com/apikey",
    )
    omdb_key = st.text_input(
        "OMDb API key",
        value=os.getenv("OMDB_API_KEY", ""),
        type="password",
        help="Get a free key at http://www.omdbapi.com/apikey.aspx",
    )
    model_name = st.selectbox(
        "Gemini model",
        ["gemini-3.6-flash", "gemini-3.5-flash"],
        index=0,
        help="3.6 Flash is newer, faster, and cheaper per token than 3.5 Flash.",
    )
    show_debug = st.checkbox(
        "Show debug info (research notes + execution trace)",
        value=False,
        help="See exactly what happened for the last message: whether Google "
        "Search grounding ran, what each step returned, and any errors, in the "
        "order they occurred - useful for troubleshooting.",
    )
    evaluate_quality = st.checkbox(
        "Evaluate recommendation quality (AI judge)",
        value=True,
        help="Runs a second, independent Gemini call that critically scores how "
        "well each recommendation fits your request. Uses one extra API call "
        "per turn with recommendations.",
    )
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "chat_history" not in st.session_state:
    # (role, text) pairs fed back into the LLM as conversational memory
    st.session_state.chat_history = []
if "display_history" not in st.session_state:
    # richer entries actually rendered on screen: {"role", "text", "movies"?}
    st.session_state.display_history = [
        {
            "role": "assistant",
            "text": (
                "Hello! I am your cinema assistant. Tell me about your favorite "
                "genres, movie eras, or themes you absolutely hate, and we can "
                "find your next watch!"
            ),
            "movies": [],
        }
    ]


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🍿 Gemini Cinema Advisor")
st.caption("Chat with your personal AI film critic. Share your preferences, and get real recommendations.")
st.divider()


# --------------------------------------------------------------------------- #
# Render existing conversation
# --------------------------------------------------------------------------- #
def render_movie_card(movie_dict: dict, omdb_client: OmdbClient | None):
    details = None
    lookup_error = None
    if omdb_client is None:
        lookup_error = "No OMDb API key set — add one in the sidebar to load posters."
    else:
        try:
            details = omdb_client.get_movie(movie_dict["title"], movie_dict.get("year"))
            if not details.found:
                lookup_error = f"OMDb: {details.error}"
            elif not details.poster:
                lookup_error = "OMDb found the movie but has no poster on file for it."
        except Exception as e:
            lookup_error = f"OMDb lookup failed: {e}"

    with st.container(border=True):
        cols = st.columns([1, 3])
        with cols[0]:
            image_rendered = False
            if details and details.found and details.poster:
                try:
                    st.image(details.poster, use_container_width=True)
                    image_rendered = True
                except Exception:
                    lookup_error = "Poster image failed to load."
            if not image_rendered:
                st.markdown("🎬")
                if lookup_error:
                    st.caption(lookup_error)
        with cols[1]:
            year_str = f" ({movie_dict['year']})" if movie_dict.get("year") else ""
            st.markdown(f"**{movie_dict['title']}{year_str}**")
            if details and details.found:
                meta_bits = [b for b in [details.genre, details.runtime] if b]
                if meta_bits:
                    st.caption(" • ".join(meta_bits))
                if details.imdb_rating and details.imdb_rating != "N/A":
                    st.markdown(f"⭐ {details.imdb_rating}/10 on IMDb")
            st.write(movie_dict.get("reason", ""))

            eval_score = movie_dict.get("eval_score")
            if eval_score is not None:
                score_color = "green" if eval_score >= 8 else "orange" if eval_score >= 5 else "red"
                st.progress(
                    eval_score / 10,
                    text=f":{score_color}[AI fit score: {eval_score}/10]",
                )
                if movie_dict.get("eval_justification"):
                    st.caption(f"🧑‍⚖️ {movie_dict['eval_justification']}")


def render_assistant_entry(entry: dict, omdb_client: OmdbClient | None):
    """Renders one assistant turn: reply text, overall AI-judge verdict (if any),
    and movie cards. Shared by both the live turn and history replay so the two
    never drift apart."""
    st.write(entry["text"])
    if entry.get("overall_score") is not None:
        score = entry["overall_score"]
        icon = "🟢" if score >= 8 else "🟡" if score >= 5 else "🔴"
        st.caption(f"{icon} **AI judge - overall fit: {score}/10.** {entry.get('overall_feedback', '')}")
    for movie in entry.get("movies", []):
        render_movie_card(movie, omdb_client)


for entry in st.session_state.display_history:
    with st.chat_message(entry["role"], avatar="🍿" if entry["role"] == "assistant" else None):
        if entry["role"] == "assistant":
            omdb_client = OmdbClient(omdb_key) if omdb_key else None
            render_assistant_entry(entry, omdb_client)
        else:
            st.write(entry["text"])


# --------------------------------------------------------------------------- #
# Chat input
# --------------------------------------------------------------------------- #
user_input = st.chat_input("Ask about a movie, share your mood, or request a recommendation...")

if user_input:
    if not gemini_key:
        st.error("Please add your Gemini API key in the sidebar to chat.")
        st.stop()

    tracing.start_trace()
    tracing.log(f"app.py: new user message received: {user_input!r} (model={model_name})")

    st.session_state.display_history.append({"role": "user", "text": user_input, "movies": []})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant", avatar="🍿"):
        with st.spinner("Thinking about what you'd love..."):
            try:
                result, research_notes = get_recommendation(
                    api_key=gemini_key,
                    history=st.session_state.chat_history,
                    user_input=user_input,
                    model_name=model_name,
                )
            except Exception as e:
                tracing.log_exception("app.py: get_recommendation call")
                st.error(f"Something went wrong talking to Gemini: {e}")
                if show_debug:
                    with st.expander("🧵 Execution trace (debug)"):
                        st.text(tracing.get_trace_text())
                st.stop()

        omdb_client = OmdbClient(omdb_key) if omdb_key else None
        movie_dicts = [m.model_dump() for m in result.movies]

        overall_score = None
        overall_feedback = None
        eval_error = None
        if evaluate_quality and result.movies:
            with st.spinner("Having the AI judge score these..."):
                evaluation, eval_error = get_evaluation(
                    api_key=gemini_key,
                    user_request=user_input,
                    movies=result.movies,
                    model_name=model_name,
                )
            if evaluation:
                overall_score = evaluation.overall_score
                overall_feedback = evaluation.overall_feedback
                eval_map = {
                    me.title.strip().lower(): me for me in evaluation.movie_evaluations
                }
                for md in movie_dicts:
                    match = eval_map.get(md["title"].strip().lower())
                    if match:
                        md["eval_score"] = match.relevance_score
                        md["eval_justification"] = match.justification

        assistant_entry = {
            "role": "assistant",
            "text": result.reply,
            "movies": movie_dicts,
            "overall_score": overall_score,
            "overall_feedback": overall_feedback,
        }
        render_assistant_entry(assistant_entry, omdb_client)

        if result.follow_up_question:
            st.info(result.follow_up_question)
        if evaluate_quality and result.movies and overall_score is None:
            if eval_error and show_debug:
                st.caption(f"⚠️ AI judge unavailable this turn: {eval_error}")
            else:
                st.caption("⚠️ AI judge unavailable this turn.")
        if show_debug:
            with st.expander("🔍 Search research notes (debug)"):
                st.text(research_notes if research_notes is not None else "(research step wasn't triggered this turn)")
            with st.expander("🧵 Execution trace (debug)"):
                st.text(tracing.get_trace_text())

    # Update memory + display history
    st.session_state.chat_history.append(("user", user_input))
    assistant_text = result.reply + (f" {result.follow_up_question}" if result.follow_up_question else "")
    st.session_state.chat_history.append(("assistant", assistant_text))
    st.session_state.display_history.append(assistant_entry)