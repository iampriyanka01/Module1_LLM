"""
LangChain + Gemini logic: turns a chat history into a structured
RecommendationResponse (see models.py) using Gemini's structured output mode.

Gemini's structured-output mode (response_schema / with_structured_output) and its
native Google Search grounding tool cannot be used in the same API call. So for any
request that looks like it needs current/recent info (new releases, "what's out in
2026", etc.) we run two calls:
  1. A grounded, plain-text research call with Google Search enabled, to pull real
     current titles.
  2. A structured call that formats those (or the model's own knowledge, for
     evergreen requests) into a validated RecommendationResponse.
"""

from __future__ import annotations

import datetime as _dt
from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from models import RecommendationResponse
import tracing

# The new consolidated google-genai SDK (used by langchain-google-genai>=4.0) wants
# its own Tool/GoogleSearch objects rather than a bare {"google_search": {}} dict -
# the old dict form is a holdover from the legacy SDK and may get silently ignored.
try:
    from google.genai.types import GoogleSearch as _GoogleSearch
    from google.genai.types import Tool as _GenAITool

    _SEARCH_TOOL = _GenAITool(google_search=_GoogleSearch())
except ImportError:  # pragma: no cover - fallback for older SDKs
    _SEARCH_TOOL = {"google_search": {}}

TODAY_STR = _dt.date.today().strftime("%B %d, %Y")

BASE_SYSTEM_PROMPT = f"""\
You are Gemini Cinema Advisor, an enthusiastic and knowledgeable film critic \
helping a user find movies to watch. Today's date is {TODAY_STR}.

Guidelines:
- Ask brief clarifying questions if you don't have enough to go on yet (favorite \
genres, mood, eras, actors/directors they love, or things they hate).
- Once you have enough signal, recommend up to 5 real, existing movies that fit.
- Never invent a movie title. Only recommend real films you are confident exist.
- Give each recommendation a short, specific reason tied to what the user said - \
avoid generic blurbs.
- Keep the conversational "reply" field short (1-3 sentences).
- If the user says they hate a theme/genre, never recommend movies in it again.
"""

SYSTEM_TEMPLATE = BASE_SYSTEM_PROMPT + "\n\n{research_block}"

RESEARCH_SYSTEM_PROMPT = f"""\
You are a film research assistant. Today's date is {TODAY_STR}. Use Google Search \
to find real, currently-released or recently-announced movies that match the \
user's request below - especially if they're asking about new/recent/2025/2026 \
releases, which may be after your own training data.

Return a short plain-text list: title, year, and a one-sentence note per movie \
(cast, plot, genre - whatever is relevant). Only include movies you are confident \
are real, based on the search results. If the request doesn't need current info \
(e.g. it's about a classic era or genre in general), just say "No research needed."
"""


def _needs_research(user_input: str) -> bool:
    """Heuristic: does this request likely need up-to-date/current info?"""
    keywords = [
        "2024", "2025", "2026", "2027", "new", "recent", "latest", "just released",
        "this year", "currently", "now playing", "in theaters", "streaming now",
        "upcoming",
    ]
    lowered = user_input.lower()
    return any(k in lowered for k in keywords)


def _is_gemini_3_plus(model_name: str) -> bool:
    """Gemini 3.x+ models no longer accept/recommend temperature/top_p/top_k -
    they're tuned for their default reasoning settings, so we skip passing it."""
    return model_name.startswith("gemini-3")


def _make_llm(api_key: str, model_name: str, temperature: float | None = None):
    kwargs = {"model": model_name, "google_api_key": api_key}
    if temperature is not None and not _is_gemini_3_plus(model_name):
        kwargs["temperature"] = temperature
    return ChatGoogleGenerativeAI(**kwargs)


def _research_current_movies(api_key: str, user_input: str, model_name: str) -> str:
    """Grounded (Google Search) plain-text call to surface real current titles."""
    tracing.log(f"recommender._research_current_movies: building search-grounded LLM (model={model_name})")
    research_llm = _make_llm(api_key, model_name, temperature=0.3).bind_tools(
        [_SEARCH_TOOL]
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", RESEARCH_SYSTEM_PROMPT), ("human", "{user_input}")]
    )
    try:
        with tracing.timed("recommender._research_current_movies: Gemini call w/ google_search tool"):
            result = (prompt | research_llm).invoke({"user_input": user_input})
        content = result.content
        if isinstance(content, str):
            tracing.log(f"recommender._research_current_movies: got string content ({len(content)} chars)")
            return content
        if isinstance(content, list):
            # Some Gemini 3.x responses return content as a list of blocks
            # instead of a plain string - pull out any text parts.
            tracing.log(f"recommender._research_current_movies: got list content ({len(content)} blocks)")
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("text")
            ]
            return "\n".join(parts) if parts else str(content)
        return str(content)
    except Exception as e:
        # Grounding can fail (older model, unsupported region, etc.) - degrade
        # gracefully rather than breaking the whole recommendation flow, but keep
        # the real error visible for debugging.
        tracing.log_exception("recommender._research_current_movies")
        return f"No research available (search grounding failed: {e})"


def build_chain(api_key: str, model_name: str, temperature: float):
    """Create a LangChain runnable that outputs a validated RecommendationResponse."""
    llm = _make_llm(api_key, model_name, temperature=temperature)
    structured_llm = llm.with_structured_output(RecommendationResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder("history"),
            ("human", "{user_input}"),
        ]
    )

    return prompt | structured_llm


def history_to_lc_messages(history: List[Tuple[str, str]]):
    """Convert [(role, text), ...] pairs from Streamlit session state into LC messages."""
    lc_messages = []
    for role, text in history:
        if role == "user":
            lc_messages.append(HumanMessage(content=text))
        else:
            lc_messages.append(AIMessage(content=text))
    return lc_messages


def get_recommendation(
    api_key: str,
    history: List[Tuple[str, str]],
    user_input: str,
    model_name: str = "gemini-3.6-flash",
) -> Tuple[RecommendationResponse, Optional[str]]:
    """Run one turn of the conversation and return (structured_response, research_notes).

    research_notes is None when the research step wasn't triggered for this turn,
    so the caller can distinguish "didn't need to search" from "search ran".
    """
    tracing.log(f"recommender.get_recommendation: called with user_input={user_input!r}, model={model_name}")
    research_notes: Optional[str] = None
    research_block = ""

    needs_research = _needs_research(user_input)
    tracing.log(f"recommender.get_recommendation: _needs_research -> {needs_research}")

    if needs_research:
        research_notes = _research_current_movies(api_key, user_input, model_name)
        research_block = (
            "Up-to-date research notes from Google Search (use these for any "
            "current/recent movie titles - they are more reliable than your training "
            f"data for anything near {TODAY_STR}. If these notes explicitly say "
            "research failed or found nothing, tell the user honestly that you "
            "couldn't confirm current releases, rather than guessing older movies "
            f"and implying they're new):\n{research_notes}"
        )

    chain = build_chain(api_key, model_name, temperature=0.7)
    lc_history = history_to_lc_messages(history)
    # research_block/user_input are passed as template *variables* (not baked into
    # the template string itself), so any literal { or } inside them - e.g. from
    # search result text - is inserted verbatim instead of being re-parsed as a
    # template placeholder, which is what caused the earlier f-string crash.
    try:
        with tracing.timed("recommender.get_recommendation: structured Gemini call"):
            result = chain.invoke(
                {"history": lc_history, "user_input": user_input, "research_block": research_block}
            )
    except Exception:
        tracing.log_exception("recommender.get_recommendation (structured call)")
        raise

    tracing.log(
        f"recommender.get_recommendation: got {len(result.movies)} movie(s) back, "
        f"follow_up={'yes' if result.follow_up_question else 'no'}"
    )
    return result, research_notes