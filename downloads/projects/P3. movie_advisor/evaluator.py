"""
An independent "judge" call that scores how well each recommended movie (and the
set as a whole) actually fits what the user asked for.

This is intentionally a *separate* Gemini call rather than asking the recommender
to grade its own homework in the same response - a fresh, critically-instructed
call gives a more honest signal than self-reported confidence.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from models import EvaluationResponse, MovieRecommendation
from recommender import _make_llm  # reuse the gemini-3-aware LLM builder
import tracing

EVALUATOR_SYSTEM_PROMPT = """\
You are a strict, independent film-recommendation critic. You did NOT make these \
recommendations - another assistant did, and your job is to judge them honestly.

For the user's request below and the list of recommended movies, score:
- Each movie individually (1-10): does it genuinely fit what the user asked for? \
A generic "good movie" that ignores their specific ask should score low, even if \
it's a great film in general.
- The set overall (1-10): does the combination cover the request well, or is it \
repetitive/off-target/too safe?

Be genuinely critical - do not default to high scores. A perfect fit is rare.
Keep justifications to one short, specific sentence each.
"""

# The user's request and the movie list are passed in as template *variables*,
# not concatenated into the prompt string, so any literal { or } in a movie title
# or reason doesn't get mis-parsed as a template placeholder.
EVALUATOR_TEMPLATE = EVALUATOR_SYSTEM_PROMPT + "\n\nUser's request: {user_request}\n\nRecommended movies:\n{movie_list}"


def _movies_to_plain_text(movies: List[MovieRecommendation]) -> str:
    lines = []
    for m in movies:
        year_str = f" ({m.year})" if m.year else ""
        genres_str = f" [{', '.join(m.genres)}]" if m.genres else ""
        lines.append(f"- {m.title}{year_str}{genres_str}: {m.reason}")
    return "\n".join(lines) if lines else "(no movies were recommended)"


def build_evaluator_chain(api_key: str, model_name: str):
    llm = _make_llm(api_key, model_name, temperature=0.2)
    structured_llm = llm.with_structured_output(EvaluationResponse)
    prompt = ChatPromptTemplate.from_messages([("system", EVALUATOR_TEMPLATE)])
    return prompt | structured_llm


def get_evaluation(
    api_key: str,
    user_request: str,
    movies: List[MovieRecommendation],
    model_name: str = "gemini-3.6-flash",
) -> Tuple[Optional[EvaluationResponse], Optional[str]]:
    """Score the recommendation set. Returns (evaluation, error) - evaluation is
    None on failure (so a broken evaluator never takes down the main recommendation
    flow), but the real error is returned too instead of being swallowed, so the
    caller can surface it for debugging."""
    tracing.log(f"evaluator.get_evaluation: called with {len(movies)} movie(s), model={model_name}")
    if not movies:
        tracing.log("evaluator.get_evaluation: no movies to evaluate, skipping")
        return None, None
    try:
        chain = build_evaluator_chain(api_key, model_name)
        movie_list_text = _movies_to_plain_text(movies)
        with tracing.timed("evaluator.get_evaluation: structured judge call"):
            result = chain.invoke({"user_request": user_request, "movie_list": movie_list_text})
        tracing.log(
            f"evaluator.get_evaluation: overall_score={result.overall_score}, "
            f"{len(result.movie_evaluations)} movie score(s) returned"
        )
        return result, None
    except Exception as e:
        tracing.log_exception("evaluator.get_evaluation")
        return None, str(e)