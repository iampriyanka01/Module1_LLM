"""
Pydantic models used across the app.

- MovieRecommendation / RecommendationResponse: the structured output we ask
  Gemini (via LangChain) to return.
- OmdbMovie: a typed wrapper around whatever OMDb's API gives us back.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class MovieRecommendation(BaseModel):
    """A single movie suggestion produced by the LLM."""

    title: str = Field(description="The exact movie title, as it would appear on IMDb/OMDb.")
    year: Optional[int] = Field(
        default=None, description="Release year of the movie, if known."
    )
    reason: str = Field(
        description="A short, specific explanation of why this movie fits the user's request."
    )
    genres: List[str] = Field(
        default_factory=list, description="A few relevant genres for this movie."
    )


class RecommendationResponse(BaseModel):
    """The full structured response the LLM must return for a recommendation turn."""

    reply: str = Field(
        description="A short, conversational reply to show the user above the movie list."
    )
    movies: List[MovieRecommendation] = Field(
        default_factory=list,
        description="0-5 movie recommendations. Empty if you are just asking a clarifying question.",
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="An optional follow-up question to help narrow down future recommendations.",
    )

    @field_validator("movies")
    @classmethod
    def _cap_movies(cls, v: List[MovieRecommendation]) -> List[MovieRecommendation]:
        return v[:5]


class MovieEvaluation(BaseModel):
    """An independent judge's score for a single recommended movie."""

    title: str = Field(description="Must exactly match the title being evaluated.")
    relevance_score: int = Field(
        description="An integer from 1 to 10: how well this specific movie fits the "
        "user's stated request/preferences.",
    )
    justification: str = Field(
        description="One short sentence on why it earned this score - be specific and critical."
    )

    @field_validator("relevance_score")
    @classmethod
    def _clamp_relevance(cls, v: int) -> int:
        # Gemini's structured-output schema converter doesn't reliably enforce
        # numeric min/max constraints (ge=/le= on the Field), so we clamp here in
        # Python instead of relying on the API to reject out-of-range values -
        # that also avoids the whole call failing if it returns e.g. 0 or 11.
        return max(1, min(10, v))


class EvaluationResponse(BaseModel):
    """Overall judge verdict for one recommendation turn."""

    overall_score: int = Field(
        description="An integer from 1 to 10: how well the recommendation set as a "
        "whole satisfies the user's request.",
    )
    overall_feedback: str = Field(
        description="1-2 sentences of honest, critical feedback on the recommendation set."
    )
    movie_evaluations: List[MovieEvaluation] = Field(default_factory=list)

    @field_validator("overall_score")
    @classmethod
    def _clamp_overall(cls, v: int) -> int:
        return max(1, min(10, v))


class OmdbMovie(BaseModel):
    """Normalized subset of fields we care about from the OMDb API response."""

    title: str
    year: Optional[str] = None
    genre: Optional[str] = None
    plot: Optional[str] = None
    poster: Optional[str] = None
    imdb_rating: Optional[str] = None
    imdb_id: Optional[str] = None
    director: Optional[str] = None
    actors: Optional[str] = None
    runtime: Optional[str] = None
    found: bool = True
    error: Optional[str] = None

    @classmethod
    def from_omdb_json(cls, data: dict) -> "OmdbMovie":
        if data.get("Response") == "False":
            return cls(
                title=data.get("_query_title", "Unknown"),
                found=False,
                error=data.get("Error", "Not found"),
            )
        return cls(
            title=data.get("Title", "Unknown"),
            year=data.get("Year"),
            genre=data.get("Genre"),
            plot=data.get("Plot"),
            poster=data.get("Poster") if data.get("Poster") != "N/A" else None,
            imdb_rating=data.get("imdbRating"),
            imdb_id=data.get("imdbID"),
            director=data.get("Director"),
            actors=data.get("Actors"),
            runtime=data.get("Runtime"),
            found=True,
        )