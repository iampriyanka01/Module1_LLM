import os
import requests

from typing import Optional
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE_URL = "https://api.themoviedb.org/3"

TMDB_HEADERS = {
    "Authorization": f"Bearer {TMDB_API_KEY}",
    "accept": "application/json",
}


# =========================================================
# IN-MEMORY USER PROFILE
# =========================================================

user_profile = {
    "name": None,

    "favorite_movies": [],

    "favorite_genres": [],

    "disliked_genres": [],

    "preferred_languages": [],

    "favorite_actors": [],

    "favorite_directors": [],

    "watched_movies": [],

    "likes_hidden_gems": None,
}


# =========================================================
# HELPER
# =========================================================

def add_unique(existing_list, new_items):
    """
    Add items to a list without duplicates.
    """

    if not new_items:
        return existing_list

    for item in new_items:

        if item and item not in existing_list:
            existing_list.append(item)

    return existing_list


def tmdb_request(endpoint, params=None):
    """
    Make a request to TMDB.
    """

    if not TMDB_API_KEY:
        raise ValueError(
            "TMDB_API_KEY is missing from .env"
        )

    url = f"{TMDB_BASE_URL}{endpoint}"

    response = requests.get(
        url,
        headers=TMDB_HEADERS,
        params=params or {},
        timeout=15,
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# USER PROFILE TOOLS
# =========================================================

def update_user_profile(
    name: Optional[str] = None,
    favorite_movies: Optional[list[str]] = None,
    favorite_genres: Optional[list[str]] = None,
    disliked_genres: Optional[list[str]] = None,
    preferred_languages: Optional[list[str]] = None,
    favorite_actors: Optional[list[str]] = None,
    favorite_directors: Optional[list[str]] = None,
    watched_movies: Optional[list[str]] = None,
    likes_hidden_gems: Optional[bool] = None,
):
    """
    Update the user's movie preference profile.
    """

    if name is not None:
        user_profile["name"] = name

    if favorite_movies:
        add_unique(
            user_profile["favorite_movies"],
            favorite_movies
        )

    if favorite_genres:
        add_unique(
            user_profile["favorite_genres"],
            favorite_genres
        )

    if disliked_genres:
        add_unique(
            user_profile["disliked_genres"],
            disliked_genres
        )

    if preferred_languages:
        add_unique(
            user_profile["preferred_languages"],
            preferred_languages
        )

    if favorite_actors:
        add_unique(
            user_profile["favorite_actors"],
            favorite_actors
        )

    if favorite_directors:
        add_unique(
            user_profile["favorite_directors"],
            favorite_directors
        )

    if watched_movies:
        add_unique(
            user_profile["watched_movies"],
            watched_movies
        )

    if likes_hidden_gems is not None:
        user_profile["likes_hidden_gems"] = likes_hidden_gems

    return {
        "success": True,
        "profile": user_profile,
    }


def get_user_profile():
    """
    Return the current user profile.
    """

    return user_profile


# =========================================================
# TMDB TOOL 1
# SEARCH MOVIES
# =========================================================

def search_movies(query: str):
    """
    Search TMDB for movies by title or keyword.
    """

    data = tmdb_request(
        "/search/movie",
        {
            "query": query,
            "language": "en-US",
            "include_adult": False,
            "page": 1,
        }
    )

    results = []

    for movie in data.get("results", [])[:8]:

        results.append({
            "id": movie.get("id"),
            "title": movie.get("title"),
            "original_title": movie.get("original_title"),
            "release_date": movie.get("release_date"),
            "overview": movie.get("overview"),
            "rating": movie.get("vote_average"),
            "vote_count": movie.get("vote_count"),
            "popularity": movie.get("popularity"),
            "poster_path": movie.get("poster_path"),
            "original_language": movie.get(
                "original_language"
            ),
        })

    return {
        "query": query,
        "results": results,
    }


# =========================================================
# TMDB TOOL 2
# GET MOVIE GENRES
# =========================================================

def get_movie_genres():
    """
    Get TMDB movie genre names and IDs.
    """

    data = tmdb_request(
        "/genre/movie/list",
        {
            "language": "en",
        }
    )

    genres = {}

    for genre in data.get("genres", []):

        genres[genre["name"]] = genre["id"]

    return genres


# =========================================================
# TMDB TOOL 3
# DISCOVER MOVIES
# =========================================================

def discover_movies(
    genre_ids: Optional[list[int]] = None,
    year: Optional[int] = None,
    min_rating: Optional[float] = None,
    language: Optional[str] = None,
    sort_by: str = "popularity.desc",
):
    """
    Discover movies using TMDB filters.
    """

    params = {
        "language": "en-US",
        "sort_by": sort_by,
        "include_adult": False,
        "include_video": False,
        "page": 1,
    }

    if genre_ids:
        params["with_genres"] = ",".join(
            str(genre_id)
            for genre_id in genre_ids
        )

    if year:
        params["primary_release_year"] = year

    if min_rating is not None:
        params["vote_average.gte"] = min_rating

    if language:
        params["with_original_language"] = language

    data = tmdb_request(
        "/discover/movie",
        params
    )

    results = []

    for movie in data.get("results", [])[:10]:

        results.append({
            "id": movie.get("id"),
            "title": movie.get("title"),
            "original_title": movie.get("original_title"),
            "release_date": movie.get("release_date"),
            "overview": movie.get("overview"),
            "rating": movie.get("vote_average"),
            "vote_count": movie.get("vote_count"),
            "popularity": movie.get("popularity"),
            "poster_path": movie.get("poster_path"),
            "original_language": movie.get(
                "original_language"
            ),
        })

    return {
        "results": results,
    }


# =========================================================
# TMDB TOOL 4
# GET MOVIE DETAILS
# =========================================================

def get_movie_details(movie_id: int):
    """
    Get detailed information about a movie.
    """

    data = tmdb_request(
        f"/movie/{movie_id}",
        {
            "language": "en-US",
            "append_to_response": "credits,videos",
        }
    )

    # -----------------------------------------------------
    # Extract cast
    # -----------------------------------------------------

    cast = []

    for person in data.get(
        "credits",
        {}
    ).get(
        "cast",
        []
    )[:10]:

        cast.append({
            "name": person.get("name"),
            "character": person.get("character"),
        })

    # -----------------------------------------------------
    # Extract directors
    # -----------------------------------------------------

    directors = []

    for person in data.get(
        "credits",
        {}
    ).get(
        "crew",
        []
    ):

        if person.get("job") == "Director":

            directors.append(
                person.get("name")
            )

    # -----------------------------------------------------
    # Extract genres
    # -----------------------------------------------------

    genres = [
        genre.get("name")
        for genre in data.get("genres", [])
    ]

    # -----------------------------------------------------
    # Return useful information
    # -----------------------------------------------------

    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "original_title": data.get("original_title"),
        "tagline": data.get("tagline"),
        "overview": data.get("overview"),
        "release_date": data.get("release_date"),
        "runtime": data.get("runtime"),
        "rating": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "genres": genres,
        "original_language": data.get(
            "original_language"
        ),
        "budget": data.get("budget"),
        "revenue": data.get("revenue"),
        "poster_path": data.get("poster_path"),
        "backdrop_path": data.get("backdrop_path"),
        "cast": cast,
        "directors": directors,
        "homepage": data.get("homepage"),
    }