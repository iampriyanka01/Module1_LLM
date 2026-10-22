"""
Thin wrapper around the OMDb API (http://www.omdbapi.com/).

Get a free API key at http://www.omdbapi.com/apikey.aspx
"""

from __future__ import annotations

import requests
from models import OmdbMovie
import tracing

OMDB_BASE_URL = "https://www.omdbapi.com/"


class OmdbClient:
    def __init__(self, api_key: str, timeout: int = 8):
        if not api_key:
            raise ValueError("OMDb API key is required.")
        self.api_key = api_key
        self.timeout = timeout

    def get_movie(self, title: str, year: int | None = None) -> OmdbMovie:
        """Look up a single movie by title (and optional year) for rich details."""
        tracing.log(f"omdb_client.get_movie: looking up title={title!r}, year={year}")
        params = {"apikey": self.api_key, "t": title, "plot": "short"}
        if year:
            params["y"] = str(year)

        try:
            with tracing.timed(f"omdb_client.get_movie: OMDb request for {title!r}"):
                resp = requests.get(OMDB_BASE_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
        except requests.RequestException as e:
            tracing.log(f"omdb_client.get_movie: request failed for {title!r} - {e}", level="ERROR")
            return OmdbMovie(title=title, found=False, error=str(e))

        data["_query_title"] = title
        movie = OmdbMovie.from_omdb_json(data)
        tracing.log(
            f"omdb_client.get_movie: OMDb found={movie.found}, poster="
            f"{'yes' if movie.poster else 'no'}, error={movie.error!r}"
        )

        # OMDb sometimes returns a poster URL that is dead/unreachable (common for
        # very recently released movies before a real poster is indexed). Verify it
        # actually resolves before we hand it to the UI, so we can fall back cleanly
        # instead of showing a broken-image icon.
        if movie.poster and not self._poster_is_reachable(movie.poster):
            tracing.log(f"omdb_client.get_movie: poster URL unreachable for {title!r}, dropping it")
            movie.poster = None
            movie.error = movie.error or "Poster URL on file is broken/unreachable."

        return movie

    def _poster_is_reachable(self, url: str) -> bool:
        if not url.startswith(("http://", "https://")):
            return False
        try:
            resp = requests.head(url, timeout=4, allow_redirects=True)
            if resp.status_code >= 400:
                return False
            content_type = resp.headers.get("Content-Type", "")
            return content_type.startswith("image/") or content_type == ""
        except requests.RequestException:
            return False

    def search(self, query: str) -> list[dict]:
        """Simple title search - useful for autocomplete-style lookups."""
        params = {"apikey": self.api_key, "s": query}
        try:
            resp = requests.get(OMDB_BASE_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            return []

        if data.get("Response") == "False":
            return []
        return data.get("Search", [])