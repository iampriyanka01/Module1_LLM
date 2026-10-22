# CineBot — Streamlit UI

A chat-based movie recommendation assistant, powered by Groq (tool-calling
LLM) and TMDB (movie data). This is a UI port of the original CLI script —
same agent loop and tools, now in a browser.

## What changed from the CLI version

- **`app.py`** replaces `main.py`. It runs the same tool-calling agent loop
  (`run_chat`), but drives a Streamlit `st.chat_message` / `st.chat_input`
  interface instead of a terminal `input()` loop.
- **`tools.py`** keeps all four TMDB functions (`search_movies`,
  `get_movie_genres`, `discover_movies`, `get_movie_details`) exactly as
  before. The one real change: the old CLI used a single **global**
  `user_profile` dict, which is fine for one person in a terminal but would
  leak between users in a web app (Streamlit runs one process shared by
  everyone who opens the page). That's now a `ProfileManager` class,
  and `app.py` creates **one instance per browser session**
  (`st.session_state.profile_manager`), so each visitor gets their own
  private profile.
- Added a **sidebar** showing the live profile (name, favorite movies,
  genres, actors, etc.) as it's learned, plus buttons to start a new chat
  or clear the profile.
- Added an inline **poster gallery** — whenever a tool call returns movies
  (from `search_movies` or `discover_movies`), their posters render as a
  strip under CineBot's reply.
- Added a small **status box** so you can see which tool is being called
  while CineBot thinks (helpful for debugging / transparency).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your API keys. Either:
   - Copy `.env.example` to `.env` and fill in your keys, **or**
   - Paste them directly into the sidebar's "API keys" expander when you
     run the app (stored only for that session, not saved to disk).

   ```
   GROQ_API_KEY=your_groq_api_key
   TMDB_API_KEY=your_tmdb_v4_read_access_token
   ```

   Note: TMDB needs the **v4 Read Access Token** (long JWT-style string),
   not the shorter v3 API key — the app sends it as a `Bearer` header.

3. Run the app:
   ```bash
   streamlit run app.py
   ```

   It'll open at `http://localhost:8501`.

## Notes / limitations

- Profile and chat history live in `st.session_state`, so they reset if
  you close the tab or restart the server — there's no database. If you
  want persistence across restarts, swap `ProfileManager` for something
  backed by SQLite or a JSON file per user.
- The Groq free tier has rate limits; if you see errors, wait a moment or
  check your usage on the Groq console.