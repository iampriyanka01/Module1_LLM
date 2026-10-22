# 🍿 Gemini Cinema Advisor

A Streamlit chat app that recommends movies. It uses:

- **Streamlit** — chat UI
- **LangChain + Gemini** (`langchain-google-genai`) — conversational recommendation logic
- **Pydantic** — validates the LLM's output into a strict schema (no free-text parsing)
- **OMDb API** — enriches each recommendation with poster, rating, genre, runtime

## Project layout

```
movie_advisor/
├── app.py            # Streamlit UI + chat loop
├── recommender.py     # LangChain chain: prompt -> Gemini -> structured output
├── models.py           # Pydantic schemas (RecommendationResponse, OmdbMovie, ...)
├── omdb_client.py       # Thin OMDb API wrapper
├── requirements.txt
└── .env.example
```

## Setup

1. **Install dependencies**
   ```bash
   python -m venv venv && source venv/bin/activate   # optional but recommended
   pip install -r requirements.txt
   ```

2. **Get API keys**
   - Gemini: https://aistudio.google.com/apikey (free tier available)
   - OMDb: http://www.omdbapi.com/apikey.aspx (free tier: 1,000 requests/day)

3. **Configure keys** — either:
   - Copy `.env.example` to `.env` and fill in your keys, or
   - Paste them directly into the sidebar when the app is running (stored only in that session).

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## How it works

1. Every user message, plus the running chat history, is sent through a LangChain
   `ChatPromptTemplate` to Gemini.
2. `llm.with_structured_output(RecommendationResponse)` forces Gemini to return
   JSON matching the `RecommendationResponse` Pydantic model — a short conversational
   reply, a list of up to 5 `MovieRecommendation`s (title, year, reason, genres), and
   an optional follow-up question. No manual JSON parsing/regex needed.
3. For each recommended title, `OmdbClient.get_movie()` looks it up on OMDb to pull a
   poster, IMDb rating, genre, and runtime, which Streamlit renders as a card.
4. Conversation memory is just a list of `(role, text)` tuples kept in
   `st.session_state`, replayed into the prompt each turn so Gemini remembers stated
   likes/dislikes (e.g. "I hate horror movies").

## Extending it

- **Swap in a vector store** for a "seen/liked" movie history so recommendations
  avoid repeats across sessions.
- **Add filters** (era, runtime, streaming availability) as sidebar widgets that get
  folded into the prompt.
- **Cache OMDb lookups** with `st.cache_data` to cut down on repeat API calls.
- **Add a `RunnableWithMessageHistory`** from LangChain if you want persistent,
  multi-session memory instead of the in-memory list used here.

  # Gemini API key: https://aistudio.google.com/apikey — sign in with a Google account, click "Create API # # key." Free tier available.
  # OMDb API key: http://www.omdbapi.com/apikey.aspx — pick the free tier (1,000 requests/day), enter your # email, and they'll send the key there
