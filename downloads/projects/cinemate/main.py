import os
import json

from dotenv import load_dotenv
from groq import Groq

from tools import (
    update_user_profile,
    get_user_profile,
    search_movies,
    get_movie_genres,
    discover_movies,
    get_movie_details,
)


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from .env"
    )


# =========================================================
# GROQ CLIENT
# =========================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# =========================================================
# MODEL
# =========================================================

MODEL = "openai/gpt-oss-20b"


# =========================================================
# SYSTEM PROMPT
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

Do NOT ask the user to explicitly tell you to remember
something.

If they say:

"I love Interstellar"

you should update their profile.

If they say:

"I don't like horror"

you should update their profile.

If they say:

"I've already seen Inception"

you should update watched_movies.

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

Whenever possible, use TMDB tools instead of relying
on your own knowledge for:

- Movie titles
- Release dates
- Ratings
- Movie descriptions
- Cast
- Directors
- Genres
- Movie discovery

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
6. Prefer movies matching their favorite genres,
   actors, directors, languages and preferences.
7. Explain WHY each movie matches their taste.

Try to recommend 3-5 movies.

=========================================================
TMDB TOOLS
=========================================================

Use search_movies when:

- The user mentions a specific movie.
- You need to verify a movie.
- You need to find a movie by title.

Use get_movie_details when:

- You need detailed information about a movie.
- You need cast/director information.
- You need more information about a movie returned
  by search_movies or discover_movies.

Use get_movie_genres when:

- You need TMDB genre IDs.

Use discover_movies when:

- The user asks for recommendations.
- The user describes a genre, year, rating, language,
  or other movie preference.

=========================================================
IMPORTANT
=========================================================

Do not recommend a movie simply because you personally
know it.

Use TMDB to obtain real candidates.

Then use the user's profile and reasoning to determine
which candidates are the best fit.

Keep answers conversational rather than returning raw JSON.
"""


# =========================================================
# GROQ TOOLS
# =========================================================

tools = [

    # -----------------------------------------------------
    # UPDATE USER PROFILE
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "update_user_profile",

            "description": """
            Update the user's movie preference profile.

            Call this whenever the user reveals information
            about their movie preferences.
            """,

            "parameters": {

                "type": "object",

                "properties": {

                    "name": {
                        "type": "string",
                        "description": "User's name"
                    },

                    "favorite_movies": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Movies the user loves"
                    },

                    "favorite_genres": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Genres the user likes"
                    },

                    "disliked_genres": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Genres the user dislikes"
                    },

                    "preferred_languages": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Preferred movie languages"
                    },

                    "favorite_actors": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Favorite actors"
                    },

                    "favorite_directors": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Favorite directors"
                    },

                    "watched_movies": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Movies the user has watched"
                    },

                    "likes_hidden_gems": {
                        "type": "boolean",
                        "description": """
                        Whether the user likes underrated
                        or lesser-known movies
                        """
                    },
                },
            },
        },
    },

    # -----------------------------------------------------
    # GET USER PROFILE
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "get_user_profile",

            "description": """
            Get the user's current movie preference profile.
            Use this before making personalized recommendations.
            """,

            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },

    # -----------------------------------------------------
    # SEARCH MOVIES
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "search_movies",

            "description": """
            Search TMDB for movies by title or keyword.

            Use this when the user mentions a specific movie
            or when you need to find a movie.
            """,

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description": "Movie title or search query"
                    }

                },

                "required": [
                    "query"
                ],
            },
        },
    },

    # -----------------------------------------------------
    # GET MOVIE GENRES
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "get_movie_genres",

            "description": """
            Get the list of movie genres and their TMDB IDs.

            Use this when you need genre IDs for discovering
            movies.
            """,

            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },

    # -----------------------------------------------------
    # DISCOVER MOVIES
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "discover_movies",

            "description": """
            Discover movies from TMDB using filters.

            Use this for movie recommendations.

            genre_ids should contain TMDB genre IDs.

            min_rating can be used when the user asks for
            highly rated movies.

            year can be used when the user specifies a year.

            language is a two-letter ISO language code such
            as 'en', 'hi', 'fr', etc.

            sort_by can be:
            popularity.desc
            vote_average.desc
            primary_release_date.desc
            """,

            "parameters": {

                "type": "object",

                "properties": {

                    "genre_ids": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "description": "TMDB genre IDs"
                    },

                    "year": {
                        "type": "integer",
                        "description": "Release year"
                    },

                    "min_rating": {
                        "type": "number",
                        "description": "Minimum TMDB rating"
                    },

                    "language": {
                        "type": "string",
                        "description": "Original language ISO code"
                    },

                    "sort_by": {
                        "type": "string",
                        "description": """
                        TMDB sorting method.
                        Example: popularity.desc
                        """
                    },

                },
            },
        },
    },

    # -----------------------------------------------------
    # MOVIE DETAILS
    # -----------------------------------------------------

    {
        "type": "function",

        "function": {

            "name": "get_movie_details",

            "description": """
            Get detailed information about a movie from TMDB.

            Use this when you need detailed information,
            cast, director, genres, runtime or other metadata.
            """,

            "parameters": {

                "type": "object",

                "properties": {

                    "movie_id": {
                        "type": "integer",
                        "description": "TMDB movie ID"
                    }

                },

                "required": [
                    "movie_id"
                ],
            },
        },
    },
]


# =========================================================
# FUNCTION MAP
# =========================================================

available_functions = {

    "update_user_profile": update_user_profile,

    "get_user_profile": get_user_profile,

    "search_movies": search_movies,

    "get_movie_genres": get_movie_genres,

    "discover_movies": discover_movies,

    "get_movie_details": get_movie_details,
}


# =========================================================
# CHAT / AGENT LOOP
# =========================================================

def run_chat(messages):

    """
    Run the Groq agent.

    The agent can make multiple function calls before
    producing the final answer.
    """

    while True:

        response = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            tools=tools,

            tool_choice="auto",

            temperature=0.7,
        )

        response_message = response.choices[0].message

        # Add assistant message to conversation
        messages.append(response_message)

        # -------------------------------------------------
        # NO TOOL CALL
        # -------------------------------------------------

        if not response_message.tool_calls:

            return response_message.content

        # -------------------------------------------------
        # TOOL CALLS
        # -------------------------------------------------

        for tool_call in response_message.tool_calls:

            function_name = tool_call.function.name

            print(
                f"\n  🔧 Tool: {function_name}"
            )

            try:

                function_args = json.loads(
                    tool_call.function.arguments
                )

                function_to_call = (
                    available_functions.get(
                        function_name
                    )
                )

                if function_to_call is None:

                    tool_result = {
                        "success": False,
                        "error": (
                            f"Unknown function: "
                            f"{function_name}"
                        ),
                    }

                else:

                    tool_result = function_to_call(
                        **function_args
                    )

                print(
                    "  ✓ Done"
                )

            except Exception as error:

                print(
                    f"  ✗ Tool error: {error}"
                )

                tool_result = {
                    "success": False,
                    "error": str(error),
                }

            # -------------------------------------------------
            # SEND TOOL RESULT BACK TO GROQ
            # -------------------------------------------------

            messages.append({

                "role": "tool",

                "tool_call_id": tool_call.id,

                "content": json.dumps(
                    tool_result
                ),
            })

        # -------------------------------------------------
        # LOOP AGAIN
        # -------------------------------------------------

        # Groq now receives the tool results and decides
        # whether it needs another tool or can answer.


# =========================================================
# MAIN CLI
# =========================================================

def main():

    print()
    print("=" * 65)

    print(
        "🎬 CineBot — Your Personal Movie Recommendation Assistant"
    )

    print("=" * 65)

    print()
    print(
        "I'll learn your movie taste as we chat."
    )

    print(
        "Type 'profile' to see what I know about you."
    )

    print(
        "Type 'exit' to quit."
    )

    print()


    # -----------------------------------------------------
    # CONVERSATION MEMORY
    # -----------------------------------------------------

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }

    ]


    # -----------------------------------------------------
    # INITIAL GREETING
    # -----------------------------------------------------

    print(
        "🤖 CineBot: "
        "Hey! 👋 Let's get to know your movie taste."
    )

    print()

    print(
        "What's one movie you absolutely loved?"
    )

    print()


    # -----------------------------------------------------
    # CHAT LOOP
    # -----------------------------------------------------

    while True:

        try:

            user_input = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):

            print(
                "\n\n👋 Goodbye!\n"
            )

            break


        if not user_input:
            continue


        # -------------------------------------------------
        # EXIT
        # -------------------------------------------------

        if user_input.lower() in {
            "exit",
            "quit",
            "bye",
        }:

            print()
            print(
                "🤖 CineBot: "
                "Thanks for chatting! See you next time. 🎬"
            )
            print()

            break


        # -------------------------------------------------
        # PROFILE COMMAND
        # -------------------------------------------------

        if user_input.lower() == "profile":

            profile = get_user_profile()

            print()
            print("🧠 YOUR MOVIE PROFILE")
            print("-" * 40)

            print(
                json.dumps(
                    profile,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            print()

            continue


        # -------------------------------------------------
        # ADD USER MESSAGE
        # -------------------------------------------------

        messages.append({

            "role": "user",

            "content": user_input,

        })


        # -------------------------------------------------
        # CALL AGENT
        # -------------------------------------------------

        try:

            print()

            response = run_chat(
                messages
            )

            print(
                f"🤖 CineBot: {response}"
            )

            print()

        except Exception as error:

            print()
            print(
                f"❌ Error: {error}"
            )
            print()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()