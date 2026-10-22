# -----------------------------------------------------------------------------
#  Imports & Environment Setup
# -----------------------------------------------------------------------------
import os
from dotenv import load_dotenv  # Loads .env file into environment variables
from google import genai         # Google GenAI SDK for Gemini Developer API
# -----------------------------------------------------------------------------
# Step 0: Load environment variables from `.env`
# -----------------------------------------------------------------------------
load_dotenv()  # Reads `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) into os.environ

# -----------------------------------------------------------------------------
# Step 1: Retrieve API key & basic validation
# -----------------------------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY")  # SDK checks this if not passed explicitly
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in environment. Please set it.")

# -----------------------------------------------------------------------------
# Step 2: Initialize the Gemini LLM client
# -----------------------------------------------------------------------------
# Create a client object using the Developer API (not Vertex AI)
client = genai.Client(api_key=api_key)  # Auth via environment variable :contentReference[oaicite:2]{index=2}

# -----------------------------------------------------------------------------
# Step 3: Define classification + response function
# -----------------------------------------------------------------------------
def classify_and_respond(feedback: str) -> dict:
    """
    Sends user feedback to Gemini and returns a category and response.
    """
    #  Build a single-turn prompt with clear instructions and format
    prompt = f"""
You are a customer support assistant.

Classify the following feedback into exactly one of: complaint, suggestion, praise.
Then compose a polite response.

Feedback: "{feedback}"

Format:
Category: <one word>
Response: <your reply>
"""

    #  Send prompt to Gemini and await the result
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )  # This method is part of `client.models` as shown in the SDK docs :contentReference[oaicite:3]{index=3}

    #  Parse the result: split lines, strip whitespace
    text = resp.text.strip().split("\n")

    #  Extract category and response from formatted output
    result = {
        "category": text[0].split(":", 1)[1].strip(),         # e.g. "complaint"
        "response": text[1].split(":", 1)[1].strip()          # e.g. "I'm sorry to hear..."
    }
    return result

# -----------------------------------------------------------------------------
# Step 4: CLI / Test runner
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "I love the service!",
        "Your app crashes every time I open it.",
        "Could you add a dark mode to the app?"
    ]

    for fb in tests:
        out = classify_and_respond(fb)
        print(f"Feedback: {fb}")
        print(f"→ Category: {out['category']}")
        print(f"→ Response: {out['response']}\n")
