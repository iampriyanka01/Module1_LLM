"""
Quick standalone check that your OMDb key actually works, independent of
Streamlit/LangChain. Run:

    python test_omdb_key.py YOUR_KEY_HERE
    # or, if OMDB_API_KEY is set in your .env / environment:
    python test_omdb_key.py
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OMDB_API_KEY")
    if not key:
        print("No key provided. Usage: python test_omdb_key.py YOUR_KEY")
        sys.exit(1)

    print(f"Testing OMDb key: {key[:4]}{'*' * max(len(key) - 4, 0)}")
    resp = requests.get(
        "https://www.omdbapi.com/",
        params={"apikey": key, "t": "Inception"},
        timeout=8,
    )
    print(f"HTTP status: {resp.status_code}")
    data = resp.json()
    print(data)

    if data.get("Response") == "True":
        print("\n✅ Key works. Poster URL:", data.get("Poster"))
    else:
        print("\n❌ Key/request failed:", data.get("Error"))
        print("Common fixes:")
        print("  - Check your inbox for an activation email and click the link.")
        print("  - Make sure there's no extra space/quote pasted into the key.")
        print("  - Confirm you haven't exceeded the free tier's 1,000 requests/day.")


if __name__ == "__main__":
    main()