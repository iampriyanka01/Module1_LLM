import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

async def main():
    """
    Demonstrates non-blocking async streaming using astream().
    """
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key not found. Please set it in your .env file.")

    print("---  Initializing Model with Async Streaming ---")

    # Initialize the Gemini model with async streaming
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        streaming=True  # Still required to enable streaming mode
    )

    # Input message
    messages = [
        {"role": "user", "content": "Write a long story about AI and humanity coexisting peacefully."}
    ]

    print(" Generating async response...\n")

    try:
        # astream() must be awaited in an async context
        async for chunk in llm.astream(messages):
            print(chunk.content, end="", flush=True)

    except Exception as e:
        print(f"\n\nAn error occurred: {e}")

    print("\n\n Async Stream Complete.")

print("This script demonstrates how to use astream() for non-blocking async streaming with LangChain and Gemini 2.5 Flash.")

# Required to run async code in a normal script
if __name__ == "__main__":
    asyncio.run(main())
