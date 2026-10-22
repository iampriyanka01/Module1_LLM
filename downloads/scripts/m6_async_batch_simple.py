import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Set it in your .env file.")

# Initialize the Gemini Model (async-compatible)
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=api_key
)

# List of prompts
batch_prompts = [
    "Summarize the history of the internet in 3 lines.",
    "Explain what blockchain is for a beginner.",
    "What are the main components of a neural network?",
    "Give three use-cases of artificial intelligence in education.",
    "Describe the importance of data privacy in healthcare systems."
]

# Define async function for abatch
async def run_abatch():
    print("🚀 Running abatch (async batch processing)...\n")
    responses = await llm.abatch(batch_prompts)

    # Display results
    for i, response in enumerate(responses):
        print(f"\n🔹 Prompt {i+1}: {batch_prompts[i]}")
        print(f"🧠 Response: {response.content}")

# Run the async function
if __name__ == "__main__":
    asyncio.run(run_abatch())
