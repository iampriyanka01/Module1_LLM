import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found in .env")

# Initialize the Gemini model (async + streaming)
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=api_key,
    streaming=True
)

# Define batch prompts
batch_prompts = [
    "Explain quantum computing in simple terms.",
    "What is the role of AI in education?",
    "Summarize the history of machine learning.",
]

# Async function to handle streaming of each prompt
async def handle_prompt(prompt, index):
    print(f"\n🔹 Prompt {index+1}: {prompt}")
    print("🧠 Response:", end=" ", flush=True)

    try:
        async for chunk in llm.astream(prompt):
            print(chunk.content, end="", flush=True)
    except Exception as e:
        print(f"\n❌ Error during astream: {e}")

    print("\n--- ✅ Stream Complete ---")

# Async background task (e.g. monitoring/logging)
async def background_logger():
    for i in range(5):
        print(f"\n🛠️ [Logger] System check {i+1}...")
        await asyncio.sleep(1.5)

# Main async runner
async def main():
    # Create prompt streaming tasks
    stream_tasks = [handle_prompt(p, i) for i, p in enumerate(batch_prompts)]
    
    # Add the logger task
    all_tasks = stream_tasks + [background_logger()]
    
    # Run all concurrently
    await asyncio.gather(*all_tasks)

if __name__ == "__main__":
    asyncio.run(main())
