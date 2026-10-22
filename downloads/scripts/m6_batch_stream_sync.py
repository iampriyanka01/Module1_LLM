import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Set it in your .env file.")

# Initialize Gemini with streaming enabled
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

# Process each prompt using stream()
for i, prompt in enumerate(batch_prompts):
    print(f"\n🔹 Prompt {i+1}: {prompt}")
    print("🧠 Response:", end=" ", flush=True)

    try:
        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")

    print("\n--- ✅ Stream Complete ---")
