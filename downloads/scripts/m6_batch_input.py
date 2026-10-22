import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Set it in your .env file.")

# Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

# Create a list of prompts (batch input)
batch_prompts = [
    "Summarize the history of the internet in 3 lines.",
    "Explain what blockchain is for a beginner.",
    "What are the main components of a neural network?",
    "Give three use-cases of artificial intelligence in education.",
    "Describe the importance of data privacy in healthcare systems."
]

# Use .batch() to get results
responses = llm.batch(batch_prompts)

# Print responses
for i, response in enumerate(responses):
    print(f"\n🔹 Prompt {i+1}: {batch_prompts[i]}")
    print(f"🧠 Response: {response.content}")
