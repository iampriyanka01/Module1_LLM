import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.output_parsers import StrOutputParser

def main():
    """
    Main function to demonstrate streaming with LangChain and Gemini 1.5 Flash.
    """
    # 1. Load Environment Variables
    # This line loads the GOOGLE_API_KEY from your .env file.
    load_dotenv()
    
    # Securely get the API key and check if it exists   
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key not found. Please set it in your .env file.")

    print("--- 🚀 Initializing Model and Chain ---")
    
    # 2. Instantiate the Gemini Model for Streaming
    # We specify the exact model name.
    # 'streaming=True' is good practice to signal our intent.
    # 'temperature' controls the creativity of the response.


# Load environment variables
load_dotenv()

# Initialize model directly
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    streaming=True
)

# Input message (as a list of messages)
messages = [
    {"role": "user", "content": "Explain black holes in simple terms in about 2000 words."}
]

# Stream the response directly from the model
print("🤖 Generating response...\n")

try:
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)

except Exception as e:
    print(f"\n\nAn error occurred: {e}")


print("\n\n✅ Stream Complete.")
print("This script demonstrates stream() method is synchronous method because this will be executed after the streaming completes. ")