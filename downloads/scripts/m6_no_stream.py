import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    """
    Demonstrates how Gemini responds without streaming.
    """
    # Load environment variables
    load_dotenv()

    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key not found. Please set it in your .env file.")

    print("--- 🚀 Initializing Model (No Streaming) ---")

    # Initialize model WITHOUT streaming
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.7,
        streaming=False  #  Turn off streaming
    )
    

    # Input message
    messages = [
        {"role": "user", "content": "Explain black holes in simple terms in about 2000 words."}
    ]

    print("🤖 Generating response...\n")

    try:
        # Use .invoke() instead of .stream()
        response = llm.invoke(messages)

        # Print the full response at once
        print(response.content)

    except Exception as e:
        print(f"\n\nAn error occurred: {e}")

    print("\n\n✅ Response Complete.")

if __name__ == "__main__":
    main()
