from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY=os.getenv("API_KEY")


# Step 2: Set up the Gemini API key ( do not do this in production)
client = genai.Client(api_key=API_KEY)


exit_words = {"bye", "exit", "quit", "goodbye", "see you"}


while True:
    user_input = input("Ask Gemini (type 'bye' or 'exit' to quit): ")
    if any(word in user_input.lower() for word in exit_words):
        print("Exiting chat. Goodbye!")
        break
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=user_input
    )
    print(response.text)
