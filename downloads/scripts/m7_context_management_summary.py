from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY=os.getenv("API_KEY")

# Step 2: Set up the Gemini API key ( do not do this in production)
client = genai.Client(api_key=API_KEY)

# Creating a new chat Session using Gemini's Chat API
chat = client.chats.create(model='gemini-3.6-flash')

# First Message
MESSAGE_1="I have 2 dogs in my home"
response = chat.send_message(MESSAGE_1)
print(MESSAGE_1)
print(response.text)


# Second Message
MESSAGE_2="How many paws are there in my home?"
response2 = chat.send_message(MESSAGE_2)
print(MESSAGE_2)
print(response2.text)

# Print Message History
print("\nConversation History")
for message in chat.get_history():
    print(f"{message.role.capitalize()}: {message.parts[0].text}")
