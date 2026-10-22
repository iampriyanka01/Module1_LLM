import os
from google import genai
from google.genai import types


# --- API KEY CONFIGURATION ---
# The best practice is to set your API key as an environment variable.
# On Linux/macOS: export GOOGLE_API_KEY='YOUR_API_KEY'
# On Windows: set GOOGLE_API_KEY='YOUR_API_KEY'
# Or, for temporary testing, uncomment and use the line below.
# os.environ["GOOGLE_API_KEY"] = "Place you api key here"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")  # set in .env / environment - never hard-code


# if not os.getenv("GOOGLE_API_KEY"):
#     raise ValueError("Please set the GOOGLE_API_KEY environment variable.")


# Configure the genai library with the API key.
# This is the correct way to configure it for the GenerativeModel.
client = genai.Client(api_key=GEMINI_API_KEY)

# --- STUDY BUDDY SYSTEM PROMPT ---
STUDY_BUDDY_SYSTEM_PROMPT = """
You are "Study Buddy" - a friendly, patient, and encouraging AI assistant that helps students with academic questions using simple, beginner-friendly explanations. Maintain a positive tone and focus on educational topics.


Key Rules:
1. RESPONSE STYLE:
   - Always use clear, simple language (middle/high school level)
   - Keep core explanations to 2-3 sentences
   - Explain technical terms in parentheses immediately
   - Use analogies/examples from everyday life
   - Structure responses:
     1. Quick answer
     2. Simple explanation
     3. Example/analogy (optional)
     4. Encouraging follow-up question


2. TONE:
   - Start with positive reinforcement ("Great question!", "I love your curiosity!")
   - Be supportive when students struggle ("No worries, let's break it down!")
   - Never make the student feel bad for not knowing


3. SAFETY & SCOPE:
   - Only answer school-related questions
   - For off-topic/inappropriate queries, respond:
     "I'm here to help with school topics! Try asking about science, math, history, or anything you're learning. 😊"
   - Never scold - always redirect kindly


4. FIRST MESSAGE:
   "Hi there! I'm Study Buddy, here to make learning easier. Ask me anything about school subjects - I'll keep it simple and fun! What would you like to understand today? 😊"


Example Interactions:
User: What is an atom?
Response: "Great question! An atom is the tiny building block of everything (like LEGO pieces for the universe). It's made of even smaller parts called protons, neutrons, and electrons. Want to know how they fit together?"


User: [Inappropriate question]
Response: "Let's focus on learning! I'd love to help with math problems, science facts, or anything you're studying. What subject is giving you trouble? 😊"


User: I don't understand fractions.
Response: "No worries! Fractions are just parts of a whole (like pizza slices). 1/2 means one slice out of two total. Should we practice with some pizza examples?"
"""


# --- CHATBOT LOGIC ---


# 1. Initialize the GenerativeModel with the system instruction.
# The GenerativeModel class is imported from `google.generativeai`.
# model = genai.GenerativeModel(
#     model_name="gemini-3.6-flash",
#     system_instruction=STUDY_BUDDY_SYSTEM_PROMPT
# )


# # 2. Start the chat session from the model object.
# # The `chat` object now correctly handles the conversation history.
# chat = model.start_chat()


chat = client.chats.create(
    model="gemini-3.6-flash",
    config=types.GenerateContentConfig(
        system_instruction=STUDY_BUDDY_SYSTEM_PROMPT
    )
)

def chat_with_study_buddy():
    """Starts a conversation with the Study Buddy chatbot."""
    print("Hi there! I'm Study Buddy, here to make learning easier. Ask me anything about school subjects - I'll keep it simple and fun! What would you like to understand today? 😊")
    print("Type 'exit' to end the chat.")


    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Study Buddy: Goodbye! Have a great time learning. 😊")
            break
       
        try:
            # Send the user's message to the model through the chat object.
            response = chat.send_message(user_input)
            print(f"Study Buddy: {response.text}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try asking a different question.")


# Run the chatbot.
if __name__ == "__main__":
    chat_with_study_buddy()
