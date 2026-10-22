import os
from dotenv import load_dotenv

# LangChain Gemini integration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------------------------------------------------------
# Step 0: Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()  # loads variables from a `.env` file into os.environ

# -----------------------------------------------------------------------------
# Step 1: Retrieve and validate the API key
# -----------------------------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY")  # fetch API key from environment
if not api_key:
    # Fail early if the API key is missing
    raise ValueError("Missing GOOGLE_API_KEY — please add it to your .env file")

# -----------------------------------------------------------------------------
# Step 2: Initialize the Gemini LLM via LangChain
# -----------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",    # specify the Gemini model
    temperature=0.3,            # moderate randomness in generation
    google_api_key=api_key      # pass API key explicitly for security
)

# -----------------------------------------------------------------------------
# Step 3: Build the chat-style prompt template
# -----------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant."),
    ("human",
     "Feedback: {feedback}\n\n"
     "Classify as exactly one of: complaint, suggestion, praise. "
     "Then respond politely in this format:\n"
     "Category: <category>\n"
     "Response: <response>")
])

# -----------------------------------------------------------------------------
# Step 4: Define function to classify feedback and generate a response
# -----------------------------------------------------------------------------
def classify_and_respond(feedback: str) -> dict:
    """
    Sends customer feedback to Gemini and returns:
      - `category`: one of 'complaint', 'suggestion', 'praise'
      - `response`: a polite, appropriate reply
    """
    # 4.1 Format the prompt with the input feedback
    formatted = prompt.format_prompt(feedback=feedback).to_messages()
    
    # 4.2 Invoke the LLM with the formatted chat prompt
    msg = llm.invoke(formatted)  # returns a message object with `.content` field
    
    # 4.3 Split the raw content into non-empty lines
    lines = [l.strip() for l in msg.content.splitlines() if l.strip()]
    
    # 4.4 Extract 'Category' via a generator expression with fallback
    cat = next(
        (line.split(":",1)[1].strip()
         for line in lines
         if line.lower().startswith("category:")),
        "unknown"  # default if not found
    )
    
    # 4.5 Extract 'Response' similarly
    resp = next(
        (line.split(":",1)[1].strip()
         for line in lines
         if line.lower().startswith("response:")),
        ""  # default to empty string if not found
    )
    
    # 4.6 Return a structured dictionary
    return {"category": cat, "response": resp}

# -----------------------------------------------------------------------------
# Step 5: Example usage when this file is run directly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "I love the service!",
        "The checkout process is too slow.",
        "Please consider adding a referral program."
    ]

    # Iterate over test inputs and print categorized results
    for fb in samples:
        out = classify_and_respond(fb)
        print(f"Feedback: {fb}")
        print(f"→ Category: {out['category']}")
        print(f"→ Response: {out['response']}\n")
