from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import json

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Step 1: Create a raw prompt (instruct LLM to return JSON)
prompt = PromptTemplate(
    template="""
Extract the following info in strict JSON format with keys: name, age, email.

Text:
{text}

Respond only with JSON.
""",
    input_variables=["text"]
)

# Step 2: Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# Step 3: Create the chain manually (no parser)
chain = prompt | llm

# Step 4: Input to be processed
input_text = "Hi, my name is Priya Sharma. I'm very 20 and my email is priya.sharma@example.com."

# Step 5: Run and get the raw result
raw_response = chain.invoke({"text": input_text})
print("Raw LLM Output:\n", raw_response.content)

# Step 6: Try to parse JSON manually (unsafe if format is wrong)
try:
    data = json.loads(raw_response.content)
    print("\nParsed Data:\n", data)
except json.JSONDecodeError as e:
    print("\n❌ JSON parsing failed:", e)
