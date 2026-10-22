from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Step 1: Define the Pydantic Model
class Person(BaseModel):
    name: str
    age: int
    email: EmailStr

# Step 2: Create the Output Parser
parser = PydanticOutputParser(pydantic_object=Person)

# Step 3: Manually format prompt string (no PromptTemplate)
input_text = "Hi, my name is Priya Sharma. I'm twenty and my email is priya.sharma@example.com."
# input_text = "Hi, my name is Priya Sharma. I'm twenty and my email is priya.sharmaexample.com."
format_instructions = parser.get_format_instructions()

final_prompt = f"""Extract person info from the following text:

{input_text}

{format_instructions}
"""

# Step 4: Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=api_key
)

# Step 5: Get LLM response
response = llm.invoke(final_prompt)

# Step 6: Parse and validate the output
try:
    structured = parser.parse(response.content)
    print("✅ Structured Output:")
    print(structured)
except Exception as e:
    print("❌ Validation failed:")
    print(e)

