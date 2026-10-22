import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, ValidationError
from langchain.prompts import PromptTemplate
import asyncio

# Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Please set it in your .env file.")

# 1. Pydantic Model
class Resume(BaseModel):
    name: str
    skills: list[str]
    experience: int = Field(..., ge=0, description="Years of experience")

# 2. Output Parser
parser = PydanticOutputParser(pydantic_object=Resume)

# 3. Prompt Template
prompt = PromptTemplate(
    template=(
        "Extract the following fields from the input text:\n"
        "- name (string)\n- skills (list of strings)\n- experience (integer in years)\n\n"
        "Text:\n{text}\n\n{format_instructions}"
    ),
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. Initialize Gemini Model with streaming
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    streaming=True
)

# 5. Batch Input: Multiple resume-style texts
resume_inputs = [
    "Rahul has four years of experience in Python and Machine Learning.",
    "Anjali worked on web development with JavaScript and React for 2 years.",
    "My name is Rakesh. I am an expert in Java and Spring Boot. 6 years total.",
    "Priya knows Python, SQL and data analytics. She has 3 years experience.",
    "Name: Vikram. Skills: C++, Rust. Experience: two years.",  # malformed number
]

# 6. Format prompts
batch_prompts = [prompt.format(text=text) for text in resume_inputs]

# 7. Async function to handle streaming + parsing
async def run():
    responses = await llm.abatch(batch_prompts)

    for i, raw_response in enumerate(responses):
        print(f"\n📄 Resume {i+1} Input: {resume_inputs[i]}")
        print("🔁 Streaming output:")
        full_text = ""

        async for chunk in llm.astream([{"role": "user", "content": batch_prompts[i]}]):
            print(chunk.content, end="", flush=True)
            full_text += chunk.content

        print("\n\n✅ Final streamed output:")
        try:
            result = parser.parse(full_text)
            print("🎯 Parsed & Validated:", result)
        except ValidationError as e:
            print("❌ Validation Failed:", e.errors())
        except Exception as e:
            print("⚠️ Error:", str(e))

# 8. Run the async loop
asyncio.run(run())
