# 1. Install required Libraries
# !pip install "langchain==1.2.7" langchain-google-genai google-generativeai

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
import os

# 2. Define tools (add & multiply)
@tool
def add(a: int, b: int) -> int:
    """Add two integers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers"""
    return a * b
    

# 3. Setup Gemini LLM (ChatGoogleGenerativeAI)
# set your API key (or pass api_key=... when instantiating)
os.environ["GOOGLE_API_KEY"] = ""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# 4. Create the agent (use create_agent instead of create_tool_calling_agent)
agent = create_agent(
    model=llm,               # pass the model instance (or a model string)
    tools=[add, multiply],   # list of tool functions (decorated with @tool)
    system_prompt="You are a helpful math assistant. Use tools when needed."
)

# 5. Run examples
resp1 = agent.invoke({"messages":[{"role":"user","content":"What is 12 plus 8?"}]})
print(resp1["messages"][-1].content)

resp2 = agent.invoke({"messages":[{"role":"user","content":"Multiply 7 and 6"}]})
print(resp2["messages"][-1].content)
