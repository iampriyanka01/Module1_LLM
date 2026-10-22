from google import genai
from google.genai import types
import os


# 1️⃣ Set Up API Key
# Use either GOOGLE_API_KEY or GEMINI_API_KEY env var
os.environ.setdefault("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))


client = genai.Client()

# 2️⃣ Define Real Tool Functions
def add(a: int, b: int) -> int:
    print(f"Executing add({a}, {b})")
    return a + b

def multiply(a: int, b: int) -> int:
    print(f"Executing multiply({a}, {b})")
    return a * b

available_functions = {"add": add, "multiply": multiply}

# 3️⃣ Declare Tool Schemas
add_decl = types.FunctionDeclaration(
    name="add",
    description="Adds two integers.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "a": types.Schema(type=types.Type.INTEGER, description="First integer"),
            "b": types.Schema(type=types.Type.INTEGER, description="Second integer"),
        },
        required=["a", "b"],
    ),
)

mul_decl = types.FunctionDeclaration(
    name="multiply",
    description="Multiplies two integers.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "a": types.Schema(type=types.Type.INTEGER, description="First integer"),
            "b": types.Schema(type=types.Type.INTEGER, description="Second integer"),
        },
        required=["a", "b"],
    ),
)

tools = [types.Tool(function_declarations=[add_decl, mul_decl])]


# 4️⃣ Common Config
config = types.GenerateContentConfig(tools=tools)

# 5️⃣ Prepare Prompt
user_prompt = "What is 3 * 12? Also, what is 11 + 49?"
messages = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]


# 6️⃣ First Call: Model may request function(s)
print(">>> Asking model...")
response_1 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=messages,
    config=config
)

# 7️⃣ Inspect model's response
content_1 = response_1.candidates[0].content
messages.append(content_1)

# 8️⃣ Execute any function_calls
tool_outputs = []
for part in content_1.parts:
    if part.function_call:
        name = part.function_call.name
        args = part.function_call.args
        print(f"> Function call: {name} with args {args}")
        fn = available_functions.get(name)
        try:
            result = fn(**args)
            tool_outputs.append(
                types.Part.from_function_response(name=name, response={"result": result})
            )
        except Exception as e:
            tool_outputs.append(
                types.Part.from_function_response(name=name, response={"error": str(e)})
            )


# 9️⃣ If functions got called, run second turn:
if tool_outputs:
    messages.append(types.Content(role="tool", parts=tool_outputs))
    print(">>> Providing function outputs back to model...")
    response_2 = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        config=config
    )
    final = response_2.candidates[0].content
    # print(final)
    print("\n Final Answer:")
    print("".join(part.text or "" for part in final.parts))
else:
    # No functions—just print model text
    print("\nResponse:")
    print("".join(part.text or "" for part in content_1.parts))