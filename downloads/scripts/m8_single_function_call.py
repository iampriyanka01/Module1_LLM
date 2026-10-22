
from google import genai
from google.genai import types
import requests
import os 

# -----------------------------
# 1. Set Your API Keys
# -----------------------------
# Replace with your own keys or use environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")  # set in .env / environment - never hard-code
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")  # set in .env / environment - never hard-code



# -----------------------------
# 2. Define the Real Function
# -----------------------------
def get_current_temperature(location):
    """Fetch real-time temperature for a given location using OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        return data['main']['temp']
    else:
        raise Exception(f"Error fetching weather: {data.get('message', 'Unknown error')}")


# -----------------------------
# 3. Define Function Schema for Gemini
# -----------------------------
weather_function = {
    "name": "get_current_temperature",
    "description": "Gets the current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name, e.g. San Francisco",
            },
        },
        "required": ["location"],
    },
}


# -----------------------------
# 4. Configure Gemini Client
# -----------------------------
client = genai.Client(api_key=GEMINI_API_KEY)
tools = types.Tool(function_declarations=[weather_function])
config = types.GenerateContentConfig(tools=[tools])


# -----------------------------
# 5. Send User Prompt to Gemini
# -----------------------------
user_prompt = "What's the temperature in  gurugram, haryana? mentioned data as well "


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_prompt,
    config=config,
)


# -----------------------------
# 6. Check for Function Call and Execute
# -----------------------------
first_part = response.candidates[0].content.parts[0]


if hasattr(first_part, "function_call") and first_part.function_call:
    function_call = first_part.function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")


    # Execute only if it's our weather function
    if function_call.name == "get_current_temperature":
        location = function_call.args.get("location")
        try:
            temperature = get_current_temperature(location)
            print(f"The current  temperature in {location} is {temperature}°C ")
        except Exception as e:
            print(f"Error: {e}")
else:
    # If Gemini doesn't call the function, just show its response
    print("No function call found. Gemini says:")
    print(response.text)
