import ast
import asyncio
import json
import operator
import os
import random
import time
import zlib
from datetime import date

import altair as alt
import pandas as pd
import requests
import streamlit as st
from pydantic import BaseModel, ValidationError

from core.llm import call_with_retry, get_client, get_model, llm_ready, log_call, make_config, types
from core.ui import analogy, cards, compare, definition, diagram, page_header, quiz, stepper, takeaway
from core.diagrams import show_dg

page_header(
    "m8",
    "Give the brain hands and eyes: tools, tool cards, the function-calling loop, multi-tool orchestration and failure handling.",
    [
        "Explain why knowledge cutoff makes tools necessary",
        "Classify tools: data retrieval, computation, action",
        "Write a tool definition: name, description, parameter schema, output structure",
        "Trace the full lifecycle: query → tool choice → JSON call → your backend runs it → final answer",
        "Orchestrate sequential and asynchronous (fan-out / fan-in) calls",
        "Handle failures: isolation, timeouts, fallbacks, schema validation, clear errors",
    ],
)

# ---------------------------------------------------------------------------
# Tool implementations (the "engine" behind each tool card)
# ---------------------------------------------------------------------------
CONDITIONS = ["clear sky", "few clouds", "light rain", "overcast", "thunderstorm", "haze", "sunny"]
RATES_TO_USD = {"USD": 1.0, "INR": 1 / 83.2, "EUR": 1 / 0.92, "GBP": 1 / 0.79, "JPY": 1 / 151.0, "AED": 1 / 3.67}
STOCKS = {"TSLA": 252.34, "AAPL": 226.10, "GOOGL": 171.50, "MSFT": 431.20, "INFY": 19.40, "RELIANCE.NS": 35.10}


def get_weather_info(city: str, date: str | None = None) -> dict:
    """Real OpenWeatherMap data if WEATHER_API_KEY is set, otherwise a deterministic mock."""
    key = os.getenv("WEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY")
    if key:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                         params={"q": city, "appid": key, "units": "metric"}, timeout=8)
        data = r.json()
        if r.status_code != 200:
            return {"error": f"Weather API error: {data.get('message', r.status_code)}"}
        return {"city": city, "temperature": f"{data['main']['temp']} °C", "condition": data["weather"][0]["description"],
                "humidity": f"{data['main']['humidity']}%", "source": "OpenWeatherMap (live)"}
    h = zlib.crc32(f"{city.lower()}{date or ''}".encode())
    return {"city": city, "date": date or "today", "temperature": f"{15 + h % 24} °C",
            "condition": CONDITIONS[h % len(CONDITIONS)], "humidity": f"{35 + h % 55}%", "source": "mock weather service"}


def get_stock_price(ticker: str) -> dict:
    t = ticker.upper()
    if t not in STOCKS:
        return {"error": f"Unknown ticker {t}. Known: {', '.join(STOCKS)}"}
    return {"ticker": t, "price": STOCKS[t], "currency": "USD", "source": "mock market data"}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    f, t = from_currency.upper(), to_currency.upper()
    if f not in RATES_TO_USD or t not in RATES_TO_USD:
        return {"error": f"Unsupported currency. Supported: {', '.join(RATES_TO_USD)}"}
    usd = float(amount) * RATES_TO_USD[f]
    return {"converted_amount": round(usd / RATES_TO_USD[t], 2), "currency": t, "rate": round(RATES_TO_USD[f] / RATES_TO_USD[t], 4),
            "source": "mock FX rates"}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg}


def calculate(expression: str) -> dict:
    """Safe arithmetic evaluator (no eval!)."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only numbers and + - * / ** % are allowed")
    try:
        return {"expression": expression, "result": round(ev(ast.parse(expression, mode="eval").body), 6)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


TOOL_IMPLS = {"get_weather_info": get_weather_info, "get_stock_price": get_stock_price,
              "convert_currency": convert_currency, "calculate": calculate}

TOOL_CARDS = [
    {"name": "get_weather_info",
     "description": "Fetches current or forecast weather for a given city and optional date. Use when the user asks about "
                    "weather, temperature, rain chances, conditions, or whether to carry an umbrella.",
     "parameters": {"type": "OBJECT", "properties": {
         "city": {"type": "STRING", "description": "Name of the city (e.g., Gurugram)"},
         "date": {"type": "STRING", "description": "Date in YYYY-MM-DD format (optional; default=today)"}},
         "required": ["city"]}},
    {"name": "get_stock_price",
     "description": "Fetches the latest stock price in USD for a ticker symbol such as TSLA or AAPL.",
     "parameters": {"type": "OBJECT", "properties": {"ticker": {"type": "STRING", "description": "Stock ticker symbol"}},
                    "required": ["ticker"]}},
    {"name": "convert_currency",
     "description": "Converts an amount from one currency to another (USD, INR, EUR, GBP, JPY, AED).",
     "parameters": {"type": "OBJECT", "properties": {
         "amount": {"type": "NUMBER", "description": "Amount to convert"},
         "from_currency": {"type": "STRING", "description": "ISO code, e.g. USD"},
         "to_currency": {"type": "STRING", "description": "ISO code, e.g. EUR"}},
         "required": ["amount", "from_currency", "to_currency"]}},
    {"name": "calculate",
     "description": "Evaluates an arithmetic expression exactly. Use for any maths: percentages, interest, totals.",
     "parameters": {"type": "OBJECT", "properties": {"expression": {"type": "STRING", "description": "e.g. 600000 * 1.2"}},
                    "required": ["expression"]}},
]

tabs = st.tabs(["Why tools?", "Types of tools", "🪪 Tool card anatomy", "🔄 Lifecycle", "🟢 Live tool loop",
                "🌦️ Your weather script, reviewed", "⚡ Async fan-out / fan-in", "🧯 Error handling", "🧾 Key takeaways", "✅ Knowledge check"])

# ---------------------------------------------------------------------------
with tabs[0]:
    definition("Knowledge cutoff", "a model only knows data up to its training date. Ask about today's weather, prices or "
               "news and it will confidently *guess*.")
    show_dg("m8:0")
    analogy("A brilliant brain locked in a room with old newspapers. It can reason and imagine, but can't see or touch "
            "today's world. Tools give it **eyes** (live data) and **hands** (actions).", title="Brain in a room")
    definition("Tool", "an external function, service or API the LLM can ask to call. The LLM sends structured input "
               "(JSON), the tool runs, and structured output comes back. **LLM = brain that thinks; tool = hands and eyes.**")
    cards([("📈 Stock analyzer", "Live prices, sudden movements, buy/sell signals"),
           ("🌦️ Activity planner", "Rain in Manali tomorrow → indoor ideas; sunny → treks"),
           ("🧳 Travel advisor", "Reviews + ratings + climate → day-by-day itinerary")], cols=3)

with tabs[1]:
    show_dg("m8:1")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 👀 Data retrieval\n*Eyes to see.* Fetch real-time or external data.\n\nGoogle/Bing search, weather, stock prices, news.")
    c2.markdown("### 🧮 Computation\n*A calculator brain.* LLMs predict maths instead of calculating.\n\nPython REPL, maths API, currency conversion.")
    c3.markdown("### 🖐️ Action\n*Hands to do things.* Change the world.\n\nCalendar, Gmail, booking APIs.")
    st.markdown("#### 🎮 Sort the request")
    reqs = {"Book me a 4★ hotel in Manali under ₹8,000": "Action", "What's the latest news on the RBI repo rate?": "Data retrieval",
            "Compound interest on ₹5,000 at 7% for 5 years": "Computation", "Schedule a meeting with Riya at 4pm": "Action",
            "Is it raining in Mumbai right now?": "Data retrieval"}
    r = st.selectbox("Request", list(reqs))
    g = st.radio("Tool type", ["Data retrieval", "Computation", "Action"], horizontal=True, index=None, key="m8_sort")
    if g:
        (st.success if g == reqs[r] else st.error)(f"{'Correct' if g == reqs[r] else 'Not quite'} — it's **{reqs[r]}**.")

with tabs[2]:
    st.subheader("A tool card is the label on the box — what the LLM sees")
    show_dg("m8:2")
    st.markdown("**Name** (identity) · **Description** (when to use it) · **Parameters schema** (input blueprint) · "
                "**Output structure** (expected result). Vague cards make the model guess — and guessing breaks apps.")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Name", "get_weather_info")
        desc = st.text_area("Description", TOOL_CARDS[0]["description"], height=110)
        params = st.data_editor(pd.DataFrame([
            {"param": "city", "type": "string", "required": True, "description": "Name of the city (e.g., Gurugram)"},
            {"param": "date", "type": "string", "required": False, "description": "YYYY-MM-DD (optional; default today)"},
        ]), num_rows="dynamic", width="stretch", key="m8_params",
            column_config={"type": st.column_config.SelectboxColumn(options=["string", "number", "integer", "boolean", "array"])})
    rows = [p for p in params.to_dict("records") if p.get("param")]
    card_json = {"name": name, "description": desc, "parameters": {
        "type": "object",
        "properties": {p["param"]: {"type": p["type"], "description": p.get("description") or ""} for p in rows},
        "required": [p["param"] for p in rows if p.get("required")]}}
    with c2:
        st.json(card_json)
    issues = []
    if not name.replace("_", "").isalnum():
        issues.append("Name should be a single identifier like `get_weather_info`.")
    if len(desc) < 40:
        issues.append("Description is short — say *what* it does and *when* to use it.")
    if not card_json["parameters"]["required"]:
        issues.append("No required parameters — the model may call it with nothing.")
    if any(not p.get("description") for p in rows):
        issues.append("Every parameter needs a description (format, example).")
    for i in issues:
        st.warning(i)
    if not issues:
        st.success("Clear, complete tool card ✔")
    st.markdown("#### Output structure (recommended)")
    st.code('{"temperature": "28°C", "condition": "Sunny", "humidity": "60%", "source": "WeatherAPI"}', language="json")
    compare("Tool card alone", "A beautiful weather UI with a “Get Forecast” button… that has no backend. All brochure, no engine.",
            "Tool card + implementation", "The card tells the LLM what's possible; **your Python function** actually calls the API.")
    st.code('''def get_weather_info(city, date=None):
    url = "https://api.openweathermap.org/data/2.5/weather"
    r = requests.get(url, params={"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}, timeout=8)
    data = r.json()
    if r.status_code != 200:
        raise Exception(f"Error fetching weather: {data.get('message', 'Unknown error')}")
    return {"temperature": f"{data['main']['temp']} °C", "condition": data["weather"][0]["description"],
            "humidity": f"{data['main']['humidity']}%", "source": "OpenWeatherAPI"}''', language="python")

with tabs[3]:
    stepper("fc_life", [
        ("User asks", "“What's the weather forecast for Gurugram today?” The LLM detects **intent** (weather) and an **information "
                      "gap** (needs live data): “I can't just guess — I need a tool.”"),
        ("Intent & tool matching", "It scans the tool cards: `get_weather_info` matches. Required `city` → “Gurugram”. "
                                   "Optional `date` → “today”."),
        ("LLM emits a structured call", "It **cannot run code** — it writes JSON instead:\n```json\n"
                                        '{"name": "get_weather_info", "arguments": {"city": "Gurugram", "date": "2025-07-17"}}\n```'),
        ("Your backend executes", "1. Read `name` → find the registered function\n2. Extract & **validate** `arguments`\n"
                                  "3. Call `get_weather_info(city=\"Gurugram\")` → `{\"temperature\": \"34°C\", \"conditions\": \"Sunny\", …}`\n\n"
                                  "**The LLM never executes anything — you do.** That keeps the system secure and controlled."),
        ("LLM writes the final answer", "The tool result goes back to the model, which turns JSON into language: "
                                        "“Gurugram is sunny with a high of 34°C… carry sunscreen!”"),
    ])
    diagram("""
digraph { rankdir=LR; node [shape=box style="rounded,filled" fontname="Helvetica"];
 U [label="User question" fillcolor="#f1f5f9"]; L1 [label="LLM\\n(chooses tool)" fillcolor="#dbeafe"]; J [label="function_call JSON" fillcolor="#fef9c3"];
 B [label="Your backend\\nruns Python / API" fillcolor="#dcfce7"]; R [label="function_response" fillcolor="#fef9c3"]; L2 [label="LLM\\n(final answer)" fillcolor="#dbeafe"];
 U->L1->J->B->R->L2; }""")

# ---------------------------------------------------------------------------
def run_tool_loop(question: str, enabled: list[str], max_rounds: int = 5):
    client, model = get_client(), get_model()
    decls = [c for c in TOOL_CARDS if c["name"] in enabled]
    cfg = make_config(tools=[types.Tool(function_declarations=decls)],
                      system=f"Today's date is {date.today().isoformat()}. Use the tools for any live data or maths. "
                             "If a tool returns an error, explain it plainly to the user.")
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    trace = []
    for rnd in range(1, max_rounds + 1):
        t0 = time.perf_counter()
        resp = call_with_retry(lambda: client.models.generate_content(model=model, contents=contents, config=cfg))
        log_call(question, resp, time.perf_counter() - t0, kind="tool-loop")
        calls = resp.function_calls or []
        if not calls:
            trace.append({"round": rnd, "type": "final", "text": resp.text or "(no text)"})
            return trace
        contents.append(resp.candidates[0].content)  # keep the model's turn (incl. thought signatures)
        parts = []
        for fc in calls:
            args = dict(fc.args or {})
            trace.append({"round": rnd, "type": "call", "name": fc.name, "args": args})
            fn = TOOL_IMPLS.get(fc.name)
            try:
                result = fn(**args) if fn else {"error": f"unknown tool {fc.name}"}
            except Exception as e:  # noqa: BLE001 - isolate tool failures
                result = {"error": f"{type(e).__name__}: {e}"}
            trace.append({"round": rnd, "type": "result", "name": fc.name, "result": result})
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))
    trace.append({"round": max_rounds, "type": "final", "text": "(stopped: too many tool rounds)"})
    return trace


def show_trace(trace):
    for step in trace:
        if step["type"] == "call":
            with st.container(border=True):
                st.markdown(f"**Round {step['round']} · 🤖 LLM requests a tool call** (it does not run it)")
                st.code(json.dumps({"name": step["name"], "arguments": step["args"]}, indent=2), language="json")
        elif step["type"] == "result":
            with st.container(border=True):
                st.markdown(f"**Round {step['round']} · 🐍 Your backend ran `{step['name']}()` and sends back**")
                st.code(json.dumps(step["result"], indent=2, ensure_ascii=False), language="json")
        else:
            st.success(f"**Final answer (round {step['round']}):**\n\n{step['text']}")


with tabs[4]:
    st.subheader("Watch Gemini choose tools, your code run them, and the answer come back")
    show_dg("m8:4")
    weather_src = "OpenWeatherMap (WEATHER_API_KEY found)" if (os.getenv("WEATHER_API_KEY") or os.getenv("OPENWEATHER_API_KEY")) else "mock weather (add WEATHER_API_KEY to .env for live data)"
    st.caption(f"Weather source: {weather_src} · stock & FX data are mock services so results are reproducible.")
    presets = {
        "Single call — weather": "What's the weather in Gurugram today? Should I carry an umbrella?",
        "Sequential — stock in EUR": "What is Tesla's current stock price in EUR?",
        "Parallel — trip check": "I'm going to Manali and Goa next weekend. Compare the weather in both, and convert 500 USD to INR.",
        "Computation — salary hike": "My salary is ₹6,00,000 per year. What will it be after a 20% hike, and what is that per month?",
        "No tool needed": "Explain in one line what a function call is.",
    }
    pick = st.radio("Example", list(presets), horizontal=True, key="m8_pick")
    q = st.text_input("Question", presets[pick], key=f"m8_q_{pick}")
    enabled = st.multiselect("Tools available to the model", list(TOOL_IMPLS), default=list(TOOL_IMPLS), key="m8_tools")
    with st.expander("Tool cards sent to Gemini"):
        st.json([c for c in TOOL_CARDS if c["name"] in enabled])
    if st.button("Run the function-calling loop", type="primary", disabled=not llm_ready(False)):
        with st.spinner("Running…"):
            try:
                st.session_state.m8_trace = run_tool_loop(q, enabled)
            except Exception as e:  # noqa: BLE001
                from core.llm import friendly_error
                st.session_state.m8_trace = None
                st.error(friendly_error(e))
    llm_ready()
    if st.session_state.get("m8_trace"):
        show_trace(st.session_state.m8_trace)
    with st.expander("📼 Pre-recorded example (no key needed): Tesla price in EUR — sequential calls"):
        show_trace([
            {"round": 1, "type": "call", "name": "get_stock_price", "args": {"ticker": "TSLA"}},
            {"round": 1, "type": "result", "name": "get_stock_price", "result": {"ticker": "TSLA", "price": 252.34, "currency": "USD"}},
            {"round": 2, "type": "call", "name": "convert_currency", "args": {"amount": 252.34, "from_currency": "USD", "to_currency": "EUR"}},
            {"round": 2, "type": "result", "name": "convert_currency", "result": {"converted_amount": 232.15, "currency": "EUR"}},
            {"round": 3, "type": "final", "text": "Tesla's stock is $252.34, which is approximately €232.15."},
        ])
        st.caption("Round 2 depends on round 1's output — that's why these calls must be **sequential**.")
    st.code('''tools = types.Tool(function_declarations=[weather_card, stock_card, fx_card])
config = types.GenerateContentConfig(tools=[tools])
contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]

while True:
    resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
    if not resp.function_calls:                      # no tool needed → final answer
        print(resp.text); break
    contents.append(resp.candidates[0].content)      # remember what the model asked for
    parts = []
    for fc in resp.function_calls:                   # may be several (parallel calls)
        result = TOOLS[fc.name](**fc.args)           # YOUR code runs the function
        parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
    contents.append(types.Content(role="user", parts=parts))   # send results back''', language="python")

# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("`3_weatherSearchaddingLLM.py` — what works and what to fix")
    show_dg("m8:5")
    st.markdown("Your script correctly defines a tool card, passes it via `types.Tool(function_declarations=[...])`, and reads "
                "`function_call.name` / `.args`. Five things to improve:")
    fixes = [
        ("1 · Secrets printed", "`print(\"Gemini key:\", GEMINI_API_KEY)` leaks both keys to the console/logs.", "Print `bool(key)` or nothing."),
        ("2 · Crash when no tool is called", "`first_part.function_call.name` raises `AttributeError` if the model answers in text.",
         "Check `if response.function_calls:` first."),
        ("3 · Loop never closes", "The temperature is printed by your own f-string; the result is **never sent back** to the LLM, so it "
                                  "can't write a grounded, conversational answer (“carry an umbrella?”).", "Append `Part.from_function_response` and call the model again."),
        ("4 · Card ≠ implementation", "The card promises *forecast* + optional `date`, but the function ignores date and returns only temp.",
         "Make the description match what the function really does, or implement date."),
        ("5 · No timeout / error isolation", "`requests.get(url)` can hang; an exception kills the script.",
         "`timeout=8`, try/except, return `{\"error\": ...}` to the LLM."),
    ]
    for t, prob, fix in fixes:
        with st.container(border=True):
            st.markdown(f"**{t}** — {prob}  \n✅ *Fix:* {fix}")
    st.markdown("#### Improved version")
    st.code('''from google import genai
from google.genai import types
import requests, os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
WEATHER_KEY = os.getenv("WEATHER_API_KEY")
MODEL = "gemini-3.5-flash-lite"

def get_current_weather(city: str) -> dict:
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                         params={"q": city, "appid": WEATHER_KEY, "units": "metric"}, timeout=8)
        data = r.json()
        if r.status_code != 200:
            return {"error": data.get("message", "weather service error")}
        return {"temperature_c": data["main"]["temp"], "condition": data["weather"][0]["description"],
                "humidity": data["main"]["humidity"]}
    except requests.RequestException as e:
        return {"error": f"weather service unreachable: {e}"}

weather_tool = {
    "name": "get_current_weather",
    "description": "Gets CURRENT weather (temperature, condition, humidity) for a city. Use for questions about "
                   "weather, temperature, rain or whether to take an umbrella.",
    "parameters": {"type": "OBJECT",
                   "properties": {"city": {"type": "STRING", "description": "City name, e.g. Pune"}},
                   "required": ["city"]},
}
config = types.GenerateContentConfig(tools=[types.Tool(function_declarations=[weather_tool])])

city = input("Enter location: ")
contents = [types.Content(role="user", parts=[types.Part.from_text(text=f"What is the weather in {city}? Do I need an umbrella?")])]
response = client.models.generate_content(model=MODEL, contents=contents, config=config)

if response.function_calls:                                   # fix 2
    fc = response.function_calls[0]
    result = get_current_weather(**fc.args)                    # your code runs the tool
    contents.append(response.candidates[0].content)
    contents.append(types.Content(role="user", parts=[
        types.Part.from_function_response(name=fc.name, response={"result": result})]))
    response = client.models.generate_content(model=MODEL, contents=contents, config=config)   # fix 3

print("LLM:", response.text)''', language="python")

# ---------------------------------------------------------------------------
class WeatherData(BaseModel):
    temperature: str
    condition: str


async def fake_api(name: str, latency: float, fails: bool, corrupt: bool):
    t0 = time.perf_counter()
    await asyncio.sleep(latency)
    if fails:
        raise ConnectionError(f"{name} API returned 503")
    if corrupt:
        return {"temp": 28}  # wrong schema on purpose
    return {"weather": {"temperature": "18°C", "condition": "Cloudy"},
            "hotels": [{"name": "Snow Valley Resort"}, {"name": "Apple Country Inn"}],
            "currency": {"USD": "83.12 INR"}}[name]


async def orchestrate(specs: dict, timeout: float, parallel: bool):
    timeline, results = [], {}
    t_start = time.perf_counter()

    async def guarded(name, spec):
        s = time.perf_counter() - t_start
        try:
            data = await asyncio.wait_for(fake_api(name, spec["latency"], spec["fail"], spec["corrupt"]), timeout=timeout)
            if name == "weather":
                WeatherData(**data)  # schema validation
            status, out = "ok", data
        except asyncio.TimeoutError:
            status, out = "timeout", {"error": f"{name} API timed out after {timeout:.1f}s"}
        except ValidationError:
            status, out = "invalid", {"error": f"{name} API returned an invalid schema"}
        except Exception as e:  # noqa: BLE001
            status, out = "failed", {"error": str(e)}
        if status != "ok" and spec.get("fallback"):
            out = {**FALLBACKS[name], "note": out["error"]}
            status += " → fallback"
        timeline.append({"api": name, "start": s, "end": time.perf_counter() - t_start, "status": status})
        results[name] = out

    if parallel:
        await asyncio.gather(*(guarded(n, s) for n, s in specs.items()))   # fan-out … fan-in
    else:
        for n, s in specs.items():
            await guarded(n, s)
    return timeline, results, time.perf_counter() - t_start


STATUS_ORDER = ["ok", "failed → fallback", "timeout → fallback", "invalid → fallback", "failed", "timeout", "invalid"]
STATUS_COLORS = ["#22c55e", "#f59e0b", "#f59e0b", "#f59e0b", "#ef4444", "#ef4444", "#ef4444"]

FALLBACKS = {"weather": {"temperature": "N/A", "condition": "Data unavailable", "source": "fallback cache (yesterday)"},
             "hotels": {"suggestions": ["Popular pick: Hotel Manali Heights"], "source": "static default"},
             "currency": {"USD": "~83 INR", "source": "cached rate"}}

with tabs[6]:
    st.subheader("Fan-out, fan-in: call independent tools at the same time")
    show_dg("m8:6")
    st.markdown("“Plan a trip to Manali: weather, hotels and USD→INR rate.” The three calls don't depend on each other, so the "
                "backend fires them **concurrently** (`asyncio.gather`) and waits only for the slowest.")
    compare("Use async when…", "2+ **independent** I/O-bound API calls · user-facing latency matters · you want partial results early",
            "Don't bother when…", "Only 1 tool · calls depend on each other (stock → currency) · CPU-heavy work (use multiprocessing)", bad_good=False)
    specs = {}
    cols = st.columns(3)
    for col, (name, lat) in zip(cols, [("weather", 1.0), ("hotels", 2.0), ("currency", 0.6)]):
        with col.container(border=True):
            st.markdown(f"**{name} API**")
            specs[name] = {"latency": st.slider("latency (s)", 0.1, 4.0, lat, 0.1, key=f"lat_{name}"),
                           "fail": st.checkbox("fails (503)", key=f"fail_{name}"),
                           "corrupt": st.checkbox("bad schema", key=f"cor_{name}") if name == "weather" else False,
                           "fallback": st.checkbox("use fallback data", True, key=f"fb_{name}")}
    timeout = st.slider("Per-call timeout (s)", 0.5, 5.0, 3.0, 0.5)
    if st.button("▶ Run sequential vs parallel", type="primary"):
        with st.spinner("Calling three mock APIs twice…"):
            seq = asyncio.run(orchestrate(specs, timeout, parallel=False))
            par = asyncio.run(orchestrate(specs, timeout, parallel=True))
        st.session_state.m8_async = (seq, par)
    if st.session_state.get("m8_async"):
        seq, par = st.session_state.m8_async
        c1, c2 = st.columns(2)
        c1.metric("Sequential total", f"{seq[2]:.2f}s")
        c2.metric("Parallel total", f"{par[2]:.2f}s", f"{seq[2] - par[2]:.2f}s faster", delta_color="normal")
        tl = pd.concat([pd.DataFrame(seq[0]).assign(mode="sequential"), pd.DataFrame(par[0]).assign(mode="parallel (async)")])
        chart = alt.Chart(tl).mark_bar(cornerRadius=4).encode(
            x=alt.X("start:Q", title="seconds"), x2="end:Q", y=alt.Y("api:N", title=None),
            color=alt.Color("status:N", scale=alt.Scale(domain=STATUS_ORDER, range=STATUS_COLORS)), row=alt.Row("mode:N", title=None),
            tooltip=["api", "status", alt.Tooltip("start:Q", format=".2f"), alt.Tooltip("end:Q", format=".2f")],
        ).properties(height=110, width=600)
        st.altair_chart(chart)
        st.markdown("**Aggregated payload sent back to the LLM (fan-in):**")
        st.json(par[1])

with tabs[7]:
    st.subheader("More tools, more failure points — fail at the tool level, not the app level")
    show_dg("m8:7")
    principles = [
        ("1 · Isolate failures", "Wrap each call in try/except; return `{\"error\": ...}` for that tool, keep the others."),
        ("2 · Timeouts, aggressively", "Per-request and overall timeouts — never trust an external API to be fast."),
        ("3 · Fallback data", "Cached value, alternative provider, or an estimate — with a disclaimer."),
        ("4 · Validate schema", "Pydantic models catch changed/missing fields before they corrupt your pipeline."),
        ("5 · Aggregate results", "Merge successes and failures into one clear payload for the LLM."),
        ("6 · Surface clear errors to the LLM", "So it can say: “I couldn't fetch hotels — the service timed out. Want popular picks instead?”"),
    ]
    cards(principles, cols=3)
    st.code('''{
  "weather":  {"temperature": "28°C", "condition": "Sunny"},
  "hotels":   {"error": "Hotel API timed out after 10s"},
  "currency": {"USD": "83.2 INR"}
}''', language="json")
    st.caption("Try it in the Async tab: tick “fails” or “bad schema”, untick fallback, and watch the payload.")


# ---------------------------------------------------------------------------
with tabs[8]:
    takeaway([
        "Tools overcome knowledge cutoff: retrieval (eyes), computation (calculator), action (hands).",
        "A tool card = name + description + parameter schema (+ output structure); your function is the engine.",
        "The LLM only emits a JSON call; your backend executes it and sends the result back for the final answer.",
        "Dependent calls run sequentially; independent ones fan out in parallel and fan in.",
        "Isolate failures, use timeouts, fall back gracefully, validate schemas, and tell the LLM what failed.",
    ])


with tabs[9]:
    quiz("m8", [
        {"q": "Who actually executes the function in function calling?", "options": ["The LLM", "Your backend code", "The user", "The API key"],
         "answer": 1, "why": "The model only suggests a structured call."},
        {"q": "Which tool-card part helps the model decide WHEN to use a tool?", "options": ["Name", "Description", "Required list", "Output structure"],
         "answer": 1, "why": "The description is the tool's job description."},
        {"q": "“Tesla price in EUR” needs stock → currency. These calls should be…", "options": ["Parallel", "Sequential", "Skipped", "Cached forever"],
         "answer": 1, "why": "The conversion needs the price first."},
        {"q": "Weather and currency succeed but hotels times out. Best response?", "options": ["Show an error page",
         "Return partial results and explain hotels are unavailable", "Retry forever", "Hide the failure"], "answer": 1,
         "why": "Isolate failures and communicate them clearly."},
    ])
