"""Generate the Google-Colab notebooks in downloads/notebooks/.

    python tools/build_notebooks.py

Every notebook:
  * installs what it needs with `!pip install`
  * reads keys from Colab Secrets (left sidebar -> key icon), falling back to a hidden getpass prompt
  * never contains a key
Projects are written into the Colab VM with `%%writefile` from the (key-free) files in downloads/projects.
"""
from __future__ import annotations

import csv
import io
import sys
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DL = ROOT / "downloads"
NB = DL / "notebooks"
NB.mkdir(parents=True, exist_ok=True)

MODEL_NOTE = '# Change to "gemini-2.5-flash" if your key does not have access to the newer model.\nMODEL = "gemini-3.5-flash-lite"'

SETUP_KEYS = '''\
import os, getpass

def get_secret(name, *, required=True, hint=""):
    """1) environment  2) Colab Secrets (key icon in the left sidebar)  3) hidden prompt.
    The value is kept in memory / os.environ only - it is never printed or saved in the notebook."""
    val = os.getenv(name)
    if not val:
        try:
            from google.colab import userdata          # works only inside Colab
            val = userdata.get(name)
        except Exception:
            val = None
    if not val and required:
        val = getpass.getpass(f"Paste {name} {hint}(input is hidden): ").strip()
    if val:
        os.environ[name] = val
    return val
'''


def md(t):
    return nbf.v4.new_markdown_cell(textwrap.dedent(t).strip())


def code(t):
    return nbf.v4.new_code_cell(textwrap.dedent(t).strip("\n"))


def header(title, intro, secrets, pip):
    rows = "\n".join(f"| `{n}` | {w} | {u} |" for n, w, u in secrets)
    return [
        md(f"""# {title}

{intro}

> 🔐 **Keys** — nothing in this notebook contains a key. In Colab open the **🔑 Secrets** panel (left sidebar), click **+ Add new secret**,
> type the *name* exactly as below, paste the *value*, and switch on **Notebook access**. If you skip this, you'll be asked once (hidden input).

| Secret name | What it is | Get it at |
|---|---|---|
{rows}

**Never** paste a key into a code cell, commit it to GitHub, or share a notebook that shows it in an output. If one leaks, delete it in the provider's console and create a new one.
"""),
        code(f"!pip install -q {pip}"),
        code(SETUP_KEYS),
    ]


def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"colab": {"provenance": [], "name": name}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    nbf.validate(nb)
    nbf.write(nb, NB / name)
    print("wrote", name, len(cells), "cells")


GEMINI = ("GEMINI_API_KEY", "Google Gemini API key", "https://aistudio.google.com/apikey")
OWM = ("OPENWEATHER_API_KEY", "OpenWeatherMap key (free plan)", "https://home.openweathermap.org/api_keys")
GROQ = ("GROQ_API_KEY", "Groq API key (LLM for CineMate)", "https://console.groq.com/keys")
TMDB = ("TMDB_API_KEY", "TMDB **v4 Read Access Token** (long `eyJ…` string, not the short v3 key)", "https://www.themoviedb.org/settings/api")
OMDB = ("OMDB_API_KEY", "OMDb key (free: 1,000 requests/day, emailed to you)", "https://www.omdbapi.com/apikey.aspx")

CLIENT = '''\
from google import genai
from google.genai import types

client = genai.Client(api_key=get_secret("GEMINI_API_KEY", hint="(from aistudio.google.com/apikey) "))
''' + MODEL_NOTE + "\nprint('Ready. Model =', MODEL)"


# ============================================================================ M6
def nb_m6():
    c = header("Module 6 · Advanced handling of LLM responses (streaming, batching, structured output)",
               "Run each section top to bottom. Every cell is self-contained, so change the prompt or the numbers and re-run it.",
               [GEMINI], "google-genai pydantic langchain-core langchain-google-genai")
    c += [code(CLIENT),
          md("## 1 · Normal (blocking) call vs streaming\nWith a normal call you wait for the whole answer. With streaming the **first words** arrive early — "
             "total time is similar, but the *perceived* wait is much shorter."),
          code('''\
import time
PROMPT = "Explain how a bill becomes law in India in about 150 words."

t0 = time.perf_counter()
r = client.models.generate_content(model=MODEL, contents=PROMPT)
print(f"BLOCKING: whole answer after {time.perf_counter()-t0:.2f}s")
print(r.text[:200], "...\\n")

t0 = time.perf_counter(); first = None; text = ""
for chunk in client.models.generate_content_stream(model=MODEL, contents=PROMPT):
    if first is None:
        first = time.perf_counter() - t0
    text += chunk.text or ""
print(f"STREAMING: first words after {first:.2f}s, finished after {time.perf_counter()-t0:.2f}s")'''),
          md("## 2 · Async streaming and async batching\n`await` lets many requests overlap. Total time ≈ the slowest call, not the sum."),
          code('''\
import asyncio, time

QUESTIONS = ["What is an API?", "What is a token?", "What is a prompt?", "What is temperature?", "What is RAG?"]

async def ask(q):
    r = await client.aio.models.generate_content(model=MODEL, contents=q + " Answer in one sentence.")
    return q, r.text.strip()

async def run_all():
    return await asyncio.gather(*(ask(q) for q in QUESTIONS))

# Colab already runs an event loop, so we simply `await`
t0 = time.perf_counter()
results = await run_all()
print(f"{len(QUESTIONS)} questions concurrently in {time.perf_counter()-t0:.2f}s")
for q, a in results:
    print("-", q, "->", a)

# sequential, for comparison
t0 = time.perf_counter()
for q in QUESTIONS:
    client.models.generate_content(model=MODEL, contents=q + " Answer in one sentence.")
print(f"sequential: {time.perf_counter()-t0:.2f}s")'''),
          md("## 3 · Batching many inputs into ONE prompt\nFewer requests → lower cost and fewer rate-limit hits. Ask for numbered output so you can split it again."),
          code('''\
import json, re
TICKETS = [
    "My order #4821 arrived broken.", "How do I change my password?", "Love the new dark mode!",
    "I was charged twice this month.", "Can you add an export-to-PDF button?", "The app crashes when I upload a photo.",
]
numbered = "\\n".join(f"{i+1}. {t}" for i, t in enumerate(TICKETS))
prompt = f"""Classify each support ticket as complaint, question, praise or suggestion.
Return ONLY a JSON list of objects: [{{"id": 1, "category": "..."}}, ...]

Tickets:
{numbered}"""
print(prompt, "\\n")
raw = client.models.generate_content(model=MODEL, contents=prompt).text
print(raw)
data = json.loads(re.search(r"\\[.*\\]", raw, re.S).group())
for d in data:
    print(d["id"], TICKETS[d["id"]-1][:40].ljust(42), "->", d["category"])'''),
          md("## 4 · Structured output with Pydantic (validate, don't trust)\nThe model is told the schema; **Pydantic** checks the result. If validation fails, handle it instead of crashing."),
          code('''\
from pydantic import BaseModel, Field, ValidationError
from typing import List

class Resume(BaseModel):
    name: str
    skills: List[str]
    experience: int = Field(..., ge=0, description="Years of experience")

texts = [
    "Rahul has four years of experience in Python and Machine Learning.",
    "Anjali worked on web development with JavaScript and React for 2 years.",
    "Name: Vikram. Skills: C++, Rust. Experience: two years.",   # words instead of a number
]
for t in texts:
    r = client.models.generate_content(
        model=MODEL, contents="Extract the resume fields from: " + t,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Resume))
    try:
        print("OK  ", Resume.model_validate_json(r.text))
    except ValidationError as e:
        print("FAIL", t, "->", e.errors()[0]["msg"])'''),
          md("## 5 · The same idea with LangChain's `PydanticOutputParser` and `abatch`"),
          code('''\
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

parser = PydanticOutputParser(pydantic_object=Resume)
prompt = PromptTemplate(
    template="Extract the following fields from the text.\\n{format_instructions}\\n\\nText:\\n{text}",
    input_variables=["text"], partial_variables={"format_instructions": parser.get_format_instructions()})
llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=os.environ["GEMINI_API_KEY"])
chain = prompt | llm | parser

outs = await chain.abatch([{"text": t} for t in texts], return_exceptions=True)
for t, o in zip(texts, outs):
    print(type(o).__name__, "|", o if not isinstance(o, Exception) else str(o)[:120])'''),
          md("## 6 · Mini project — customer-feedback assistant\nClassify the feedback **and** draft a polite reply, with a safe fallback if the model's format is wrong."),
          code('''\
def classify_and_respond(feedback: str) -> dict:
    prompt = f"""You are a customer support assistant.
Classify the feedback as exactly one of: complaint, suggestion, praise.
Then write a short polite reply.

Feedback: "{feedback}"

Format:
Category: <one word>
Response: <your reply>"""
    text = client.models.generate_content(model=MODEL, contents=prompt).text.strip().splitlines()
    try:
        return {"category": text[0].split(":", 1)[1].strip(), "response": text[1].split(":", 1)[1].strip()}
    except Exception:
        return {"category": "unknown", "response": "(could not parse the model output)", "raw": text}

for fb in ["I love the service!", "Your app crashes every time I open it.", "Could you add a dark mode to the app?"]:
    print(fb, "->", classify_and_respond(fb), "\\n")''')]
    save("M6_response_handling.ipynb", c)


# ============================================================================ M7
def nb_m7():
    c = header("Module 7 · Conversational interfaces: memory and context management",
               "An LLM call is **stateless**. A chatbot feels like it remembers only because *you* send the history every time.",
               [GEMINI], "google-genai openai")
    c += [code(CLIENT),
          md("## 1 · No context: the model forgets"),
          code('''\
for msg in ["I have 2 dogs in my home.", "How many paws are there in my home?"]:
    r = client.models.generate_content(model=MODEL, contents=msg)
    print("YOU:", msg); print("AI :", r.text.strip()[:300], "\\n")'''),
          md("## 2 · Manual history: you resend everything"),
          code('''\
history = []
def chat_manual(user_text):
    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    r = client.models.generate_content(model=MODEL, contents=history)
    history.append(types.Content(role="model", parts=[types.Part(text=r.text)]))
    return r.text

print(chat_manual("I have 2 dogs in my home."))
print(chat_manual("How many paws are there in my home?"))
print("messages sent on turn 2:", len(history))'''),
          md("## 3 · `client.chats.create` — the SDK keeps the history for you"),
          code('''\
chat = client.chats.create(model=MODEL)
print(chat.send_message("I have 2 dogs in my home.").text)
print(chat.send_message("How many paws are there in my home?").text)
print("\\n--- history ---")
for m in chat.get_history():
    print(m.role.upper().ljust(5), m.parts[0].text[:90].replace("\\n", " "))'''),
          md("""## 4 · Context management with summarisation
Long chats cost more and eventually overflow the context window. Strategy: when the history passes `SUMMARY_THRESHOLD` messages,
**summarise the old part** and keep only the last few turns verbatim.

Edit the numbers and re-run to see how many messages are actually sent."""),
          code('''\
SUMMARY_THRESHOLD = 6      # summarise once the history is longer than this
KEEP_LAST = 2              # ... and keep this many recent messages untouched

history, summary = [], ""

def summarise(old_messages, previous_summary):
    text = "\\n".join(f"{m['role']}: {m['text']}" for m in old_messages)
    prompt = (f"Previous summary: {previous_summary or '(none)'}\\n\\nNew conversation:\\n{text}\\n\\n"
              "Update the summary in at most 2 sentences. Keep names, numbers and preferences.")
    return client.models.generate_content(model=MODEL, contents=prompt).text.strip()

def send(user_text):
    global history, summary
    history.append({"role": "user", "text": user_text})
    if len(history) > SUMMARY_THRESHOLD:
        old, history = history[:-KEEP_LAST], history[-KEEP_LAST:]
        summary = summarise(old, summary)
    parts = []
    if summary:
        parts.append(types.Content(role="user", parts=[types.Part(text=f"(Summary of earlier conversation: {summary})")]))
        parts.append(types.Content(role="model", parts=[types.Part(text="Understood.")]))
    for m in history:
        parts.append(types.Content(role=m["role"], parts=[types.Part(text=m["text"])]))
    r = client.models.generate_content(model=MODEL, contents=parts)
    history.append({"role": "model", "text": r.text})
    print(f"[sent {len(parts)} messages | summary: {'yes' if summary else 'no'}]")
    return r.text

for q in ["Hi, I'm Priya and I live in Pune.", "I love trekking.", "My favourite food is poha.", "I have a dog called Bruno.",
          "I work as a data analyst.", "What do you remember about me?"]:
    print("YOU:", q); print("AI :", send(q).strip()[:250], "\\n")
print("Summary now:", summary)'''),
          md("## 5 · Multi-turn chat through the OpenAI Python library\nGemini also speaks the OpenAI protocol, so code written for `openai` works with a different `base_url` and your Gemini key."),
          code('''\
from openai import OpenAI
oa = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
messages = [{"role": "system", "content": "You are a concise, friendly assistant."}]
for q in ["My name is Priya.", "What's my name?"]:
    messages.append({"role": "user", "content": q})
    a = oa.chat.completions.create(model=MODEL, messages=messages).choices[0].message.content
    messages.append({"role": "assistant", "content": a})
    print("YOU:", q); print("AI :", a.strip(), "\\n")''')]
    save("M7_conversational_context.ipynb", c)


# ============================================================================ M7 chatbot
def nb_m7_chatbot():
    c = header("Module 7 · Auto-summary chatbot (context management)",
               "Every N user turns a separate call writes a 2-sentence summary. In *Summary + last K* mode the model receives only that summary plus the latest messages, "
               "so the prompt stops growing. This is the Colab version of the app's chatbot tab.",
               [GEMINI], "google-genai pandas")
    c += [code(CLIENT),
          md("## 1 · Settings (edit and re-run)"),
          code('SUMMARIZE_EVERY = 3     # write a summary every N user turns\nKEEP_LAST = 4           # messages kept verbatim in summary mode\nMODE = "summary"        # "full" or "summary"'),
          md("## 2 · The chatbot"),
          code('''\
import pandas as pd
msgs, summary, turns, stats = [], "", 0, []

def contents_to_send():
    recent = msgs if (MODE == "full" or not summary) else msgs[-KEEP_LAST:]
    pre = []
    if MODE != "full" and summary:
        pre = [types.Content(role="user", parts=[types.Part(text=f"(Context summary of earlier conversation: {summary})")]),
               types.Content(role="model", parts=[types.Part(text="Understood.")])]
    return pre + [types.Content(role=m["role"], parts=[types.Part(text=m["text"])]) for m in recent]

def chat(user_text):
    global summary, turns
    msgs.append({"role": "user", "text": user_text}); turns += 1
    sent = contents_to_send()
    r = client.models.generate_content(model=MODEL, contents=sent)
    msgs.append({"role": "model", "text": r.text})
    tokens = getattr(r.usage_metadata, "prompt_token_count", None)
    stats.append({"turn": turns, "mode": "summary + recent" if (MODE != "full" and summary) else "full history",
                  "messages sent": len(sent), "tokens sent": tokens})
    if turns % SUMMARIZE_EVERY == 0:
        convo = "\\n".join(f"{m['role']}: {m['text']}" for m in msgs)
        prev = f"Previous summary: {summary}\\n\\n" if summary else ""
        summary = client.models.generate_content(
            model=MODEL, contents=prev + "Summarize the conversation below in AT MOST 2 short sentences (under 45 words). "
                                          "Keep names, numbers and preferences.\\n\\n" + convo).text.strip()
    return r.text'''),
          md("## 3 · Scripted demo (no typing needed)"),
          code('''\
for q in ["Hi, I'm Priya from Pune.", "I love trekking and poha.", "I have a dog called Bruno.", "I work as a data analyst.",
          "I'm planning a trip in December.", "What do you remember about me?"]:
    print("YOU:", q); print("AI :", chat(q).strip()[:300], "\\n")
print("SUMMARY:", summary)
pd.DataFrame(stats)'''),
          md("## 4 · Chat yourself\nRun the cell, type a message, press Enter. Type `quit` to stop."),
          code('''\
while True:
    t = input("You: ").strip()
    if t.lower() in {"quit", "exit", ""}:
        break
    print("AI :", chat(t).strip(), "\\n")
pd.DataFrame(stats)'''),
          md('Compare: set `MODE = "full"`, re-run sections 2–3 and watch **tokens sent** keep climbing, while summary mode levels off.')]
    save("M7_auto_summary_chatbot.ipynb", c)


# ============================================================================ M8
def nb_m8():
    c = header("Module 8 · Function calling and tool integration",
               "The model never runs your code. It returns a **function call** (name + arguments); *you* run it and send the result back.",
               [GEMINI, OWM], "google-genai requests langchain-core langchain-google-genai")
    c += [code(CLIENT),
          md("## 1 · Single tool: live weather\n**New OpenWeatherMap key?** It can take 10–120 minutes to activate; until then you'll see a 401 error."),
          code('''\
import requests
OWM_KEY = get_secret("OPENWEATHER_API_KEY", hint="(from home.openweathermap.org/api_keys) ")

def get_current_temperature(location: str) -> float:
    """Return the current temperature in Celsius for a city, e.g. 'Gurugram,IN'."""
    r = requests.get("https://api.openweathermap.org/data/2.5/weather",
                     params={"q": location, "appid": OWM_KEY, "units": "metric"}, timeout=15)
    data = r.json()
    if r.status_code != 200:
        raise RuntimeError(f"OpenWeather error {r.status_code}: {data.get('message')}")
    return data["main"]["temp"]

weather_decl = {
    "name": "get_current_temperature",
    "description": "Gets the current temperature for a given location.",
    "parameters": {"type": "object",
                   "properties": {"location": {"type": "string", "description": "City name, e.g. San Francisco"}},
                   "required": ["location"]},
}
config = types.GenerateContentConfig(tools=[types.Tool(function_declarations=[weather_decl])])

resp = client.models.generate_content(model=MODEL, contents="What's the temperature in Gurugram, Haryana?", config=config)
part = resp.candidates[0].content.parts[0]
print("Model asked for:", part.function_call)
if part.function_call:
    args = dict(part.function_call.args)
    print("Result:", get_current_temperature(**args), "°C")'''),
          md("## 2 · The full loop: send the result back so the model can answer in words"),
          code('''\
contents = [types.Content(role="user", parts=[types.Part(text="Is it warm in Delhi right now? Give the temperature.")])]
resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
call = resp.candidates[0].content.parts[0].function_call
if call:
    contents.append(resp.candidates[0].content)                       # the model's tool request
    try:
        result = {"result": get_current_temperature(**dict(call.args))}
    except Exception as e:
        result = {"error": str(e)}                                    # tell the model the tool failed
    contents.append(types.Content(role="user", parts=[types.Part.from_function_response(name=call.name, response=result)]))
    final = client.models.generate_content(model=MODEL, contents=contents, config=config)
    print(final.text)
else:
    print(resp.text)'''),
          md("## 3 · Multiple tools, one question (manual loop)"),
          code('''\
def add(a: int, b: int) -> int:
    print(f"  executing add({a}, {b})"); return a + b
def multiply(a: int, b: int) -> int:
    print(f"  executing multiply({a}, {b})"); return a * b
available = {"add": add, "multiply": multiply}

def decl(name, desc):
    return types.FunctionDeclaration(name=name, description=desc, parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={"a": types.Schema(type=types.Type.INTEGER, description="First integer"),
                    "b": types.Schema(type=types.Type.INTEGER, description="Second integer")},
        required=["a", "b"]))

cfg = types.GenerateContentConfig(tools=[types.Tool(function_declarations=[decl("add", "Adds two integers."), decl("multiply", "Multiplies two integers.")])],
                                  automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
msgs = [types.Content(role="user", parts=[types.Part(text="What is 3 * 12? Also, what is 11 + 49?")])]
r1 = client.models.generate_content(model=MODEL, contents=msgs, config=cfg)
msgs.append(r1.candidates[0].content)
outs = []
for p in r1.candidates[0].content.parts:
    if p.function_call:
        fn = available[p.function_call.name]
        outs.append(types.Part.from_function_response(name=p.function_call.name, response={"result": fn(**dict(p.function_call.args))}))
msgs.append(types.Content(role="user", parts=outs))
print(client.models.generate_content(model=MODEL, contents=msgs, config=cfg).text)'''),
          md("## 4 · Automatic function calling\nPass plain Python functions (with type hints and a docstring) and the SDK runs the whole loop for you."),
          code('''\
r = client.models.generate_content(model=MODEL, contents="What is 7 times 6, plus 100?",
                                   config=types.GenerateContentConfig(tools=[add, multiply]))
print(r.text)
print("calls made:", [h.parts[0].function_call.name for h in (r.automatic_function_calling_history or []) if h.parts and h.parts[0].function_call])'''),
          md("## 5 · Tool modes: AUTO · ANY · NONE\n`AUTO` lets the model decide, `ANY` forces a tool call, `NONE` forbids it."),
          code('''\
for mode in ["AUTO", "ANY", "NONE"]:
    cfg = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=[decl("add", "Adds two integers.")])],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode)),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    r = client.models.generate_content(model=MODEL, contents="Say hello to me.", config=cfg)
    p = r.candidates[0].content.parts[0]
    print(mode.ljust(5), "->", "FUNCTION CALL " + p.function_call.name if p.function_call else "text: " + (p.text or "")[:60])'''),
          md("## 6 · The same tools through LangChain"),
          code('''\
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

@tool
def lc_add(a: int, b: int) -> int:
    """Add two integers"""
    return a + b

@tool
def lc_multiply(a: int, b: int) -> int:
    """Multiply two integers"""
    return a * b

llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=os.environ["GEMINI_API_KEY"]).bind_tools([lc_add, lc_multiply])
reg = {"lc_add": lc_add, "lc_multiply": lc_multiply}
ai = llm.invoke("What is 12 plus 8, and what is 7 times 6?")
for tc in ai.tool_calls:
    print(tc["name"], tc["args"], "->", reg[tc["name"]].invoke(tc["args"]))''')]
    save("M8_function_calling.ipynb", c)


# ============================================================================ M9
def nb_m9():
    from core.course_prompts import EVALUATOR_SYSTEM_PROMPT, STUDY_BUDDY_SYSTEM_PROMPT
    c = header("Module 9 · Evaluating GenAI outputs (LLM-as-judge)",
               "A *Target* model (Study Buddy) answers; a **separate** evaluator model scores the answer against a rubric. A fresh call keeps the judge unbiased.",
               [GEMINI], "google-genai")
    c += [code(CLIENT),
          md("## 1 · The two prompts (edit them!)"),
          code('STUDY_BUDDY_SYSTEM_PROMPT = """\n' + STUDY_BUDDY_SYSTEM_PROMPT.strip().replace('"""', "'''") + '\n"""'),
          code('EVALUATOR_SYSTEM_PROMPT = """\n' + EVALUATOR_SYSTEM_PROMPT.strip().replace('"""', "'''") + '\n"""'),
          md("## 2 · Run the target model"),
          code('''\
USER_QUERY = "What is photosynthesis?"      # try an off-topic one too, e.g. "Write my homework essay about football"

target = client.models.generate_content(
    model=MODEL, contents=USER_QUERY,
    config=types.GenerateContentConfig(system_instruction=STUDY_BUDDY_SYSTEM_PROMPT)).text
print(target)'''),
          md("## 3 · Judge it with a fresh call"),
          code('''\
import json, re
judge_input = f"""Original Prompt (User Query):
{USER_QUERY}

Target Model's Output:
{target}
"""
raw = client.models.generate_content(
    model=MODEL, contents=judge_input,
    config=types.GenerateContentConfig(system_instruction=EVALUATOR_SYSTEM_PROMPT)).text

def safe_parse_json(text):
    m = re.search(r"\\{.*\\}", text, re.S)
    try:
        return json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        return {}

scores = safe_parse_json(raw)
print(json.dumps(scores, indent=2) if scores else raw)'''),
          md("## 4 · Try a batch of queries and compare\nBecause the judge is a separate call you can score many answers and look for patterns."),
          code('''\
queries = ["What is an atom?", "I don't understand fractions.", "Write my history essay for me.", "Who will win the next election?"]
rows = []
for q in queries:
    ans = client.models.generate_content(model=MODEL, contents=q, config=types.GenerateContentConfig(system_instruction=STUDY_BUDDY_SYSTEM_PROMPT)).text
    verdict = client.models.generate_content(model=MODEL, contents=f"Original Prompt (User Query):\\n{q}\\n\\nTarget Model's Output:\\n{ans}\\n",
                                             config=types.GenerateContentConfig(system_instruction=EVALUATOR_SYSTEM_PROMPT)).text
    rows.append({"query": q, **{k: v for k, v in safe_parse_json(verdict).items() if isinstance(v, (int, float))}})
import pandas as pd
pd.DataFrame(rows)''')]
    save("M9_llm_as_judge.ipynb", c)


# ============================================================================ project notebooks
def writefile_cell(rel_path, text):
    return code(f"%%writefile {rel_path}\n{text.rstrip()}\n")


def project_files(proj, files):
    base = DL / "projects" / proj
    cells = []
    dirs = sorted({str(Path(f).parent) for f in files if str(Path(f).parent) != "."})
    if dirs:
        cells.append(code("import os\nfor d in " + repr(dirs) + ":\n    os.makedirs(d, exist_ok=True)"))
    for f in files:
        cells.append(writefile_cell(f, (base / f).read_text("utf-8")))
    return cells


def nb_scamguard():
    base = DL / "projects/scamguard_ai"
    c = header("Project · ScamGuard AI — scam-message detector",
               "A 3-step pipeline (**build prompt → call LLM → parse JSON**) plus an evaluation script that measures accuracy, recall and F1 on a labelled dataset.",
               [GEMINI], "google-genai pandas pydantic python-dotenv tqdm")
    c += [md("## 1 · Create the project files in this Colab session\nEach cell below writes one file of the original project (`%%writefile`). Read them as you go — it's the same code as in the app's project page."),
          code("%cd /content\n!mkdir -p scamguard_ai\n%cd scamguard_ai")]
    files = ["utils.py", "config.py", "llm/__init__.py", "llm/client.py", "llm/parser.py", "llm/validator.py", "llm/prompts.py",
             "llm/prompts/react.md", "llm/prompts/fewshot.md", "llm/prompts/strict_json.txt", "llm/prompts/simplified.txt",
             "llm/prompts/few_shot_examples.txt", "pipeline/__init__.py", "pipeline/scam_detector/__init__.py",
             "pipeline/scam_detector/builder.py", "pipeline/scam_detector/executor.py", "pipeline/scam_detector/parser.py",
             "pipeline/scam_detector/detector.py", "evaluate.py"]
    c += project_files("scamguard_ai", files)
    # dataset: 8-row test set + a stratified 60-row sample of the training set
    rows = list(csv.DictReader(io.StringIO((base / "scam_detection_dataset.csv").read_text("utf-8"))))
    by = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    sample = []
    for lab, rs in by.items():
        step = max(1, len(rs) // 30)
        sample += rs[::step][:30]
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader(); w.writerows(sample)
    c += [md(f"## 2 · Data\nA {len(sample)}-message sample of the course dataset is embedded so this runs anywhere. "
             "To use the **full 1,000-message file**, run the upload cell afterwards."),
          writefile_cell("scam_sample.csv", out.getvalue()),
          writefile_cell("test_scam_dataset.csv", (base / "test_scam_dataset.csv").read_text("utf-8")),
          code("# OPTIONAL: upload scam_detection_dataset.csv (downloadable from the app) to evaluate on all rows\n"
               "# from google.colab import files; files.upload()"),
          md("## 3 · Key"),
          code(SETUP_KEYS + '\nget_secret("GEMINI_API_KEY", hint="(from aistudio.google.com/apikey) ")\n'
               'os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"   # or "gemini-3.5-flash-lite" if your key supports it'),
          md("## 4 · Detect a single message"),
          code('''\
import sys; sys.path.insert(0, "/content/scamguard_ai")
from pipeline.scam_detector.detector import ScamDetector
import json

detector = ScamDetector()          # strategy = "react" (see llm/prompts/react.md)
for msg in ["Congratulations! You've won a free iPhone. Click this link and enter your bank details to claim.",
            "Dear customer, your electricity bill of Rs 840 is due on the 5th. Pay on the official portal.",
            "Your package from Amazon is delayed. Track it here: bit.ly/track123"]:
    print(msg); print(json.dumps(detector.detect(msg), indent=2), "\\n")'''),
          md("## 5 · See exactly which prompt is sent\nChange the strategy by editing `PROMPT = load_prompt(...)` in `llm/prompts.py`; here we just print what the pipeline builds."),
          code('''\
from pipeline.scam_detector.builder import build_prompt
print(build_prompt("URGENT: Your account will be suspended! Call 555-FAKE immediately.", "react"))'''),
          md("## 6 · Evaluate on the sample (accuracy · recall · F1)\n`--limit` keeps API calls low while you're on a free quota; remove it to use everything."),
          code("!python evaluate.py scam_sample.csv --limit 20 --batch-size 5"),
          md("### Where to go next\n* Swap the prompt strategy (`fewshot.md`, `strict_json.txt`, `simplified.txt`) and compare F1.\n* Validate every result with `llm/validator.py` (Pydantic).\n* Add a retry when the parser returns “Uncertain”.")]
    save("P1_ScamGuard_AI.ipynb", c)


def nb_specsense():
    c = header("Project · SpecSense AI — explainable product recommendations",
               "Free-text shopping need → hard specs + soft preferences → explainable recommendation. First the **original** ReAct pipeline (answers from model memory), "
               "then the **catalog-grounded** version (extract → match with pandas → explain) with gap detection.",
               [GEMINI], "google-genai pandas pydantic python-dotenv")
    files = ["utils.py", "config.py", "llm/__init__.py", "llm/client.py", "llm/prompts.py", "llm/prompts/react.md", "pipeline/__init__.py",
             "pipeline/spec_sensor/__init__.py", "pipeline/spec_sensor/builder.py", "pipeline/spec_sensor/executor.py",
             "pipeline/spec_sensor/parser.py", "pipeline/spec_sensor/sensor.py", "catalog_matcher.py", "data/products_sample.csv"]
    c += [md("## 1 · Create the project files in this Colab session\nEach cell writes one file (`%%writefile`). `catalog_matcher.py` and the sample catalog are the grounded add-on."),
          code("%cd /content\n!mkdir -p specsense_ai\n%cd specsense_ai")]
    c += project_files("specsense_ai", files)
    c += [md("## 2 · Key"),
          code(SETUP_KEYS + '\nget_secret("GEMINI_API_KEY", hint="(from aistudio.google.com/apikey) ")\nos.environ["GEMINI_MODEL"] = "gemini-2.5-flash"   # or "gemini-3.5-flash-lite" if your key supports it'),
          md("## 3 · Original pipeline (recommends from the model's memory)\nLook at `review_snippets` and the product name — then compare with the catalog in section 5."),
          code('''\
import sys, json; sys.path.insert(0, "/content/specsense_ai")
from pipeline.spec_sensor.sensor import SpecSensor
sensor = SpecSensor()
q = "Find an affordable, lightweight laptop under rupees 30k suitable for coding and programming students."
orig = sensor.detect(q)
print(json.dumps(orig, indent=2, ensure_ascii=False))'''),
          md("## 4 · The catalog (12 fictional laptops)"),
          code("import pandas as pd\ncat = pd.read_csv('data/products_sample.csv')\ncat.drop(columns=['description'])"),
          code('''\
name = orig.get("recommended_product", "")
print("Recommended by the original pipeline:", name)
print("Exists in our catalog?", any(name.lower() == n.lower() for n in cat["name"]))'''),
          md("## 5 · Grounded version: extract → match → explain\nThe matching step is plain pandas, so it is exact and testable; the LLM only extracts needs and explains the retrieved rows."),
          code('''\
from catalog_matcher import recommend
res = recommend("Lightweight laptop under 30000 for coding students, good battery backup")
print("Constraints :", res.pop("_constraints")); print("Exact match :", res.pop("_exact")); print("Candidates  :", res.pop("_candidates"), "\\n")
print(json.dumps(res, indent=2, ensure_ascii=False))'''),
          md("## 6 · Gap detection\nAsk for something the catalog cannot satisfy: under ₹30,000 **and** 16 GB RAM."),
          code('''\
import catalog_matcher as cm
rows, exact = cm.match(cm.load_catalog(), {"max_price_inr": 30000, "min_ram_gb": 16})
print("Exact match?", exact)
rows[["name", "price_inr", "ram_gb", "violations"]]'''),
          md("## 7 · Automatic checks (no LLM)\nIs the product in the catalog? Are the hard constraints met? Is the review quote verbatim?"),
          code('''\
def check(rec, limits):
    row = cat[cat.name == rec.get("recommended_product")]
    exists = len(row) == 1
    ok = exists and row.price_inr.iloc[0] <= limits["price_inr"] and row.weight_kg.iloc[0] <= limits["weight_kg"]
    quote = exists and all(sn.lower() in row.review.iloc[0].lower() for sn in rec.get("review_snippets", []))
    return {"in catalog": exists, "hard constraints met": ok, "review verbatim": quote}
lim = {"price_inr": 30000, "weight_kg": 1.5}
print("original :", check(orig, lim)); print("grounded :", check(res, lim))'''),
          md("### Try next\n* Replace `data/products_sample.csv` with your own catalog (same columns).\n* Add a `min_screen_inches` constraint in `HARD` and the extract prompt.\n* Ask the LLM judge (see the app's SpecSense page) to score explainability.")]
    save("P3_SpecSense_AI.ipynb", c)


def nb_advisor():
    c = header("Project · Gemini Cinema Advisor — structured output + OMDb + LLM-as-judge",
               "LangChain + Gemini return a **Pydantic-validated** recommendation; OMDb adds posters and ratings; a *separate* judge call scores the result. "
               "Here we run it without Streamlit.",
               [GEMINI, OMDB], "langchain langchain-google-genai pydantic requests python-dotenv")
    files = ["tracing.py", "models.py", "omdb_client.py", "recommender.py", "evaluator.py"]
    c += [code("%cd /content\n!mkdir -p movie_advisor\n%cd movie_advisor")]
    c += project_files("movie_advisor", files)
    c += [md("## Keys"),
          code(SETUP_KEYS + '\nGEMINI = get_secret("GEMINI_API_KEY", hint="(aistudio.google.com/apikey) ")\nOMDB = get_secret("OMDB_API_KEY", hint="(omdbapi.com/apikey.aspx) ")'),
          md("## Ask for recommendations\nHistory is a list of `(role, text)` tuples. Words like *new*, *latest*, *2026* trigger the extra Google-Search research call."),
          code('''\
import sys; sys.path.insert(0, "/content/movie_advisor")
from recommender import get_recommendation
from omdb_client import OmdbClient
from evaluator import get_evaluation

MODEL_NAME = "gemini-3.5-flash-lite"      # or "gemini-2.5-flash"
history = []
user_input = "I love slow-burn sci-fi like Arrival and hate jump-scare horror. Suggest 3 films."
result, research = get_recommendation(GEMINI, history, user_input, MODEL_NAME)
print(result.reply, "\\n")
omdb = OmdbClient(OMDB)
for m in result.movies:
    info = omdb.get_movie(m.title, m.year)
    print(f"* {m.title} ({m.year}) - IMDb {info.imdb_rating} | {info.runtime} | {info.genre}\\n    why: {m.reason}")
if result.follow_up_question:
    print("\\nFollow-up:", result.follow_up_question)'''),
          md("## Judge the answer (separate call)"),
          code('''\
evaluation, err = get_evaluation(GEMINI, user_input, result.movies, MODEL_NAME)
print(err or "")
if evaluation:
    print("Overall:", evaluation.overall_score, "/10 -", evaluation.overall_feedback)
    for e in evaluation.movie_evaluations:
        print(" -", e.title, e.relevance_score, "/10:", e.justification)'''),
          md("## Why this design?\n* `with_structured_output(Model)` = no regex parsing; Pydantic **validates**.\n* Structured output and Google Search **can't share one call**, so research and formatting are two calls.\n* The judge is a *fresh* call so it can't grade its own homework.\n* `tracing.py` logs each step and its timing — handy for debugging.")]
    save("P2_Cinema_Advisor_LangChain_OMDb.ipynb", c)


if __name__ == "__main__":
    for fn in (nb_m6, nb_m7, nb_m7_chatbot, nb_m8, nb_m9, nb_scamguard, nb_advisor, nb_specsense):
        fn()
