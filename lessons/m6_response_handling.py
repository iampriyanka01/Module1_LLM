import json
import re
import time

import pandas as pd
import streamlit as st
from pydantic import BaseModel, ValidationError, field_validator

from core.llm import generate, generate_many, llm_ready, run_live, stream
from core.ui import analogy, cards, compare, definition, page_header, quiz, takeaway
from core.diagrams import show_dg

page_header(
    "m6",
    "From prompt to product: LangChain, streaming, batching, structured output with Pydantic, graceful failure and guardrails.",
    [
        "Explain why frameworks like LangChain exist",
        "Stream responses token by token (sync vs async)",
        "Process many inputs at once with batching",
        "Validate LLM output against a schema with Pydantic / PydanticOutputParser",
        "Handle validation failures: try-except, logging, retry, fallback, cleanup",
        "Describe guardrails: Checker, Corrector, Rail and Guard",
    ],
)

tabs = st.tabs(["Why LangChain?", "Capabilities", "⚡ Streaming", "📦 Batch input", "🧱 Pydantic validation",
                "🩹 Handling failures", "🚧 Guardrails", "🧾 Key takeaways", "✅ Knowledge check"])

# ---------------------------------------------------------------------------
with tabs[0]:
    analogy("You don't build a large website from raw HTML files — you use Flask or Django. Likewise you don't build "
            "production GenAI apps from raw prompts and function calls — you use a framework like LangChain.",
            title="Web frameworks")
    show_dg("m6:0")
    st.markdown("Scenario: an **AI HR assistant** that reads hundreds of résumés, matches roles, remembers candidates and "
                "answers policy questions from PDFs. Plain Python quickly gets messy:")
    probs = [
        ("1 · Chaining steps", "extract job title → summarise → recommend role", "`LLMChain`, `SequentialChain`, LCEL pipes: modular, reusable steps"),
        ("2 · Memory", "“Who is Elon Musk?” → “Where was he born?” needs history stitched manually",
         "`ConversationBufferMemory`, summary memory: history + token limits handled"),
        ("3 · Tools", "“Tax on ₹10 lakh under the new regime?” needs current slabs + exact maths",
         "Search, calculator and custom Python tools the model can choose"),
        ("4 · Company documents", "“What's our refund policy?” lives in a PDF", "Document loaders + retrieval: answer from *your* data"),
        ("5 · Scale", "100 tickets at once, live typing, many users", "`.batch()/.abatch()`, `.stream()/.astream()`, async chains"),
    ]
    st.dataframe(pd.DataFrame(probs, columns=["Problem", "In plain Python…", "LangChain gives you"]), hide_index=True, width="stretch")

with tabs[1]:
    st.subheader("10 capabilities, one HR assistant")
    show_dg("m6:1")
    caps = [
        ("LLM integration", "Plug in Gemini, GPT-4, Claude as the brain", "“Summarize this résumé's key qualifications.”"),
        ("Prompt templates", "Reusable prompts with {placeholders}", "“Summarize the résumé of {candidate_name} for a {job_title} role.”"),
        ("Chains", "Fixed multi-step pipelines", "extract title → summarise skills → recommend role"),
        ("Memory", "Remember earlier turns", "“I've worked here 5 years.” … “What promotion am I eligible for?”"),
        ("Tools", "Calculator, search, APIs", "“New salary after 20% hike on ₹6,00,000?” → ₹7,20,000"),
        ("Agents", "AI decides which tool to use", "Search HR PDF for total leaves → subtract 12 → answer"),
        ("Retrieval", "Answer from your documents", "“Maternity leave policy?” → “26 weeks paid”"),
        ("Streaming", "Token-by-token output", "Long tax explanation appears as it's typed"),
        ("Batch", "Many inputs at once", "500 résumés → 500 structured summaries"),
        ("Deployment", "LangServe exposes chains as APIs", "Polite rejection email endpoint for the ATS"),
    ]
    st.dataframe(pd.DataFrame(caps, columns=["Capability", "What it does", "HR example"]), hide_index=True, width="stretch")
    st.markdown("#### “Hello World” with LangChain + Gemini")
    st.code("""# pip install langchain-google-genai langchain-core python-dotenv
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
# At most 4 model calls a minute (the free tier allows 5), shared by every agent in this notebook
limiter = InMemoryRateLimiter(requests_per_second=4 / 60, check_every_n_seconds=0.5, max_bucket_size=1)
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"),
                             rate_limiter=limiter)
response = llm.invoke("Explain quantum computing in simple terms")   # .invoke() = run once, one input → one output
print(response.content)""", language="python")

# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Streaming — show the answer while it's being written")
    show_dg("m6:2")
    analogy("Watching a friend's message appear in WhatsApp as they type feels fast and alive. Waiting 20 seconds for a "
            "wall of text feels broken.", title="WhatsApp typing…")
    compare("Synchronous (`stream()`)", "You order tea, **wait**, then order a snack. Simple, but the program is **blocked** "
            "until streaming finishes — no other work, not great for many users.",
            "Asynchronous (`astream()`)", "Order tea and snacks together; while tea brews you reply to emails. **Non-blocking** — "
            "stream to one user while logging, showing “typing…”, checking order status, serving others.", bad_good=False)
    st.code("""# LangChain                                   # Google GenAI SDK (used in this app)
for chunk in llm.stream("Explain tax regimes"):   for chunk in client.models.generate_content_stream(
    print(chunk.content, end="")                       model=MODEL, contents=prompt):
                                                       print(chunk.text, end="")
async for chunk in llm.astream(prompt):  # async
    print(chunk.content, end="")""", language="python")
    st.markdown("#### 🟢 Feel the difference")
    sp = st.text_input("Prompt", "Explain how income tax slabs work, in about 200 words, for a first-time salaried employee.", key="m6_stream_p")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Without streaming** (wait for everything)")
        if st.button("Run blocking call", disabled=not llm_ready(False), key="m6_block"):
            with st.spinner("Waiting for the full response…"):
                try:
                    txt, _, secs = generate(sp)
                    st.metric("Time until you see anything", f"{secs:.2f}s")
                    st.markdown(txt)
                except Exception as e:  # noqa: BLE001
                    st.error(str(e)[:300])
    with c2:
        st.markdown("**With streaming** (see the first words early)")
        if st.button("Run streaming call", disabled=not llm_ready(False), key="m6_stream"):
            t0 = time.perf_counter()
            first = {}

            def gen():
                for chunk in stream(sp):
                    first.setdefault("t", time.perf_counter() - t0)
                    yield chunk

            try:
                st.write_stream(gen())
                st.metric("Time to first words", f"{first.get('t', 0):.2f}s", f"total {time.perf_counter() - t0:.2f}s", delta_color="off")
            except Exception as e:  # noqa: BLE001
                st.error(str(e)[:300])
    llm_ready()

# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Batch input — don't hit “Print” 100 times")
    show_dg("m6:3")
    st.markdown("Send many inputs together instead of looping one by one: **faster**, **fewer round-trips**, "
                "**scalable**, **cleaner code**. `.batch()` blocks until all are done; `.abatch()` is the async, non-blocking version.")
    compare("Manual loop", "```python\nresults = []\nfor r in resumes:\n    results.append(llm.invoke(r))\n```",
            "LangChain batch", "```python\nresults = llm.batch(resumes)          # sync\nresults = await llm.abatch(resumes)   # async\n```")
    tickets = st.text_area("One support ticket per line",
                           "My refund hasn't arrived after 10 days.\nHow do I change my delivery address?\n"
                           "The app crashes when I open my orders page.\nLoved the fast delivery, thanks!\n"
                           "I was charged twice for the same order.", height=140, key="m6_batch")
    items = [t.strip() for t in tickets.splitlines() if t.strip()]
    tmpl = "Classify this support ticket into one of [Refund, Delivery, Bug, Billing, Praise] and suggest a one-line reply. Format: <category> | <reply>\nTicket: {t}"

    def _batch():
        t0 = time.perf_counter()
        out = generate_many([tmpl.format(t=t) for t in items], temperature=0)
        return out, time.perf_counter() - t0

    res = run_live(f"🟢 Process {len(items)} tickets in parallel", _batch, key="m6_batch_run")
    if res:
        out, wall = res
        seq = sum(o[2] for o in out)
        c1, c2, c3 = st.columns(3)
        c1.metric("Wall-clock (parallel)", f"{wall:.2f}s")
        c2.metric("Sum of individual calls (≈ sequential)", f"{seq:.2f}s")
        c3.metric("Speed-up", f"{seq / wall:.1f}×" if wall else "–")
        st.dataframe(pd.DataFrame({"ticket": items, "result": [o[0] or o[1] for o in out],
                                   "seconds": [round(o[2], 2) for o in out]}), hide_index=True, width="stretch")
        st.caption("Results come back in the same order as the inputs — just like `.batch()`.")

# ---------------------------------------------------------------------------
class Person(BaseModel):
    name: str
    age: int
    email: str

    @field_validator("email")
    @classmethod
    def valid_email(cls, v: str) -> str:
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}", v):
            raise ValueError("not a valid email address")
        return v


def extract_json(text: str) -> dict:
    """Pull the first {...} block out of chatty output / ```json fences."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


with tabs[4]:
    st.subheader("LLMs are helpful — but not always predictable")
    show_dg("m6:4")
    st.markdown("Ask for JSON with `name, age, email` and you may get missing fields, extra fields, renamed keys or "
                "`\"age\": \"twenty\"`. At 10,000 résumés, one bad record can **crash the app or corrupt your database**.")
    definition("Pydantic", "a Python library that validates and parses data using type hints. Define a model once; it "
                           "converts what it safely can (`\"25\"` → `25`) and raises `ValidationError` for the rest.")
    st.code("""from pydantic import BaseModel, EmailStr     # EmailStr needs: pip install "pydantic[email]"

class Person(BaseModel):
    name: str
    age: int
    email: EmailStr""", language="python")
    st.markdown("#### 🧪 Playground — be the LLM and try to break the schema")
    samples = {
        "✅ Perfect": '{"name": "Priya Sharma", "age": 20, "email": "priya.sharma@example.com"}',
        "🔁 Age as string (coerced)": '{"name": "Priya Sharma", "age": "20", "email": "priya.sharma@example.com"}',
        "❌ Age in words": '{"name": "Priya Sharma", "age": "twenty", "email": "priya.sharma@example.com"}',
        "❌ Missing email": '{"name": "Priya Sharma", "age": 20}',
        "❌ Bad email": '{"name": "Priya Sharma", "age": 20, "email": "priya at example dot com"}',
        "💬 Chatty model + fences": 'Sure! Here is the JSON:\n```json\n{"name": "Priya Sharma", "age": 20, "email": "priya.sharma@example.com"}\n```',
    }
    s = st.radio("Load a sample LLM output", list(samples), horizontal=True, key="m6_sample")
    raw = st.text_area("Raw LLM output", samples[s], height=110, key=f"m6_raw_{s}")
    try:
        data = extract_json(raw)
        person = Person(**data)
        st.success("Validated ✔")
        st.code(repr(person), language="python")
    except ValidationError as e:
        st.error("ValidationError")
        st.dataframe(pd.DataFrame([{"field": ".".join(map(str, er["loc"])), "problem": er["msg"], "got": str(er.get("input"))}
                                   for er in e.errors()]), hide_index=True, width="stretch")
    except (ValueError, json.JSONDecodeError) as e:
        st.error(f"Could not even parse JSON: {e}")

    st.markdown("#### 🟢 Live: PydanticOutputParser-style extraction")
    st.caption("The parser's `get_format_instructions()` adds the JSON schema to the prompt; `.parse()` validates the reply.")
    text_in = st.text_input("Unstructured text", "Hi, my name is Priya Sharma. I'm twenty and my email is priya.sharma@example.com.", key="m6_ext")
    fmt = json.dumps(Person.model_json_schema(), indent=None)
    prompt = f"Extract person info from the text below.\n{text_in}\n\nReturn ONLY a JSON object that matches this JSON schema:\n{fmt}"
    with st.expander("Prompt sent (includes format instructions)"):
        st.code(prompt, language="text")

    def _extract():
        txt = generate(prompt, temperature=0)[0]
        try:
            return txt, Person(**extract_json(txt)), None
        except Exception as e:  # noqa: BLE001
            return txt, None, str(e)

    res = run_live("Extract & validate", _extract, key="m6_extract")
    if res:
        txt, obj, err = res
        st.code(txt, language="json")
        if obj:
            st.success(f"Structured output: `{obj!r}`")
        else:
            st.error(f"Validation failed: {err}")
    st.code("""from langchain.output_parsers import PydanticOutputParser   # (langchain_core.output_parsers in newer versions)
parser = PydanticOutputParser(pydantic_object=Person)
prompt = f"Extract person info:\\n{text}\\n{parser.get_format_instructions()}"
response = llm.invoke(prompt)
try:
    person = parser.parse(response.content)
except Exception as e:
    print("Validation failed:", e)""", language="python")

# ---------------------------------------------------------------------------
WORDS = {"eighteen": 18, "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty one": 21, "twenty-five": 25,
         "twenty five": 25, "thirty": 30, "forty": 40}


def cleanup(data: dict) -> tuple[dict, list[str]]:
    """Technique 5 - post-processing: fix recurring, predictable issues."""
    fixes = []
    d = dict(data)
    for alt in ("full_name", "Name", "person_name"):
        if alt in d and "name" not in d:
            d["name"] = d.pop(alt)
            fixes.append(f"renamed '{alt}' → 'name'")
    if "e-mail" in d and "email" not in d:
        d["email"] = d.pop("e-mail")
        fixes.append("renamed 'e-mail' → 'email'")
    age = d.get("age")
    if isinstance(age, str):
        m = re.match(r"\s*(\d+)", age)
        if m:
            d["age"] = int(m.group(1))
            fixes.append(f"age '{age}' → {d['age']}")
        elif age.strip().lower() in WORDS:
            d["age"] = WORDS[age.strip().lower()]
            fixes.append(f"age '{age}' → {d['age']}")
    if isinstance(d.get("email"), str):
        e = d["email"].strip().lower().replace(" at ", "@").replace(" dot ", ".")
        if e != d["email"]:
            fixes.append(f"email normalised → {e}")
            d["email"] = e
    return d, fixes


with tabs[5]:
    st.subheader("At scale, perfection isn't the goal — robustness is")
    show_dg("m6:5")
    techniques = [
        ("1 · Try-except", "First line of defence: catch `ValidationError` so one bad output doesn't crash the app.", "Like a web form showing an error instead of crashing the site."),
        ("2 · Log invalid output", "Store the raw output that failed — your goldmine for debugging and prompt fixes.", "CCTV: rewind and see what happened."),
        ("3 · Retry with a reformatted prompt", "Ask the LLM to fix its own output: “Return only valid JSON matching this schema.”", "“Could you resend that as a PDF instead of a screenshot?”"),
        ("4 · Fallback to a safe default", "If retries fail, return a default record so the pipeline keeps running.", "A complimentary drink while your meal is delayed."),
        ("5 · Post-processing & cleanup", "Fix predictable issues in code (\"25 years\" → 25, string → list) before validating.", "Extracting the 10-digit number from “+91-98765-43210 ext.234”."),
    ]
    cards([(t, f"{d}\n\n*{a}*") for t, d, a in techniques], cols=5)
    st.markdown("#### 🧪 Run a messy output through the full safety pipeline")
    messy = {
        "Chatty + '25 years'": 'Of course! {"full_name": "Arjun Rao", "age": "25 years", "email": "arjun@example.com"}',
        "Age in words, e-mail key": '{"name": "Meera", "age": "twenty", "e-mail": "meera at example dot com"}',
        "Garbage (no JSON)": "I'm sorry, I couldn't find any details in that text.",
        "Already perfect": '{"name": "Kabir", "age": 31, "email": "kabir@example.org"}',
    }
    pick = st.selectbox("LLM output", list(messy), key="m6_messy")
    raw = st.text_area("Raw output", messy[pick], key=f"m6_messy_raw_{pick}")
    use_live_retry = st.toggle("Use Gemini for the retry step (otherwise simulated)", value=False, disabled=not llm_ready(False))
    if st.button("Run pipeline", type="primary"):
        log = []
        result = None
        try:
            data = extract_json(raw)
            log.append(("parse", "✅", "JSON object found"))
            data, fixes = cleanup(data)
            log.append(("5 · cleanup", "🧹", "; ".join(fixes) or "nothing to fix"))
            result = Person(**data)
            log.append(("1 · validate (try)", "✅", "passed"))
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            log.append(("1 · validate (except)", "❌", str(e).splitlines()[0][:120]))
            log.append(("2 · log", "📝", f"stored raw output ({len(raw)} chars) for review"))
            if use_live_retry:
                try:
                    fixed = generate("Your previous output was not valid. Re-emit ONLY a JSON object with keys name (string), age (integer), "
                                     f"email (valid address). If a value is unknown use null.\nPrevious output:\n{raw}", temperature=0)[0]
                    result = Person(**cleanup(extract_json(fixed))[0])
                    log.append(("3 · retry (Gemini)", "✅", f"fixed: {fixed[:80]}"))
                except Exception as e2:  # noqa: BLE001
                    log.append(("3 · retry (Gemini)", "❌", str(e2).splitlines()[0][:120]))
            else:
                log.append(("3 · retry (simulated)", "❌", "model returned the same kind of output"))
            if result is None:
                result = {"name": "UNKNOWN", "age": None, "email": None, "needs_review": True}
                log.append(("4 · fallback", "🛟", "returned safe default record; pipeline continues"))
        st.dataframe(pd.DataFrame(log, columns=["step", "status", "detail"]), hide_index=True, width="stretch")
        st.markdown("**Final record passed downstream:**")
        st.code(repr(result), language="python")

# ---------------------------------------------------------------------------
PII = [(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", "email address"),
       (r"(?:\+91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b", "[PHONE]", "phone number"),
       (r"\b[A-Z]{5}\d{4}[A-Z]\b", "[PAN]", "PAN number")]
TOXIC = {"stupid": "questionable", "idiot": "person", "useless": "unhelpful", "pathetic": "disappointing"}
BRAND = {"guys": "team", "ASAP": "at the earliest", "gonna": "going to"}


def checker(text: str) -> list[tuple[str, str]]:
    found = []
    for pat, _, label in PII:
        for m in re.findall(pat, text):
            found.append(("PII", f"{label}: {m}"))
    for w in TOXIC:
        if re.search(rf"\b{w}\b", text, re.I):
            found.append(("Toxicity", w))
    for w in BRAND:
        if re.search(rf"\b{re.escape(w)}\b", text):
            found.append(("Brand style", w))
    return found


def corrector(text: str, category: str) -> str:
    """Fixes ONE category per pass so the Rail's loop is visible."""
    if category == "PII":
        for pat, rep, _ in PII:
            text = re.sub(pat, rep, text)
    elif category == "Toxicity":
        for w, rep in TOXIC.items():
            text = re.sub(rf"\b{w}\b", rep, text, flags=re.I)
    elif category == "Brand style":
        for w, rep in BRAND.items():
            text = re.sub(rf"\b{re.escape(w)}\b", rep, text)
    return text


with tabs[6]:
    st.subheader("Guardrails — quality control for an unpredictable engine")
    show_dg("m6:6")
    st.markdown("**Why?** LLMs are probabilistic, context-sensitive, vulnerable to prompt injection, can hallucinate, can be "
                "biased/unsafe, and increasingly make *decisions* (loans, résumés, legal answers) — like an intern making CEO decisions.")
    st.dataframe(pd.DataFrame([
        ("🔎 Checker", "The watchdog", "Detects violations — toxicity, bias, PII, bad format. Flags, doesn't change."),
        ("🔧 Corrector", "The fixer", "Repairs flagged content: soften tone, mask PII, fix structure."),
        ("🔁 Rail", "The conductor", "Runs check → correct → re-check, limits retries, logs each pass."),
        ("🛡️ Guard", "The orchestrator / editor-in-chief", "Coordinates everything, applies org rules, approves the final output, keeps the audit trail."),
    ], columns=["Component", "Nickname", "Role"]), hide_index=True, width="stretch")
    st.markdown("#### 🧪 Enterprise email assistant — run the guardrail pipeline")
    draft = st.text_area("Raw LLM draft",
                         "Hi guys, the CEO is stupid for delaying the launch. Call me at 98765 43210 or mail "
                         "rahul.k@acme.com ASAP. My PAN is ABCDE1234F.", height=100, key="m6_guard")
    max_passes = st.slider("Rail: max correction passes", 1, 5, 3)
    if st.button("Run Guard", type="primary"):
        text, audit = draft, []
        for p in range(1, max_passes + 1):
            issues = checker(text)
            audit.append({"pass": p, "checker found": ", ".join(f"{c}: {d}" for c, d in issues) or "nothing ✅"})
            if not issues:
                break
            cat = issues[0][0]
            text = corrector(text, cat)
            audit[-1]["corrector fixed"] = cat
        remaining = checker(text)
        st.dataframe(pd.DataFrame(audit), hide_index=True, width="stretch")
        if remaining:
            st.error(f"Guard: still {len(remaining)} issue(s) after {max_passes} passes → **blocked**, escalated for human review.")
        else:
            st.success("Guard: all checks passed → **approved for sending**.")
        compare("Before", draft, "After", text)
        st.caption("The Rail fixes one category per pass here to make the loop visible; real systems (e.g. Guardrails AI, NeMo Guardrails) "
                   "combine rule-based and LLM-based checkers.")


# ---------------------------------------------------------------------------
with tabs[7]:
    takeaway([
        "LangChain packages prompts, chains, memory, tools, retrieval, streaming, batching and deployment.",
        "Streaming cuts perceived latency; async streaming keeps apps responsive for many users.",
        "Batching processes many inputs at once with cleaner code (`.batch()` / `.abatch()`).",
        "Validate every structured output with Pydantic; plan for failure with try/except, logging, retries, fallbacks and cleanup.",
        "Guardrails = Checker (detect) → Corrector (fix) → Rail (loop) → Guard (approve + audit).",
    ])


with tabs[8]:
    quiz("m6", [
        {"q": "Main UX benefit of streaming?", "options": ["Lower token cost", "Users see the answer start almost immediately",
         "Better accuracy", "No API key needed"], "answer": 1, "why": "Time-to-first-token drops; total time is similar."},
        {"q": "`.abatch()` differs from `.batch()` because it…", "options": ["Is cheaper", "Is non-blocking (async)", "Uses no tokens", "Validates JSON"],
         "answer": 1, "why": "Async lets your app keep working while the batch runs."},
        {"q": "Pydantic receives `\"age\": \"twenty\"` for an `int` field. What happens?", "options": ["Silently stores 'twenty'",
         "Converts to 20 automatically", "Raises a ValidationError", "Deletes the field"], "answer": 2,
         "why": "\"25\" can be coerced; words can't — handle it with cleanup or a retry."},
        {"q": "Which guardrail component only flags problems?", "options": ["Checker", "Corrector", "Rail", "Guard"], "answer": 0,
         "why": "The Checker is the watchdog; the Corrector fixes."},
    ])
