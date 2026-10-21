import json
import math
import time

import pandas as pd
import requests
import streamlit as st

from core.ui import analogy, compare, definition, page_header, quiz, stepper, takeaway

page_header(
    "m51",
    "Pre-class prep: what really happens between typing a URL and seeing a page — and how apps talk to each other.",
    [
        "Trace a web request: DNS → TCP handshake → packets → reassembly → connection close",
        "Explain client-server architecture and what an API is",
        "Compare serverful and serverless APIs",
        "Read and write JSON",
        "Use HTTP methods and interpret status codes",
    ],
)

tabs = st.tabs(["🌍 Journey of a request", "📦 Packets", "Client · Server · API", "Serverful vs serverless",
                "🧾 JSON", "🧪 API playground", "Status codes", "🧾 Key takeaways", "✅ Knowledge check"])

with tabs[0]:
    st.subheader("You type codingninjas.com and hit Enter…")
    stepper("net_journey", [
        ("DNS resolution — find the address",
         "The internet uses **IP addresses** (e.g. `142.250.190.36`), not names. DNS is the **phonebook of the internet**.\n\n"
         "1. Browser checks its **local DNS cache**\n2. Asks the **recursive resolver** (your ISP)\n"
         "3. Resolver asks **root server → TLD server (.com) → authoritative server**\n4. IP returned — now the browser knows where to go.\n\n"
         "*Like finding Pizza Hut's number: check contacts (cache), then ask directories until you get it.*"),
        ("TCP three-way handshake — say hello",
         "TCP guarantees reliable, ordered delivery. Before any data:\n\n"
         "1. **SYN** — client: “I want to talk, here's my sequence number.”\n"
         "2. **SYN-ACK** — server: “Sure, here's mine; I acknowledge yours.”\n"
         "3. **ACK** — client: “Ready!”\n\n*Like a phone call: dial → “Hello?” → “Hi, can we talk?”*"),
        ("Data transfer — split into packets",
         "Data (HTML, images, video) travels as small **packets** (~1500 bytes). Each has a **header** (source & destination IP, "
         "sequence number, protocol, error-check) and a **payload** (the actual data). Packets may take different routes. "
         "*(See the Packets tab.)*"),
        ("Reassembly — put the pages back in order",
         "The receiver orders packets by sequence number, **asks for retransmission** of missing/corrupt ones, then renders the page. "
         "*Like arranging numbered letter pages — and calling your friend if page 5 is missing.*"),
        ("Connection termination — FIN-ACK",
         "1. Client **FIN** “I'm done sending.” 2. Server **ACK** “Got it.” 3. Server **FIN** “I'm done too.” "
         "4. Client **ACK** “All done!”\n\nClean closure frees resources (unclosed connections linger in `TIME_WAIT`)."),
    ])
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box style="rounded,filled" fontname="Helvetica" fillcolor="#eef2ff"];
 a [label="Type URL"]; b [label="DNS → IP"]; c [label="TCP handshake\\nSYN · SYN-ACK · ACK"]; d [label="Packets"]; e [label="Reassemble & render"]; f [label="FIN · ACK · FIN · ACK"];
 a->b->c->d->e->f; }""")

with tabs[1]:
    st.subheader("Why break data into packets?")
    analogy("Sending a large book by a courier that only accepts small envelopes: tear out each page, number it, send "
            "separately. The receiver confirms each one (ACK).", title="Book by post")
    size_kb = st.slider("File size (KB)", 1, 5000, 500)
    mtu = st.select_slider("Packet size (bytes)", [576, 1000, 1500, 9000], value=1500)
    n = math.ceil(size_kb * 1024 / mtu)
    c1, c2, c3 = st.columns(3)
    c1.metric("Packets needed", f"{n:,}")
    c2.metric("Header overhead (≈40 B each)", f"{n * 40 / 1024:,.1f} KB")
    c3.metric("If 1% are lost", f"{math.ceil(n * 0.01):,} resent")
    st.caption("The notes' example: a 500 KB image ≈ 340 packets at 1500 bytes each.")
    st.code("""Packet #17
├── Header: src=192.168.1.5  dst=142.250.190.36  seq=17  proto=TCP  checksum=0x3fa2
└── Payload: "...<div class='hero'>Learn GenAI..." """, language="text")
    compare("IP address", "Your device's **delivery address** on the internet. Can change when you switch Wi-Fi.",
            "MAC address", "Your network card's **permanent fingerprint** (Media Access Control). Tied to hardware.", bad_good=False)

with tabs[2]:
    st.subheader("Client-server architecture")
    analogy("In a restaurant you (client) order, the kitchen (server) cooks, and the waiter (API) carries the request and "
            "brings back the food. You never enter the kitchen.", title="Restaurant")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box style="rounded,filled" fontname="Helvetica"];
 C [label="Client\\n(you / your app / browser)" fillcolor="#dbeafe"]; A [label="API\\n(the waiter: rules for asking)" fillcolor="#fef9c3"];
 S [label="Server\\n(kitchen: Gemini, ChatGPT…)" fillcolor="#dcfce7"];
 C->A [label=" request (menu order)"]; A->S [label=" structured request"]; S->A [label=" response"]; A->C [label=" food = JSON"]; }""")
    definition("API (Application Programming Interface)", "a set of rules — a communication bridge — that lets one piece of "
               "software talk to another. It doesn't do the cooking; it understands your order, delivers it, and brings "
               "the result back in a format your app understands.")

with tabs[3]:
    st.dataframe(pd.DataFrame({
        "": ["Idea", "Analogy", "LLM example", "Pros", "Cons"],
        "Serverful": ["Dedicated server/VM, always running", "Kitchen open 24×7 even with no orders",
                      "LLaMA/Mistral on your VM behind FastAPI/Flask", "Full control, consistent latency (always warm)",
                      "Pay when idle, manual scaling, you patch & monitor"],
        "Serverless": ["Cloud spins up compute on demand", "Kitchen that appears when you order",
                       "Calling Gemini via the SDK", "No infra, pay per use, elastic (1 → 1M calls)",
                       "Cold starts, less control, vendor lock-in"],
    }).set_index(""), width="stretch")
    st.markdown("#### 💰 Cost intuition")
    req = st.slider("Requests per day", 0, 200_000, 5_000, 1_000)
    vm = st.number_input("Serverful: VM cost per month ($)", value=600)
    per = st.number_input("Serverless: cost per 1,000 requests ($)", value=2.0)
    sl = req * 30 / 1000 * per
    c1, c2 = st.columns(2)
    c1.metric("Serverful / month", f"${vm:,.0f}")
    c2.metric("Serverless / month", f"${sl:,.0f}", delta=f"{'cheaper' if sl < vm else 'more expensive'}",
              delta_color="normal" if sl < vm else "inverse")
    st.caption("Low or spiky traffic favours serverless; steady heavy traffic can favour a dedicated server.")

with tabs[4]:
    definition("JSON (JavaScript Object Notation)", "a lightweight text format of **key-value pairs** for exchanging data — "
               "the language APIs speak. Not a programming language. Like a clinic form with labelled fields.")
    raw = st.text_area("Edit the JSON (try breaking it — remove a quote or add a trailing comma)",
                       json.dumps({"name": "Nikhil", "age": 22, "interests": ["AI", "Coding", "Space"], "is_student": True}, indent=2),
                       height=200)
    try:
        obj = json.loads(raw)
        c1, c2 = st.columns(2)
        with c1:
            st.success("Valid JSON")
            st.json(obj)
        with c2:
            if isinstance(obj, dict):
                st.dataframe(pd.DataFrame({"key": list(obj), "value": [json.dumps(v) for v in obj.values()],
                                           "type": [type(v).__name__ for v in obj.values()]}), hide_index=True, width="stretch")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}")
        st.caption("This exact error is what crashes apps when an LLM returns slightly broken JSON — Module 6 shows how to handle it.")

with tabs[5]:
    st.subheader("Call a real API")
    st.caption("Uses the free mock API **jsonplaceholder.typicode.com** — writes are faked, nothing is stored.")
    c1, c2 = st.columns([1, 3])
    method = c1.selectbox("HTTP method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
    defaults = {"GET": "/users/1", "POST": "/posts", "PUT": "/posts/1", "PATCH": "/posts/1", "DELETE": "/posts/1"}
    path = c2.text_input("Endpoint", defaults[method], key=f"ep_{method}")
    meaning = {"GET": "Give me something (read)", "POST": "Here's data to process/create — LLM APIs use POST to send prompts",
               "PUT": "Replace an existing resource completely", "PATCH": "Change part of a resource", "DELETE": "Remove it"}
    st.caption(f"**{method}** → {meaning[method]}")
    body = None
    if method in ("POST", "PUT", "PATCH"):
        body = st.text_area("Request body (JSON)", json.dumps({"title": "Learning APIs", "body": "JSON is easy", "userId": 1}, indent=2))
    if st.button("Send request", type="primary"):
        url = "https://jsonplaceholder.typicode.com" + path
        try:
            t0 = time.perf_counter()
            r = requests.request(method, url, json=json.loads(body) if body else None, timeout=10)
            ms = (time.perf_counter() - t0) * 1000
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", r.status_code, r.reason)
            c2.metric("Time", f"{ms:.0f} ms")
            c3.metric("Content-Type", r.headers.get("content-type", "?").split(";")[0])
            try:
                st.json(r.json())
            except ValueError:
                st.code(r.text[:2000])
            st.markdown(f"**Client:** this app · **Server:** jsonplaceholder · **API:** `{method} {path}` · **Format:** JSON")
        except Exception as e:  # noqa: BLE001
            st.error(f"Network error: {e}. (Are you offline or behind a firewall?)")
    st.code(f"""import requests
resp = requests.{method.lower()}("https://jsonplaceholder.typicode.com{path}"{', json={...}' if body else ''})
print(resp.status_code)   # 200 / 201 / 404 ...
print(resp.json())        # parsed JSON → Python dict""", language="python")

with tabs[6]:
    codes = pd.DataFrame([
        ("1xx", 100, "Continue", "Headers received, send the body", "Informational"),
        ("2xx", 200, "OK", "Success, here is your data", "Success"),
        ("2xx", 201, "Created", "New resource created (POST)", "Success"),
        ("3xx", 301, "Moved Permanently", "Resource has a new URL", "Redirection"),
        ("3xx", 302, "Found", "Temporarily redirected", "Redirection"),
        ("4xx", 400, "Bad Request", "Malformed / missing data — fix your input", "Client error"),
        ("4xx", 401, "Unauthorized", "Missing or wrong API key", "Client error"),
        ("4xx", 403, "Forbidden", "Key valid, but no permission", "Client error"),
        ("4xx", 404, "Not Found", "Resource (or model name!) doesn't exist", "Client error"),
        ("4xx", 429, "Too Many Requests", "Rate limit hit — slow down, back off", "Client error"),
        ("5xx", 500, "Internal Server Error", "Server-side problem — retry later", "Server error"),
        ("5xx", 503, "Service Unavailable", "Overloaded/maintenance — retry with backoff", "Server error"),
    ], columns=["class", "code", "name", "meaning", "category"])
    cat = st.multiselect("Filter", codes["category"].unique().tolist(), default=codes["category"].unique().tolist())
    st.dataframe(codes[codes["category"].isin(cat)], hide_index=True, width="stretch")
    st.markdown("#### What should your code do?")
    code = st.selectbox("Your app received…", [400, 401, 404, 429, 500, 503])
    action = {400: "Fix the request (validate input) — retrying the same thing won't help.",
              401: "Check the API key / .env loading. Don't retry.",
              404: "Check the URL or model name. Don't retry.",
              429: "Wait and retry with exponential backoff; add caching/rate limiting.",
              500: "Retry a few times with backoff, then show a friendly error.",
              503: "Retry with backoff (2s, 4s, 8s…) — exactly what `call_with_retry` in your chatbot does."}
    st.info(action[code])


# ---------------------------------------------------------------------------
with tabs[7]:
    takeaway([
        "DNS finds the IP; TCP handshakes; data travels as packets; the receiver reassembles; FIN-ACK closes.",
        "Client asks, server answers, the API is the agreed way to ask (the waiter).",
        "Serverful = always-on control; serverless = on-demand, pay-per-use (how Gemini is served).",
        "APIs speak JSON: key-value pairs.",
        "GET reads, POST sends (LLM prompts), PUT/PATCH update, DELETE removes; 2xx ok, 4xx your fault, 5xx theirs.",
    ])


with tabs[8]:
    quiz("m51", [
        {"q": "What does DNS do?", "options": ["Encrypts data", "Translates domain names to IP addresses", "Splits data into packets", "Closes connections"],
         "answer": 1, "why": "DNS is the phonebook of the internet."},
        {"q": "Which HTTP method do LLM APIs typically use to send a prompt?", "options": ["GET", "POST", "DELETE", "PATCH"],
         "answer": 1, "why": "You send data (the prompt) to be processed."},
        {"q": "Your app gets 401 from Gemini. Most likely cause?", "options": ["Server overloaded", "Invalid/missing API key", "Model too slow", "Bad JSON from the model"],
         "answer": 1, "why": "401 Unauthorized = credentials problem."},
        {"q": "Calling Gemini through its SDK is an example of…", "options": ["Serverful API", "Serverless API", "Peer-to-peer", "Local inference"],
         "answer": 1, "why": "Google spins up inference on demand; you manage no servers."},
    ])
