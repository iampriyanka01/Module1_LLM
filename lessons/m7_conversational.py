import math
import re
import time
from collections import Counter

import pandas as pd
import streamlit as st

from core import sim
from core.llm import (call_with_retry, friendly_error, generate, get_client, get_model, llm_ready, log_call,
                      make_config, response_text, run_live)
from core.ui import analogy, compare, definition, page_header, quiz, stepper, takeaway

page_header(
    "m7",
    "Chat Completion: roles, system prompts, the illusion of memory, and managing the context window.",
    [
        "Explain why one-shot text generation fails for conversations",
        "Structure chats with system, user and assistant messages",
        "Control persona and behaviour with a system prompt",
        "Explain how Chat Completion works under the hood (it's stateless!)",
        "Manage the context window with truncation, summarization and prioritization — and save cost",
    ],
)

tabs = st.tabs(["Why plain prompts fail", "Roles & messages", "🎭 System-prompt playground", "Under the hood",
                "🧠 Memory & context window", "✂️ Context strategies", "💬 Your auto-summary chatbot", "OpenAI-compatible code", "🧾 Key takeaways", "✅ Knowledge check"])

# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("The simple movie chatbot")
    st.markdown("User: *Who directed Inception?* → Bot: *Christopher Nolan.*  \nUser: *What other movies did **he** direct?* → "
                "Bot: *I'm sorry, who do you mean?* 🤦")
    lim = [
        ("1 · No built-in memory", "Every prompt starts from scratch — like a support agent asking for your order ID every sentence."),
        ("2 · Manual context stitching", "You paste the whole history into each prompt: longer, costlier, error-prone."),
        ("3 · No role awareness", "One blob of text — the model can't tell instruction vs user vs its own reply."),
        ("4 · No behaviour control", "“Respond concisely” must be repeated in every prompt or it's forgotten."),
        ("5 · Not scalable", "A call centre where every call reaches a new agent with zero history."),
    ]
    for t, d in lim:
        st.markdown(f"**{t}** — {d}")
    st.markdown("#### 🟢 Prove it: stateless calls vs a chat session")
    st.code("""# stateless                                    # chat session
client.models.generate_content(..., "I have 2 dogs in my house.")   chat = client.chats.create(model=MODEL)
client.models.generate_content(..., "How many paws are in my house?")   chat.send_message("I have 2 dogs in my house.")
                                                                         chat.send_message("How many paws are in my house?")""",
            language="python")

    def _compare():
        a1 = generate("I have 2 dogs in my house.")[0]
        a2 = generate("How many paws are in my house? Answer briefly.")[0]
        chat = get_client().chats.create(model=get_model())
        b1 = response_text(call_with_retry(lambda: chat.send_message("I have 2 dogs in my house.")))
        b2 = response_text(call_with_retry(lambda: chat.send_message("How many paws are in my house? Answer briefly.")))
        hist = [(m.role, m.parts[0].text if m.parts else "") for m in chat.get_history()]
        return a2, b2, hist

    res = run_live("Run both", _compare, key="m7_paws")
    if res:
        c1, c2 = st.columns(2)
        with c1.container(border=True):
            st.caption("Stateless — 2nd answer")
            st.markdown(res[0])
        with c2.container(border=True):
            st.caption("Chat session — 2nd answer")
            st.markdown(res[1])
        with st.expander("chat.get_history()"):
            for role, text in res[2]:
                st.markdown(f"**{role.capitalize()}:** {text[:300]}")

# ---------------------------------------------------------------------------
with tabs[1]:
    definition("Chat Completion", "interacting with a model through a structured list of role-tagged messages. It "
               "(1) formats the chat history into model input and (2) generates the next **assistant** message.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 🎬 system\nThe invisible director: persona, tone, rules. Set once at the start — like the note a "
                "support agent reads before every call.")
    c2.markdown("### 🙋 user\nThe human: questions, commands, input. Every user message is something to respond to.")
    c3.markdown("### 🤖 assistant\nThe model's replies. You only write these yourself to replay earlier history.")
    analogy("A student asks (user), the tutor answers (assistant), and an off-stage moderator says “be concise and simple” (system).",
            title="Classroom")
    st.markdown("#### 🛠️ Build a conversation — every message is just `role` + `content`")
    if "m7_msgs" not in st.session_state:
        st.session_state.m7_msgs = [
            {"role": "system", "content": "You are a helpful travel planner."},
            {"role": "user", "content": "What's a good 3-day itinerary for Goa?"},
            {"role": "assistant", "content": "Day 1: North Goa beaches… Day 2: Aguada Fort, Anjuna… Day 3: Morjim, Panjim cruise."},
            {"role": "user", "content": "Can you also suggest a couple-friendly resort near Morjim?"},
        ]
    msgs = st.session_state.m7_msgs
    edited = st.data_editor(pd.DataFrame(msgs), num_rows="dynamic", width="stretch", key="m7_editor",
                            column_config={"role": st.column_config.SelectboxColumn(options=["system", "user", "assistant"], required=True)})
    msgs = [m for m in edited.to_dict("records") if m.get("role") and m.get("content")]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What your code sends (structured)**")
        st.json(msgs, expanded=True)
    with c2:
        st.markdown("**What the model actually reads (flattened)**")
        st.code("\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in msgs) + "\nAssistant:", language="text")
    st.caption("The SDK/backend flattens the list into one prompt; the model then predicts the next assistant tokens.")

# ---------------------------------------------------------------------------
PERSONAS = {
    "Poetic assistant": "You are a poetic assistant. Always answer in rhymes or verses, and maintain a lyrical tone.",
    "Physics teacher": "You are a knowledgeable high school physics teacher. Explain concepts clearly with examples, avoid jargon unless necessary.",
    "Friendly & informal": "You are a friendly and informal assistant. Use emojis, ask follow-up questions, keep a light, cheerful tone.",
    "Structured (3 bullets)": "You are a structured assistant. Always answer in exactly 3 bullet points. Never add commentary.",
    "Movie expert, concise": "You are a movie expert. Respond in at most 2 sentences. Recommend only family-friendly films.",
}
with tabs[2]:
    st.subheader("One instruction, every reply")
    st.markdown("The system prompt sets the model's **initial mindset**; every response is filtered through it. Without it, "
                "behaviour is generic and inconsistent.")
    persona = st.selectbox("Persona", list(PERSONAS) + ["(write your own)"], key="m7_persona")
    sys_prompt = st.text_area("System instruction", PERSONAS.get(persona, "You are …"), key=f"m7_sys_{persona}")
    if st.session_state.get("m7_chat_sys") != sys_prompt:
        st.session_state.m7_chat_sys = sys_prompt
        st.session_state.m7_chat_hist = []
    if st.button("🔄 Reset chat", key="m7_reset"):
        st.session_state.m7_chat_hist = []
    for m in st.session_state.m7_chat_hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if llm_ready():
        q = st.chat_input("Ask anything — e.g. “What is the moon made of?”", key="m7_chat_in")
        if q:
            st.session_state.m7_chat_hist.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                        for m in st.session_state.m7_chat_hist]
            with st.chat_message("assistant"):
                try:
                    t0 = time.perf_counter()
                    resp = call_with_retry(lambda: get_client().models.generate_content(
                        model=get_model(), contents=contents, config=make_config(system=sys_prompt)))
                    ans = response_text(resp)
                    log_call(q, resp, time.perf_counter() - t0, kind="chat")
                except Exception as e:  # noqa: BLE001
                    ans = friendly_error(e)
                st.markdown(ans)
            st.session_state.m7_chat_hist.append({"role": "assistant", "content": ans})
    st.caption("Notice: the whole history is re-sent on every turn (`contents=[...]`). That's the 'memory'.")

# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("How does an LLM “chat”? It predicts one token at a time")
    st.markdown("1. Split text into **tokens** → 2. turn tokens into **vectors** → 3. **attention** relates every token to the "
                "others (“cat … on the” → expect a location) → 4. score every possible next token → 5. **pick** one → 6. repeat.")
    st.markdown("#### Step 5 — decoding strategies")
    strat = st.radio("Strategy", ["Greedy", "Random sampling", "Top-k", "Top-p (nucleus)"], horizontal=True)
    k, p, t = 10, 1.0, 1.0
    if strat == "Greedy":
        t = 0.01
    elif strat == "Top-k":
        k = st.slider("k", 1, 10, 3)
    elif strat == "Top-p (nucleus)":
        p = st.slider("p", 0.1, 1.0, 0.9, 0.05)
    df = sim.decode_table(sim.CAT_PROBS, t, k, p)
    st.bar_chart(df.set_index("token")["final sampling %"], height=220)
    notes = {"Greedy": "Always the top token ('mat'). Deterministic and fast; can be dull and repetitive.",
             "Random sampling": "Any token, weighted by probability — creative but can be incoherent.",
             "Top-k": "Sample only from the k best — balances precision and creativity; k is arbitrary.",
             "Top-p (nucleus)": "Smallest set covering p of the probability — adapts to the model's confidence."}
    st.info(notes[strat])
    st.markdown("#### What happens after you hit Enter on “My name is Ninja”")
    stepper("chat_flow", [
        ("Wrap as a message", '`{"role": "user", "content": "My name is Ninja"}` is appended to the history list (after the system message).'),
        ("Flatten", "System: You are a helpful assistant.\\nUser: My name is Ninja → one prompt string."),
        ("Tokenize", "“My” → 3798, “name” → 1715, “is” → 318, “Ninja” → 92005 …"),
        ("Predict token by token", "“Nice” → “Nice to” → “Nice to meet” → “Nice to meet you, Ninja!” (streamed)"),
        ("Decode", "Token IDs turned back into readable text."),
        ("Wrap & append", '`{"role": "assistant", "content": "Nice to meet you…"}` is added to the history.'),
        ("Next turn: “What is my name?”", "The **whole list** (system + both earlier messages + new question) is sent again. "
                                          "The model “remembers” Ninja only because it's in the prompt."),
    ])

# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("It's not memory — it's prompt engineering")
    st.markdown("LLMs are **stateless**. In text completion, “How tall is **it**?” fails because the model never saw "
                "“Mount Everest”. In chat completion, the app re-sends the full history every call — that's the illusion of memory.")
    analogy("A conversation on a whiteboard of fixed size: as you keep writing, the oldest lines are wiped to make room. "
            "Ask about something that's been wiped and they'll say “Sorry, can you remind me?”", title="Rolling whiteboard")
    definition("Context window", "the maximum number of tokens a model can see in one request — system message, every "
               "user and assistant turn, formatting and whitespace all count.")
    convo = [("system", "You are a helpful assistant."), ("user", "Hi! My name is Ninja and I'm vegetarian."),
             ("assistant", "Nice to meet you, Ninja! How can I help?"), ("user", "Plan a 3-day trip to Goa for me."),
             ("assistant", "Day 1: Calangute & Baga beaches, water sports, dinner at a beach shack. Day 2: Aguada Fort, Anjuna flea market, sunset at Chapora. Day 3: Morjim beach and a Mandovi river cruise."),
             ("user", "Which beach is quietest?"), ("assistant", "Ashwem and Morjim are quieter than Baga or Calangute."),
             ("user", "Suggest a resort near Morjim."), ("assistant", "Look at boutique beach-hut resorts along Morjim–Ashwem."),
             ("user", "Now suggest dinner places for me on day 2.")]
    toks = [sim.estimate_tokens(c) + 4 for _, c in convo]
    window = st.slider("Context window (tokens)", 20, sum(toks), 70, 5, key="m7_window")
    visible, used = set(), 0
    sys_t = toks[0]
    used = sys_t
    visible.add(0)
    for i in range(len(convo) - 1, 0, -1):  # keep newest first, system always kept
        if used + toks[i] <= window:
            visible.add(i)
            used += toks[i]
        else:
            break
    for i, ((role, content), t) in enumerate(zip(convo, toks)):
        icon = "🟢" if i in visible else "⚪"
        st.markdown(f"{icon} `{t:>3} tok` **{role}:** " + (content if i in visible else f"~~{content}~~"))
    if 1 not in visible:
        st.error("The message with **“Ninja, vegetarian”** fell out of the window → the dinner suggestions may include non-veg places "
                 "and the model may ask your name again.")
    else:
        st.success("The name/diet message is still visible → personalised answer possible.")
    st.markdown("#### Context windows of popular models (from the notes)")
    st.dataframe(pd.DataFrame([
        ("GPT-3.5 Turbo", "16K"), ("GPT-4 (standard)", "8K"), ("GPT-4 (32K variant)", "32K"), ("GPT-4 Turbo / GPT-4o", "128K"),
        ("Gemini 1.5 Pro", "128K (1M preview)"), ("Gemini 2.0 / 2.5 Pro", "1M"), ("Claude 2.1", "200K"), ("Llama-4 Scout / Maverick", "1M+"),
    ], columns=["Model", "Max context window (tokens)"]), hide_index=True, width="stretch")
    st.caption("Values as stated in the course notes; newer models change often — check the provider's docs.")

# ---------------------------------------------------------------------------
def bow(text):
    return Counter(w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2)


def cosine(a, b):
    num = sum(a[w] * b[w] for w in a)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


with tabs[5]:
    st.subheader("Three ways to fit a long conversation — and cut cost")
    history = [
        ("user", "Hi, how are you?"), ("assistant", "Great, thanks! How can I help?"),
        ("user", "I want to go to Goa for 3 days with my partner. Budget is 40,000 rupees."),
        ("assistant", "Here's a suggested North Goa itinerary with romantic beach stays…"),
        ("user", "lol I binge-watched a show till 3am yesterday 😅"), ("assistant", "Ha! Hope you got some rest."),
        ("user", "We're vegetarian and don't drink alcohol."), ("assistant", "Noted — I'll suggest veg-friendly cafés and skip bar-heavy spots."),
        ("user", "Is Ashwem beach good for couples?"), ("assistant", "Yes, Ashwem is quiet and scenic — great for couples."),
    ]
    question = st.text_input("Current question", "Suggest restaurants for our dinner on day 2.", key="m7_q")
    strat = st.radio("Strategy", ["Full history", "Truncation (last N)", "Summarization", "Prioritization (relevance)"], horizontal=True)
    if strat == "Full history":
        ctx = [f"{r}: {c}" for r, c in history]
    elif strat == "Truncation (last N)":
        n = st.slider("Keep last N messages", 1, len(history), 4)
        ctx = [f"{r}: {c}" for r, c in history[-n:]]
        st.caption("Simple sliding window — but here it drops the budget, and if N < 5 also the 'vegetarian' constraint.")
    elif strat == "Summarization":
        summary = ("Summary so far: User and partner are planning a 3-day North Goa trip, budget ₹40,000, vegetarian, no alcohol; "
                   "likes quiet beaches such as Ashwem.")
        ctx = [summary]
        st.caption("Older turns compressed into one line (a real app asks the LLM to write this). Caution: summaries can drop nuance.")
    else:
        qv = bow(question + " food vegetarian restaurant budget trip goa")
        scored = sorted(((cosine(bow(c), qv), r, c) for r, c in history), reverse=True)
        top = st.slider("Keep top-k most relevant messages", 1, len(history), 3)
        keep = {c for _, _, c in scored[:top]}
        ctx = [f"{r}: {c}" for r, c in history if c in keep]
        st.dataframe(pd.DataFrame([{"relevance": round(s, 2), "role": r, "message": c} for s, r, c in scored]), hide_index=True, width="stretch")
        st.caption("Offline stand-in for **embedding-based relevance**: real systems embed each message and compare meaning, "
                   "not just shared words. Rule-based **categorization** (tagging 'high priority' messages) is the manual alternative.")
    prompt_ctx = "\n".join(ctx + [f"user: {question}"])
    st.code(prompt_ctx, language="text")
    full_tokens = sim.estimate_tokens("\n".join(f"{r}: {c}" for r, c in history)) + sim.estimate_tokens(question)
    now_tokens = sim.estimate_tokens(prompt_ctx)
    c1, c2 = st.columns(2)
    c1.metric("Prompt tokens (≈)", now_tokens, f"{now_tokens - full_tokens:+d} vs full history", delta_color="inverse")
    c2.metric("Kept 'vegetarian'?", "Yes" if "vegetarian" in prompt_ctx.lower() else "No — answer may be wrong")
    res = run_live("🟢 Ask Gemini with this context", lambda: generate(prompt_ctx + "\nassistant:")[0], key=f"m7_ctx_{strat}")
    if res:
        with st.container(border=True):
            st.markdown(res)

    st.markdown("#### 💸 Why it matters for cost: tokens re-sent over a long chat")
    turns = st.slider("Conversation length (turns)", 5, 60, 30)
    per_turn = st.slider("Tokens per turn (user + assistant)", 50, 600, 200, 50)
    rows = []
    cum = {"Full history": 0, "Truncation (last 4 turns)": 0, "Summary + last 2 turns": 0}
    for t in range(1, turns + 1):
        cum["Full history"] += t * per_turn
        cum["Truncation (last 4 turns)"] += min(t, 4) * per_turn
        cum["Summary + last 2 turns"] += (min(t, 2) * per_turn) + (80 if t > 2 else 0)
        rows.append({"turn": t, **cum})
    df = pd.DataFrame(rows).set_index("turn")
    st.line_chart(df, height=260)
    st.caption(f"After {turns} turns: full history sends **{cum['Full history']:,}** tokens in total vs "
               f"**{cum['Summary + last 2 turns']:,}** with summary + recent turns. Always ask: does the model need this, and is it worth paying for?")

# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Your `chatbot_app.py` — upgraded")
    st.markdown("Your app summarised the conversation every 5 user turns and showed it in the sidebar. Here the same idea "
                "becomes a **context-management strategy**: the running summary is fed back to the model so older turns "
                "can be dropped.")
    with st.expander("What changed vs your original"):
        st.markdown(
            "- **Same:** `call_with_retry` with exponential backoff on `ServerError`, no retry on `ClientError`; summary via a "
            "separate one-off call so it doesn't pollute chat history.\n"
            "- **New:** memory mode *Summary + last N turns* actually sends `summary + recent turns` instead of the whole history, "
            "and shows the token difference.\n"
            "- **Fixed:** the original read `GOOGLE_API_KEY` but the warning asked for `GEMINI_API_KEY` — this app accepts both.")
    c1, c2, c3 = st.columns(3)
    every = c1.number_input("Summarize every N user turns", 2, 10, 3, key="m7_every")
    mode = c2.radio("Memory mode", ["Full history", "Summary + last N turns"], key="m7_mode")
    keep_n = c3.number_input("N recent turns to keep", 1, 6, 2, key="m7_keepn")
    ss = st.session_state
    ss.setdefault("sb_msgs", [])
    ss.setdefault("sb_summary", "")
    ss.setdefault("sb_turns", 0)
    ss.setdefault("sb_stats", [])
    if st.button("🔄 Reset conversation", key="sb_reset"):
        ss.sb_msgs, ss.sb_summary, ss.sb_turns, ss.sb_stats = [], "", 0, []
    if ss.sb_summary:
        st.info(f"📋 **Running summary (after {ss.sb_turns - ss.sb_turns % every} turns):** {ss.sb_summary}")
    for m in ss.sb_msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    def build_contents():
        msgs = ss.sb_msgs
        if mode == "Summary + last N turns" and ss.sb_summary:
            recent = msgs[-(keep_n * 2 - 1):]  # last N user turns incl. the new one
            pre = [{"role": "user", "parts": [{"text": f"(Context summary of earlier conversation: {ss.sb_summary})"}]},
                   {"role": "model", "parts": [{"text": "Understood, I'll keep that in mind."}]}]
        else:
            recent, pre = msgs, []
        return pre + [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in recent]

    if llm_ready():
        q = st.chat_input("Type your message…", key="sb_in")
        if q:
            ss.sb_msgs.append({"role": "user", "content": q})
            ss.sb_turns += 1
            with st.chat_message("user"):
                st.markdown(q)
            contents = build_contents()
            with st.chat_message("assistant"):
                try:
                    t0 = time.perf_counter()
                    resp = call_with_retry(lambda: get_client().models.generate_content(model=get_model(), contents=contents))
                    ans = response_text(resp)
                    log_call(q, resp, time.perf_counter() - t0, kind="chat")
                    u = resp.usage_metadata
                    full = sum(sim.estimate_tokens(m["content"]) for m in ss.sb_msgs)
                    ss.sb_stats.append({"turn": ss.sb_turns, "input tokens sent": (u.prompt_token_count if u else None),
                                        "≈ full-history tokens": full})
                except Exception as e:  # noqa: BLE001
                    ans = friendly_error(e)
                st.markdown(ans)
            ss.sb_msgs.append({"role": "assistant", "content": ans})
            if ss.sb_turns % every == 0:
                with st.spinner("Summarizing conversation…"):
                    convo = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in ss.sb_msgs)
                    prev = f"Previous summary: {ss.sb_summary}\n\n" if ss.sb_summary else ""
                    try:
                        ss.sb_summary = generate(prev + "Summarize the following conversation in 3-4 concise sentences, capturing "
                                                 "key facts about the user, topics and decisions:\n\n" + convo)[0]
                        st.toast(f"📋 New summary after {ss.sb_turns} turns")
                    except Exception as e:  # noqa: BLE001
                        st.warning(f"Summary failed: {e}")
                st.rerun()
    if ss.sb_stats:
        st.dataframe(pd.DataFrame(ss.sb_stats), hide_index=True, width="stretch")
        st.caption("Switch memory mode to *Summary + last N turns* after a few turns and watch input tokens stop growing.")

# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Already have OpenAI-style code? Point it at Gemini")
    st.markdown("Gemini exposes an **OpenAI-compatible endpoint**, including function calling. Change three things: the key, the "
                "`base_url` and the model name — everything else stays the same.")
    st.code("""from openai import OpenAI

client = OpenAI(
    api_key="YOUR_GEMINI_API_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "You are a friendly and concise assistant."},
        {"role": "user", "content": "What's the capital of Japan?"},
        {"role": "assistant", "content": "Tokyo."},
        {"role": "user", "content": "Give me three facts about it."},
    ],
    # tools=[...], tool_choice="auto"   # function calling works too
)
print(response.choices[0].message.content)""", language="python")
    compare("Google GenAI SDK roles", "`user` and **`model`** (system goes in `system_instruction`)",
            "OpenAI-style roles", "`system`, `user`, **`assistant`**", bad_good=False)


# ---------------------------------------------------------------------------
with tabs[8]:
    takeaway([
        "Plain prompts have no memory, roles or persistent behaviour — Chat Completion adds structure.",
        "Messages = role + content; roles are system (director), user (human), assistant (model).",
        "The system prompt sets persona, tone and rules once for the whole session.",
        "LLMs are stateless: 'memory' = re-sending history inside the context window.",
        "Truncate, summarize or prioritize to fit the window — the same strategies cut token cost.",
    ])


with tabs[9]:
    quiz("m7", [
        {"q": "Which role sets persona and rules for the whole conversation?", "options": ["user", "assistant", "system", "tool"],
         "answer": 2, "why": "The system message is the invisible director."},
        {"q": "How does a chat model “remember” your name from earlier?", "options": ["It stores it in its weights",
         "The app re-sends the history in the prompt", "It searches the internet", "It has a long-term memory database"],
         "answer": 1, "why": "Models are stateless; history lives in the prompt."},
        {"q": "Your 8K-token chat exceeds the window. Simplest fix?", "options": ["Truncation — keep the last N messages",
         "Raise the temperature", "Remove the system prompt", "Switch to greedy decoding"], "answer": 0,
         "why": "A sliding window keeps the most recent turns (but may drop key facts)."},
        {"q": "Summarizing old turns mainly saves…", "options": ["Accuracy", "Tokens (and cost) while keeping intent", "API keys", "Latency to zero"],
         "answer": 1, "why": "Hundreds of tokens of backstory become a short summary."},
    ])
