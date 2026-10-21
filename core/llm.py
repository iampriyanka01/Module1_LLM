"""
core/llm.py
===========
One place for everything that talks to Gemini.

Every page imports from here, so the "how do I call the model" logic
(API-key lookup, client creation, retry with exponential backoff,
error messages) exists exactly once - the same idea as the
study_buddy.py / evaluator.py split in the course project.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ClientError, ServerError

    GENAI_AVAILABLE = True
except ImportError:  # the app still works offline for explanations/simulators
    GENAI_AVAILABLE = False
    genai = types = None
    APIError = ClientError = ServerError = Exception

try:
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    InMemoryRateLimiter = ChatGoogleGenerativeAI = None

# Load .env that sits next to app.py, no matter where Streamlit was launched
# from (the "key not found" bug from the chatbot exercise).
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

MODEL_CHOICES = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
]
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# At most 4 model calls a minute (the free tier allows 5).
REQUESTS_PER_MINUTE = 4


# ---------------------------------------------------------------------------
# Shared rate limiter
# ---------------------------------------------------------------------------
class _SimpleLimiter:
    """Fallback token bucket with the same acquire() API, used only if
    langchain-core isn't installed."""

    def __init__(self, requests_per_second: float, check_every_n_seconds: float = 0.5, max_bucket_size: float = 1):
        import threading
        self.rate, self.check, self.max = requests_per_second, check_every_n_seconds, max_bucket_size
        self.tokens, self.last, self.lock = 0.0, None, threading.Lock()

    def _consume(self) -> bool:
        with self.lock:
            now = time.monotonic()
            if self.last is None:
                self.last = now
            self.tokens = min(self.tokens + (now - self.last) * self.rate, self.max)
            self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    def acquire(self, *, blocking: bool = True) -> bool:
        if not blocking:
            return self._consume()
        while not self._consume():
            time.sleep(self.check)
        return True


@st.cache_resource(show_spinner=False)
def get_rate_limiter():
    """ONE limiter for the whole app process - every page, every chat, every
    parallel worker thread and the LangChain `llm` all draw from it."""
    cls = InMemoryRateLimiter if LANGCHAIN_AVAILABLE else _SimpleLimiter
    return cls(requests_per_second=REQUESTS_PER_MINUTE / 60, check_every_n_seconds=0.5, max_bucket_size=1)


def _rate_limited(fn):
    limiter = get_rate_limiter()

    def wrapper(*args, **kwargs):
        limiter.acquire()  # blocks until a slot is free
        return fn(*args, **kwargs)

    wrapper.__wrapped__ = fn
    return wrapper


# ---------------------------------------------------------------------------
# Key + client
# ---------------------------------------------------------------------------
def env_api_key() -> str | None:
    """Accept either variable name (the course files used both)."""
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def get_api_key() -> str | None:
    return st.session_state.get("api_key") or env_api_key()


def get_model() -> str:
    return st.session_state.get("model") or DEFAULT_MODEL


def get_client():
    """Return a cached genai.Client for the current key (None if no key)."""
    key = get_api_key()
    if not key or not GENAI_AVAILABLE:
        return None
    cached = st.session_state.get("_client")
    if cached is not None and st.session_state.get("_client_key") == key:
        return cached
    client = genai.Client(api_key=key)
    # Route every generation call through the shared limiter. Chat sessions
    # (client.chats.create(...).send_message) call these same methods on the
    # same Models object, so they are limited too.
    models = client.models
    models.generate_content = _rate_limited(models.generate_content)
    models.generate_content_stream = _rate_limited(models.generate_content_stream)
    st.session_state["_client"] = client
    st.session_state["_client_key"] = key
    return client


def get_llm(model: str | None = None, **kwargs):
    """LangChain chat model that shares the same rate limiter:

        limiter = InMemoryRateLimiter(requests_per_second=4 / 60, check_every_n_seconds=0.5, max_bucket_size=1)
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", rate_limiter=limiter)
    """
    key = get_api_key()
    if not key or not LANGCHAIN_AVAILABLE:
        return None
    mdl = model or get_model()
    cache_id = (key, mdl, tuple(sorted(kwargs.items())))
    cache = st.session_state.setdefault("_llms", {})
    if cache_id not in cache:
        cache[cache_id] = ChatGoogleGenerativeAI(model=mdl, google_api_key=key,
                                                 rate_limiter=get_rate_limiter(), **kwargs)
    return cache[cache_id]


def llm_ready(show_hint: bool = True) -> bool:
    """True when a live call is possible; otherwise optionally explain why."""
    if not GENAI_AVAILABLE:
        if show_hint:
            st.warning("Install the Gemini SDK to run live demos: `pip install google-genai`")
        return False
    if not get_api_key():
        if show_hint:
            st.info(
                "🔑 This is a **live demo**. Add your Gemini API key in the sidebar "
                "(or put `GEMINI_API_KEY=...` in a `.env` file next to `app.py`) to run it. "
                "Everything else on this page works without a key."
            )
        return False
    return True


# ---------------------------------------------------------------------------
# Retry with exponential backoff (Module 5.2 best practice)
# ---------------------------------------------------------------------------
def call_with_retry(fn, *, max_attempts: int = 4, base_delay: float = 2, notify: bool = True):
    """Retry only transient server errors (5xx / overload). Client errors
    (bad key, bad request) are re-raised immediately - retrying won't fix them."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except ServerError:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))  # 2s, 4s, 8s ...
            if notify:
                st.toast(f"Gemini is busy (attempt {attempt}/{max_attempts}). Retrying in {delay:.0f}s…")
            time.sleep(delay)


def friendly_error(e: Exception, model: str | None = None) -> str:
    text = str(e)
    if isinstance(e, ServerError):
        return f"😕 Gemini's servers are overloaded or unavailable (5xx). Try again in a moment.\n\n`{text[:300]}`"
    if isinstance(e, ClientError):
        if "API key" in text or "API_KEY" in text or "401" in text or "403" in text:
            return f"❌ The API key was rejected (4xx). Check the key in the sidebar.\n\n`{text[:300]}`"
        if "404" in text or "not found" in text.lower():
            return (f"❌ Model `{model or get_model()}` was not found for your key. Pick another model "
                    f"in the sidebar.\n\n`{text[:300]}`")
        if "429" in text:
            return (f"⏳ Rate limit hit (429 Too Many Requests). This app already paces itself to "
                    f"{REQUESTS_PER_MINUTE} calls/minute, so your key's quota may be used up elsewhere (or the daily "
                    f"limit reached). Wait a bit and retry.\n\n`{text[:300]}`")
        return f"❌ Request rejected (4xx): `{text[:400]}`"
    return f"⚠️ Unexpected error: `{text[:400]}`"


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def make_config(system: str | None = None, temperature=None, top_p=None, top_k=None,
                max_tokens=None, tools=None, json_mode: bool = False, model=None, **extra):
    kwargs = {}
    # Gemini 3.x: sampling params are left at defaults (Google recommends it; lite may reject them)
    if (model or "").startswith("gemini-3"):
        temperature = top_p = top_k = None
    if system:
        kwargs["system_instruction"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    if tools is not None:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    kwargs.update(extra)
    return types.GenerateContentConfig(**kwargs) if kwargs else None


def response_text(resp) -> str:
    """resp.text can be None (e.g. output cut off, safety block, only a tool call)."""
    try:
        txt = resp.text
    except Exception:
        txt = None
    if txt:
        return txt
    try:
        reason = resp.candidates[0].finish_reason
    except Exception:
        reason = "unknown"
    return f"_(The model returned no text. finish_reason = {reason}. If you set a small token limit, raise it.)_"


def generate(prompt, *, system=None, temperature=None, top_p=None, top_k=None,
             max_tokens=None, model=None, json_mode=False, retry=True):
    """Single generate_content call. Returns (text, raw_response, seconds)."""
    client = get_client()
    cfg = make_config(system, temperature, top_p, top_k, max_tokens, json_mode=json_mode, model=model or get_model())
    t0 = time.perf_counter()
    call = lambda: client.models.generate_content(model=model or get_model(), contents=prompt, config=cfg)
    try:
        resp = call_with_retry(call) if retry else call()
    except Exception as e:
        log_call(prompt, None, time.perf_counter() - t0, error=str(e)[:120], model=model)
        raise
    secs = time.perf_counter() - t0
    log_call(prompt, resp, secs, model=model)
    return response_text(resp), resp, secs


def log_call(prompt, resp, secs, *, error=None, model=None, kind="generate"):
    """Session-wide request log (Module 5.2: logging & monitoring)."""
    u = usage(resp) if resp is not None else {}
    st.session_state.setdefault("call_log", []).append({
        "time": time.strftime("%H:%M:%S"),
        "page": st.session_state.get("_current_page", ""),
        "kind": kind,
        "model": model or get_model(),
        "prompt (first 60 chars)": (prompt if isinstance(prompt, str) else str(prompt))[:60],
        "input tok": u.get("input tokens"),
        "output tok": u.get("output tokens"),
        "latency s": round(secs, 2),
        "status": "error" if error else "ok",
        "error": error or "",
    })


def generate_many(prompts, *, system=None, temperature=None, max_tokens=None, model=None,
                  json_mode=False, max_workers: int = 5):
    """Run several prompts concurrently (thread pool). Streamlit's session
    state isn't available inside worker threads, so client/model are
    resolved here first. Returns a list of (text | None, error | None, seconds)
    in the same order as `prompts` - a failure in one doesn't sink the rest."""
    from concurrent.futures import ThreadPoolExecutor

    client, mdl = get_client(), model or get_model()
    cfg = make_config(system, temperature, max_tokens=max_tokens, json_mode=json_mode, model=mdl)

    def one(p):
        t0 = time.perf_counter()
        try:
            r = call_with_retry(lambda: client.models.generate_content(model=mdl, contents=p, config=cfg),
                                notify=False, base_delay=1)
            return response_text(r), None, time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            return None, friendly_error(e, mdl), time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(one, prompts))
    for p, (txt, err, secs) in zip(prompts, results):  # log from the main thread
        log_call(p, None, secs, error=err, model=mdl, kind="parallel")
    return results


def stream(prompt, *, system=None, temperature=None, model=None):
    """Yield text chunks as they arrive (generate_content_stream)."""
    client = get_client()
    cfg = make_config(system, temperature, model=model or get_model())
    for chunk in client.models.generate_content_stream(model=model or get_model(), contents=prompt, config=cfg):
        if getattr(chunk, "text", None):
            yield chunk.text


def usage(resp) -> dict:
    u = getattr(resp, "usage_metadata", None)
    if not u:
        return {}
    return {
        "input tokens": u.prompt_token_count or 0,
        "output tokens": u.candidates_token_count or 0,
        "thinking tokens": u.thoughts_token_count or 0,
        "total tokens": u.total_token_count or 0,
    }


def run_live(label: str, fn, *, key: str,
             spinner: str = f"Asking Gemini… (paced to {REQUESTS_PER_MINUTE} calls/min, so multi-call demos take a moment)",
             button_type="primary"):
    """Button + spinner + friendly errors. fn() is executed only on click.
    The result is stored in session_state[key] so it survives reruns."""
    if st.button(label, key=f"btn_{key}", type=button_type, disabled=not llm_ready(show_hint=False)):
        with st.spinner(spinner):
            try:
                st.session_state[key] = fn()
            except Exception as e:  # noqa: BLE001 - surface any SDK error nicely
                st.session_state[key] = None
                st.error(friendly_error(e))
    if not get_api_key():
        st.caption("🔑 Add a Gemini API key in the sidebar to enable this button.")
    return st.session_state.get(key)
