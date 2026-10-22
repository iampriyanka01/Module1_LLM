"""Editable, runnable code boxes for the lessons.

    code_lab("m6_stream", CODE, title="Stream a response")

Shows the code in an editable area, a ▶ Run button, a ↺ Reset button and the
program's printed output (updated live, so streaming looks like streaming).

SECURITY: the code runs with `exec` on the machine hosting the app, using that
machine's API key. That is fine on your own laptop or a private app. On a
public deployment anyone could run code on your server, so set
`CODE_LAB_DISABLED=1` (env var or Streamlit secret) to turn the Run button off.
"""
from __future__ import annotations

import contextlib
import io
import os
import time
import traceback

import streamlit as st

from core.llm import get_api_key, get_client, get_llm, get_model, llm_ready

_CSS = """
<style>
textarea[aria-label^="✏️ Editable code"] { font-family: "Source Code Pro", Menlo, Consolas, monospace !important;
    font-size: 0.85rem !important; line-height: 1.45 !important; tab-size: 4; }
</style>
"""


def _disabled() -> bool:
    flag = os.getenv("CODE_LAB_DISABLED", "")
    try:
        flag = flag or str(st.secrets.get("CODE_LAB_DISABLED", ""))
    except Exception:  # noqa: BLE001 - no secrets file
        pass
    return flag.strip().lower() in {"1", "true", "yes", "on"}


class _Live(io.TextIOBase):
    """stdout replacement that refreshes a placeholder as text arrives."""

    def __init__(self, placeholder):
        self.ph, self.buf, self._last = placeholder, "", 0.0

    def write(self, s):
        self.buf += s
        now = time.perf_counter()
        if now - self._last > 0.08:
            self.ph.code(self.buf or " ", language="text", wrap_lines=True)
            self._last = now
        return len(s)

    def flush(self):
        self.ph.code(self.buf or " ", language="text", wrap_lines=True)


def _namespace(extra: dict | None) -> dict:
    import json
    import math
    import re

    import pandas as pd
    from google import genai
    from google.genai import types

    from core.llm import generate

    key = get_api_key()
    ns = {
        "__name__": "__codelab__",
        "genai": genai, "types": types, "os": os, "time": time, "re": re, "json": json, "math": math, "pd": pd,
        "client": get_client(),            # shared client, already rate-limited
        "MODEL": get_model(),
        "llm": get_llm(),                  # LangChain chat model (None if not installed)
        "generate": generate,
        "new_client": lambda: genai.Client(api_key=key),   # use inside asyncio code (fresh client per event loop)
    }
    ns.update(extra or {})
    return ns


def code_lab(key: str, code: str, *, title: str | None = None, height: int | None = None,
             extra: dict | None = None, needs_key: bool = True, note: str | None = None):
    """Editable code + Run. `code` is the starting text; ↺ Reset restores it."""
    st.markdown(_CSS, unsafe_allow_html=True)
    sk = f"cl_{key}"
    if sk not in st.session_state:
        st.session_state[sk] = code
    lines = code.count("\n") + 1
    h = height or min(560, max(140, 24 * lines + 40))
    label = f"✏️ Editable code — {title}" if title else "✏️ Editable code"
    if _disabled():
        st.code(code, language="python")
        st.caption("Editing/running is switched off on this deployment (CODE_LAB_DISABLED).")
        return
    st.text_area(label, key=sk, height=h)
    c1, c2, c3 = st.columns([1.3, 1, 5])
    run = c1.button("▶ Run code", key=f"{sk}_run", type="primary",
                    disabled=needs_key and not llm_ready(show_hint=False))
    if c2.button("↺ Reset", key=f"{sk}_reset"):
        st.session_state[sk] = code
        st.session_state.pop(f"{sk}_out", None)
        st.rerun()
    c3.caption(note or "Change anything and press Run. Variables ready for you: `client`, `MODEL`, `types`, `genai`, `llm`, `generate`.")
    if needs_key and not get_api_key():
        st.caption("🔑 Add a Gemini API key in the sidebar to run this.")
    out_ph = st.empty()
    if run:
        src = st.session_state[sk]
        live = _Live(out_ph)
        t0 = time.perf_counter()
        err = None
        with contextlib.redirect_stdout(live):
            try:
                exec(compile(src, "<your code>", "exec"), _namespace(extra))  # noqa: S102 - intentional, see module docstring
            except BaseException as e:  # noqa: BLE001 - show any error to the learner
                err = "".join(traceback.format_exception_only(type(e), e)).strip()
                if len(err) > 600:
                    err = err[:600] + " …"
        secs = time.perf_counter() - t0
        st.session_state[f"{sk}_out"] = (live.buf, err, secs)
    saved = st.session_state.get(f"{sk}_out")
    if saved:
        buf, err, secs = saved
        out_ph.code(buf or "(no output — add a print(...) to see something)", language="text", wrap_lines=True)
        if err:
            st.error(f"⚠️ {err}")
        st.caption(f"Finished in {secs:.2f}s")
