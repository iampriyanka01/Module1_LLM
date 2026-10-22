"""
A very lightweight execution tracer.

Every meaningful step (which function ran, what it decided, how long an API call
took, what error it hit) gets logged here. `app.py` calls `start_trace()` once at
the top of handling a user message, every other module calls `log()` /
`log_exception()` as it works, and `app.py` renders `get_trace_text()` in a debug
expander at the end - so you get a full, ordered trace of exactly what happened
for that turn, without needing to dig through a terminal.

Note: this uses a module-level buffer, so it assumes one request is being traced
at a time per process - fine for a single-user local Streamlit app like this one,
but not safe for real concurrent multi-user traffic.
"""

from __future__ import annotations

import logging
import time
import traceback

logger = logging.getLogger("cinema_advisor")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(_handler)

_records: list[str] = []


def start_trace() -> None:
    """Call once at the start of handling a user turn to reset the trace buffer."""
    _records.clear()
    log("--- new turn started ---")


def log(message: str, level: str = "INFO") -> None:
    ts = time.strftime("%H:%M:%S")
    _records.append(f"[{ts}] [{level}] {message}")
    if level == "ERROR":
        logger.error(message)
    else:
        logger.debug(message)


def log_exception(context: str) -> None:
    """Log the current exception (call this from inside an except block)."""
    tb = traceback.format_exc()
    log(f"EXCEPTION in {context}:\n{tb}", level="ERROR")


class timed:
    """Context manager: `with timed("omdb.get_movie"): ...` logs start/end + duration."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        log(f"→ starting: {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed_ms = (time.perf_counter() - self.start) * 1000
        if exc_type is not None:
            log(f"✗ failed: {self.label} after {elapsed_ms:.0f}ms - {exc_type.__name__}: {exc}", level="ERROR")
        else:
            log(f"✓ finished: {self.label} in {elapsed_ms:.0f}ms")
        return False  # never suppress the exception


def get_trace_text() -> str:
    return "\n".join(_records) if _records else "(no trace recorded for this turn)"