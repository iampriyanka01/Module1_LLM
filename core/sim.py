"""
core/sim.py
===========
Offline simulators used across lessons (no API key needed):
next-token sampling (temperature / top-k / top-p), a toy tokenizer,
and a token-count estimate.
"""

from __future__ import annotations

import math
import random
import re

import pandas as pd

# "The cat sat on the ___" - illustrative next-token probabilities
CAT_SENTENCE = "The cat sat on the ___"
CAT_PROBS = {
    "mat": 0.45, "sofa": 0.18, "floor": 0.12, "roof": 0.08, "bed": 0.06,
    "windowsill": 0.04, "throne": 0.03, "keyboard": 0.02, "spaceship": 0.01, "banana": 0.01,
}

STARS_SENTENCE = "The stars ___ across the midnight sky"
STARS_PROBS = {
    "twinkled": 0.30, "shone": 0.22, "sparkled": 0.14, "shimmered": 0.10, "danced": 0.08,
    "glittered": 0.06, "waltzed": 0.04, "sighed": 0.03, "marched": 0.02, "exploded": 0.01,
}


def apply_temperature(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """Re-scale a distribution: p_i^(1/T) normalised. T→0 = greedy, T>1 = flatter."""
    t = max(temperature, 1e-3)
    logits = {w: math.log(p) / t for w, p in probs.items()}
    m = max(logits.values())
    exp = {w: math.exp(v - m) for w, v in logits.items()}
    z = sum(exp.values())
    return {w: v / z for w, v in exp.items()}


def top_k_filter(probs: dict[str, float], k: int) -> set[str]:
    ranked = sorted(probs, key=probs.get, reverse=True)
    return set(ranked[:k])


def top_p_filter(probs: dict[str, float], p: float) -> set[str]:
    """Smallest set of top tokens whose cumulative probability reaches p."""
    kept, cum = set(), 0.0
    for w in sorted(probs, key=probs.get, reverse=True):
        kept.add(w)
        cum += probs[w]
        if cum >= p - 1e-9:
            break
    return kept


def decode_table(base: dict[str, float], temperature: float, top_k: int, top_p: float) -> pd.DataFrame:
    scaled = apply_temperature(base, temperature)
    keep = top_k_filter(scaled, top_k) & top_p_filter(scaled, top_p)
    z = sum(scaled[w] for w in keep) or 1
    rows = []
    for w in sorted(scaled, key=scaled.get, reverse=True):
        rows.append({
            "token": w,
            "original %": round(base[w] * 100, 1),
            "after temperature %": round(scaled[w] * 100, 1),
            "kept?": "✅" if w in keep else "—",
            "final sampling %": round(scaled[w] / z * 100, 1) if w in keep else 0.0,
        })
    return pd.DataFrame(rows)


def sample(df: pd.DataFrame, n: int, seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    pool = df[df["final sampling %"] > 0]
    return rng.choices(list(pool["token"]), weights=list(pool["final sampling %"]), k=n)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
_COMMON = {"the", "a", "an", "is", "of", "to", "and", "in", "it", "on", "for", "you", "i", "my", "ai"}


def toy_tokenize(text: str) -> list[str]:
    """Illustrative sub-word tokenizer: short/common words stay whole, long
    words are split into chunks, punctuation is its own token. Real
    tokenizers (BPE/SentencePiece) learn the splits from data."""
    tokens = []
    for piece in re.findall(r"\s?\w+|[^\w\s]", text):
        word = piece.strip()
        lead = " " if piece.startswith(" ") else ""
        if not word.isalpha() or len(word) <= 6 or word.lower() in _COMMON:
            tokens.append(lead + word)
        else:
            chunks = [word[i:i + 4] for i in range(0, len(word), 4)]
            if len(chunks[-1]) < 2 and len(chunks) > 1:
                chunks[-2] += chunks.pop()
            tokens.append(lead + chunks[0])
            tokens.extend(chunks[1:])
    return tokens


def estimate_tokens(text: str) -> int:
    """Rule of thumb for English: ~4 characters per token."""
    return max(1, round(len(text) / 4)) if text else 0
