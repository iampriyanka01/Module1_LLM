"""Catalog-grounded SpecSense (the "intelligence layer" from the problem statement).

The original pipeline (main.py / streamlit/app.py) asks the model to recommend a product from MEMORY - so the product,
its specs and the "customer reviews" can be invented. This module fixes that in three steps:

    1. EXTRACT   LLM turns the free-text query into hard constraints + soft preferences (JSON)
    2. MATCH     plain pandas filters the company catalog (exact, testable, no hallucination);
                 if nothing fits, it returns the products that break the fewest constraints (gap detection)
    3. EXPLAIN   LLM writes the recommendation using ONLY the retrieved rows (meets / partially / trade-offs / review / alternatives)

Run:  python catalog_matcher.py "Lightweight laptop under 30000 for coding, good battery"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CATALOG_PATH = Path(__file__).parent / "data" / "products_sample.csv"

EXTRACT_PROMPT = """Extract the shopper's needs from the query. Return ONLY JSON with these keys (use null when not mentioned):
{"max_price_inr": number|null, "max_weight_kg": number|null, "min_ram_gb": number|null, "min_storage_gb": number|null,
 "min_battery_hours": number|null, "soft_preferences": ["short phrases such as 'good for coding', 'travel friendly'"]}
Translate words into numbers when obvious (e.g. 'under 30k' -> 30000, 'lightweight' -> max_weight_kg 1.5).

Query:
"""

EXPLAIN_PROMPT = """You are a transparent product advisor. Recommend using ONLY the catalog rows provided - never invent a product,
a spec or a review. Use the row's `review` text as the review snippet.
Return ONLY JSON:
{"recommended_product": "<name>", "meets": ["hard criteria fully met"], "partially_meets": ["soft preferences only partly met"],
 "trade_offs": ["compromises"], "review_snippets": ["quote from the row's review"], "alternatives": ["other catalog names + one-line reason"],
 "gap_note": "<empty string if everything is met, else what could not be met>"}
"""

HARD = {  # constraint -> (column, comparison)
    "max_price_inr": ("price_inr", "<="), "max_weight_kg": ("weight_kg", "<="), "min_ram_gb": ("ram_gb", ">="),
    "min_storage_gb": ("storage_gb", ">="), "min_battery_hours": ("battery_hours", ">="),
}


def load_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def violations(row: pd.Series, constraints: dict) -> list[str]:
    """Which hard constraints does this product break? (empty list = fully matches)"""
    out = []
    for key, (col, op) in HARD.items():
        want = constraints.get(key)
        if want is None:
            continue
        ok = row[col] <= want if op == "<=" else row[col] >= want
        if not ok:
            out.append(f"{col} {row[col]} (wanted {op} {want})")
    return out


def shortfall(row: pd.Series, constraints: dict) -> float:
    """How far (relative) the product misses its violated constraints - used to rank near-misses."""
    total = 0.0
    for key, (col, op) in HARD.items():
        want = constraints.get(key)
        if want and ((row[col] > want) if op == "<=" else (row[col] < want)):
            total += abs(row[col] - want) / want
    return round(total, 3)


def match(df: pd.DataFrame, constraints: dict, top_k: int = 3) -> tuple[pd.DataFrame, bool]:
    """Return (rows, exact). exact=False means no product met every hard constraint, so these are the nearest."""
    df = df[df["in_stock"] == "yes"].copy()
    df["violations"] = df.apply(lambda r: violations(r, constraints), axis=1)
    df["n_viol"] = df["violations"].map(len)
    df["shortfall"] = df.apply(lambda r: shortfall(r, constraints), axis=1)
    best = df["n_viol"].min()
    exact = best == 0
    if exact:
        rows = df[df["n_viol"] == 0].sort_values("price_inr").head(top_k)
    else:  # no perfect match: show the nearest misses (fewest broken constraints, then smallest shortfall)
        rows = df.sort_values(["n_viol", "shortfall", "price_inr"]).head(top_k)
    return rows, exact


def recommend(query: str, llm=None) -> dict:
    from llm.client import LLMClient          # imported here so the pure-pandas helpers above work without an API key
    from utils import extract_json_from_text
    llm = llm or LLMClient()
    constraints = extract_json_from_text(llm.call(EXTRACT_PROMPT + query)) or {}
    rows, exact = match(load_catalog(), constraints)
    payload = rows.drop(columns=["n_viol", "shortfall"]).to_dict(orient="records")
    prompt = (f"{EXPLAIN_PROMPT}\nUser query: {query}\nExtracted constraints: {json.dumps(constraints)}\n"
              f"Exact match found: {exact}\nCatalog rows (the ONLY products you may mention):\n{json.dumps(payload, ensure_ascii=False)}")
    result = extract_json_from_text(llm.call(prompt))
    result["_constraints"], result["_exact"], result["_candidates"] = constraints, exact, rows["name"].tolist()
    return result


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Lightweight laptop under 30000 for coding, good battery"
    print(json.dumps(recommend(q), indent=2, ensure_ascii=False))
