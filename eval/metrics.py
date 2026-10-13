"""
Pure scoring functions for the eval (no project imports, easy to unit test).

Retrieval scoring: a retrieved chunk is a HIT when its doc matches row["gold_doc"]
AND its section matches any of row["gold_sections"].
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Iterable

KS = (1, 3, 5)


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def doc_match(gold_doc: str, hit_doc: str | None) -> bool:
    g, h = _norm(gold_doc), _norm(hit_doc)
    return bool(g) and bool(h) and (g == h or g in h or h in g)


def section_match(gold_sec: str, hit_sec: str | None) -> bool:
    """Equal, or one is a prefix of the other (tolerates truncated headings)."""
    g, h = _norm(gold_sec), _norm(hit_sec)
    return bool(g) and bool(h) and (g == h or h.startswith(g) or g.startswith(h))


def score_retrieval(row: dict, hits: list[tuple[str | None, str | None]]) -> dict:
    """hits = [(doc, section), ...] best first. Returns per-query metrics."""
    gold_secs = row.get("gold_sections") or []
    first: int | None = None
    covered_at: dict[int, set[int]] = {}  # rank -> gold section indexes newly covered
    for rank, (doc, sec) in enumerate(hits, start=1):
        if not doc_match(row["gold_doc"], doc):
            continue
        for i, gs in enumerate(gold_secs):
            if section_match(gs, sec):
                covered_at.setdefault(rank, set()).add(i)
                if first is None:
                    first = rank
    out: dict = {"first_hit_rank": first, "mrr": (1.0 / first) if first else 0.0}
    for k in KS:
        out[f"recall@{k}"] = 1.0 if (first is not None and first <= k) else 0.0
        cov = set().union(*(s for r, s in covered_at.items() if r <= k)) if covered_at else set()
        out[f"coverage@{k}"] = (len(cov) / len(gold_secs)) if gold_secs else 0.0
    return out


def evaluate_retrieval(
    rows: Iterable[dict], retrieve_fn: Callable[[str], list[tuple[str | None, str | None]]]
) -> list[dict]:
    """retrieve_fn(query) -> [(doc, section), ...]. One result dict per row."""
    results = []
    for row in rows:
        hits = retrieve_fn(row["q"])
        res = score_retrieval(row, hits)
        res.update(
            id=row["id"],
            category=row["category"],
            top=[f"{d} | {s}" for d, s in hits],
        )
        results.append(res)
    return results


METRICS = ["recall@1", "recall@3", "recall@5", "mrr", "coverage@5"]


def aggregate(results: list[dict], metrics: list[str] = METRICS) -> dict[str, dict[str, float]]:
    """Mean of each metric per category, plus an 'ALL' row. Each row also carries 'n'."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        buckets[r["category"]].append(r)
        buckets["ALL"].append(r)
    out = {}
    for cat, rs in buckets.items():
        out[cat] = {"n": len(rs)}
        for m in metrics:
            vals = [r[m] for r in rs if r.get(m) is not None]
            out[cat][m] = sum(vals) / len(vals) if vals else float("nan")
    return out


# ----------------------------------------------------------------------
# Answer scoring (coarse, cheap). Use the LLM judge for real correctness.
# ----------------------------------------------------------------------
def fact_recall(gold_answer: str, answer: str, extract: Callable[[str], set[str]]) -> float | None:
    """Share of the gold answer's numbers (days, amounts, dates) that appear in the answer.
    None when the gold answer contains no numbers (metric not applicable)."""
    gold = extract(gold_answer)
    if not gold:
        return None
    return len(gold & extract(answer)) / len(gold)


CATEGORY_ORDER = ["exact-code", "paraphrase", "vague", "overall"]


def order_categories(cats: Iterable[str]) -> list[str]:
    """Known categories in a fixed order, then any others alphabetically, 'ALL' last."""
    cats = [c for c in cats if c != "ALL"]
    known = [c for c in CATEGORY_ORDER if c in cats]
    other = sorted(c for c in cats if c not in CATEGORY_ORDER)
    return known + other + ["ALL"]