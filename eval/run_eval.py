"""Retrieval evaluation for bank-rag (Phase 3a: no LLM needed).

Run from the repo root:
    python -m eval.run_eval                          # all modes, hit@1/3/5 + MRR
    python -m eval.run_eval --modes hybrid --k 1 3 5 10
    python -m eval.run_eval --check                  # validate qa_set.jsonl and gold snippets only
    python -m eval.run_eval --find "every two years" # find a phrase in the chunks (to write gold_snippet)

qa_set.jsonl: one JSON object per line.
    id              unique id, e.g. "pol-01"
    category        policy | exact_code | advice | pii
    question        the user query
  policy / exact_code (retrieval is scored on these):
    gold_docs       list of substrings; the chunk's document title must contain one of them
    gold_snippet    exact phrase that must appear in the gold chunk (copy it from `--find`
                    output or chunks.jsonl, not from the PDF, so it matches the extracted text)
    gold_answer     reference answer, used by the answer-accuracy step (Phase 3b)
    must_contain    strings the final answer must contain (numbers, codes)
  advice:
    should_refuse   true: the system must decline personal financial advice
  pii:
    forbidden_in_output   raw (fake) PII strings that must never appear in the answer/trace/cache
    expected_pii_types    types the PII layer should detect
  any:
    verified        false until you have checked the gold fields against the source document
    notes           free text

A retrieved chunk counts as a hit if its document matches `gold_docs` AND its text contains
`gold_snippet` (whitespace, case and hyphen spacing are ignored when comparing).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime

from app.config import settings
from retrieval.ingest import load_chunks

CATEGORIES = {"policy", "exact_code", "advice", "pii"}
RETRIEVAL_CATEGORIES = ("policy", "exact_code")
ALL_MODES = ("dense", "bm25", "hybrid")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def norm(s: str) -> str:
    """Case-, whitespace- and hyphen-spacing-insensitive form used for matching."""
    s = (
        s.replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
    )
    s = re.sub(r"\s*-\s*", "-", s)  # "high- risk" == "high-risk"
    return re.sub(r"\s+", " ", s).strip().lower()


def load_qa(path) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} not found")
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            sys.exit(f"{path}:{n}: invalid JSON ({exc})")
    return rows


def validate(rows: list[dict]) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for r in rows:
        rid = r.get("id", "?")
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
        cat = r.get("category")
        if cat not in CATEGORIES:
            problems.append(f"{rid}: category must be one of {sorted(CATEGORIES)}")
        if not r.get("question"):
            problems.append(f"{rid}: question is empty")
        if cat in RETRIEVAL_CATEGORIES:
            if not r.get("gold_docs"):
                problems.append(f"{rid}: gold_docs is required")
            if not r.get("gold_snippet"):
                problems.append(f"{rid}: gold_snippet missing (document-only matching is weak)")
        if cat == "advice" and r.get("should_refuse") is not True:
            problems.append(f"{rid}: advice rows need should_refuse: true")
        if cat == "pii" and not r.get("forbidden_in_output"):
            problems.append(f"{rid}: pii rows need forbidden_in_output")
    return problems


def gold_matches(chunk, row: dict) -> bool:
    docs = row.get("gold_docs") or []
    if docs and not any(d.lower() in chunk.metadata.doc.lower() for d in docs):
        return False
    snippet = row.get("gold_snippet")
    if snippet and norm(snippet) not in norm(chunk.text):
        return False
    return True


def check_gold(rows: list[dict], chunks) -> int:
    """Warn when a gold snippet matches no chunk (typo, stale index) or too many (too generic)."""
    bad = 0
    for r in rows:
        if r.get("category") not in RETRIEVAL_CATEGORIES:
            continue
        n = sum(1 for c in chunks if gold_matches(c, r))
        if n == 0:
            print(f"  GOLD NOT FOUND  {r['id']}: no chunk matches gold_docs + gold_snippet")
            bad += 1
        elif n > 5:
            print(f"  TOO GENERIC     {r['id']}: {n} chunks match; use a more specific snippet")
    return bad


def find(phrase: str, limit: int = 10) -> None:
    p = norm(phrase)
    n = 0
    for c in load_chunks():
        t = norm(c.text)
        i = t.find(p)
        if i < 0:
            continue
        n += 1
        if n <= limit:
            m = c.metadata
            print(f"\n[{m.doc[:55]} | {m.section} | p.{m.page}]")
            print("   ..." + t[max(0, i - 70) : i + len(p) + 70] + "...")
    print(f"\n{n} chunk(s) contain {phrase!r}")


# ----------------------------------------------------------------------
# Retrieval evaluation
# ----------------------------------------------------------------------
def evaluate(rows: list[dict], modes, ks) -> list[dict]:
    from retrieval.hybrid import get_retriever

    retriever = get_retriever()
    max_k = max(ks)
    fetch_k = max(settings.fetch_k, max_k)
    per_question = []
    for r in rows:
        if r.get("category") not in RETRIEVAL_CATEGORIES:
            continue
        ranks = {}
        for mode in modes:
            hits = retriever.retrieve(r["question"], top_k=max_k, fetch_k=fetch_k, mode=mode)
            ranks[mode] = next((h.rank for h in hits if gold_matches(h.chunk, r)), None)
        per_question.append({"id": r["id"], "category": r["category"], "question": r["question"], "ranks": ranks})
    return per_question


def summarize(per_question: list[dict], modes, ks) -> dict:
    summary: dict = {}
    groups = {"ALL": per_question}
    for cat in RETRIEVAL_CATEGORIES:
        groups[cat] = [q for q in per_question if q["category"] == cat]
    for mode in modes:
        summary[mode] = {}
        for name, qs in groups.items():
            if not qs:
                continue
            ranks = [q["ranks"][mode] for q in qs]
            entry = {"n": len(qs)}
            for k in ks:
                entry[f"hit@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / len(qs)
            entry["mrr"] = sum((1 / r) if r else 0.0 for r in ranks) / len(qs)
            summary[mode][name] = entry
    return summary


def print_summary(summary: dict, modes, ks) -> None:
    head = f"{'mode':<8} {'group':<11} {'n':>3} " + " ".join(f"{'hit@' + str(k):>6}" for k in ks) + f" {'MRR':>6}"
    print("\n" + head)
    print("-" * len(head))
    for mode in modes:
        for name, e in summary[mode].items():
            cells = " ".join(f"{e[f'hit@{k}']:>6.2f}" for k in ks)
            print(f"{mode:<8} {name:<11} {e['n']:>3} {cells} {e['mrr']:>6.2f}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", default=str(settings.eval_qa_path))
    ap.add_argument("--modes", nargs="+", default=list(ALL_MODES), choices=ALL_MODES)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--check", action="store_true", help="validate the question set only")
    ap.add_argument("--find", metavar="PHRASE", help="search chunks for a phrase and exit")
    args = ap.parse_args()

    if args.find:
        find(args.find)
        return

    from pathlib import Path

    rows = load_qa(Path(args.qa))
    counts = defaultdict(int)
    for r in rows:
        counts[r.get("category", "?")] += 1
    unverified = sum(1 for r in rows if not r.get("verified"))
    print(f"{len(rows)} questions: " + ", ".join(f"{c}={n}" for c, n in sorted(counts.items())))
    print(f"{unverified} not yet marked verified")

    problems = validate(rows)
    for p in problems:
        print(f"  PROBLEM  {p}")
    bad_gold = check_gold(rows, load_chunks())
    if args.check:
        print(f"\n{len(problems)} schema problem(s), {bad_gold} gold snippet(s) not found")
        sys.exit(1 if problems or bad_gold else 0)

    per_question = evaluate(rows, args.modes, args.k)
    summary = summarize(per_question, args.modes, args.k)
    print_summary(summary, args.modes, args.k)

    max_k = max(args.k)
    for mode in args.modes:
        missed = [q for q in per_question if q["ranks"][mode] is None]
        if missed:
            print(f"Not found in top {max_k} ({mode}):")
            for q in missed:
                print(f"  {q['id']}: {q['question'][:90]}")

    settings.ensure_dirs()
    out = settings.eval_results_dir / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(
            {
                "config": {
                    "embedding_model": settings.embedding_model,
                    "chunk_size_tokens": settings.chunk_size_tokens,
                    "chunk_overlap_tokens": settings.chunk_overlap_tokens,
                    "fetch_k": settings.fetch_k,
                    "top_k": settings.top_k,
                    "rrf_k": settings.rrf_k,
                    "n_chunks": len(load_chunks()),
                    "k_values": args.k,
                },
                "summary": summary,
                "per_question": per_question,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()