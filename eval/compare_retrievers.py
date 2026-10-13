"""
Compare dense vs BM25 vs hybrid retrieval on the gold set.

Run from the repo root:
    python -m eval.compare_retrievers
    python -m eval.compare_retrievers --qa eval/qa_set_credit_card.jsonl --show-misses

Only rows that have gold_doc + gold_sections are scored (safety rows are skipped).
Writes eval/results/retrieval_<timestamp>.json
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from datetime import datetime
from pathlib import Path

from eval.metrics import aggregate, evaluate_retrieval, order_categories

ROOT = Path(__file__).resolve().parents[1]
MODES = ("dense", "bm25", "hybrid")


def load_rows(patterns: list[str]) -> list[dict]:
    rows: list[dict] = []
    for pat in patterns:
        for path in sorted(glob.glob(str(ROOT / pat)) or glob.glob(pat)):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    seen, uniq = set(), []
    for r in rows:  # drop duplicate ids across files
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    return [r for r in uniq if r.get("gold_doc") and r.get("gold_sections")]


def print_table(per_mode: dict[str, dict], metric: str) -> None:
    cats = order_categories(next(iter(per_mode.values())).keys())
    print(f"\n{metric}")
    print(f"{'category':<12}{'n':>4}  " + "".join(f"{m:>9}" for m in per_mode))
    for c in cats:
        n = next(iter(per_mode.values()))[c]["n"]
        print(f"{c:<12}{n:>4}  " + "".join(f"{per_mode[m][c][metric]:>9.2f}" for m in per_mode))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", nargs="+", default=["eval/qa_set*.jsonl"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--show-misses", action="store_true", help="print queries where hybrid missed")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    for name in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)

    from retrieval.hybrid import get_retriever

    rows = load_rows(args.qa)
    if not rows:
        raise SystemExit("No scorable rows found (need gold_doc + gold_sections).")
    r = get_retriever()
    print(f"{len(rows)} questions | k={args.k}")

    per_mode, details = {}, {}
    for mode in MODES:
        def fn(q: str, mode=mode):
            hits = r.retrieve(q, top_k=args.k, mode=mode)
            return [(h.chunk.metadata.doc, h.chunk.metadata.section) for h in hits]

        res = evaluate_retrieval(rows, fn)
        details[mode] = res
        per_mode[mode] = aggregate(res)

    for metric in ("recall@5", "mrr", "coverage@5"):
        print_table(per_mode, metric)

    if args.show_misses:
        print("\nHybrid misses:")
        for x in details["hybrid"]:
            if x["first_hit_rank"] is None:
                print(f"- {x['id']} [{x['category']}] top: {x['top'][:2]}")

    out = ROOT / "eval" / "results" / f"retrieval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"k": args.k, "summary": per_mode, "details": details}, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()