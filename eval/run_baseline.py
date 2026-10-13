"""
Plain-RAG baseline: retrieve top-k -> generate -> score. This is the number the
self-healing graph has to beat in phase 8.

Run from the repo root:
    python -m eval.run_baseline                       # hybrid retrieval, cheap metrics only
    python -m eval.run_baseline --mode dense --judge  # dense baseline + LLM judge
    python -m eval.run_baseline --limit 5             # smoke test (5 questions)

Per question it records: retrieval hit, number-recall vs gold answer, numbers in the
answer that are NOT in the retrieved chunks (ungrounded), optional judge verdict,
latency and tokens. Results go to eval/results/baseline_<mode>_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from eval.compare_retrievers import ROOT, load_rows
from eval.metrics import fact_recall, order_categories, score_retrieval

REFUSAL_MARKER = "don't have enough information"


def run_baseline(
    rows: list[dict],
    retrieve_fn: Callable[[str], list],           # q -> list[RetrievedChunk]
    generate_fn: Callable[[str, list], dict],     # (q, hits) -> {answer, prompt_tokens, ...}
    extract: Callable[[str], set[str]],
    ungrounded_fn: Callable[[str, list[dict]], list[str]],
    judge_fn: Callable[[dict, str], dict] | None = None,
    pause: float = 0.0,
) -> list[dict]:
    results = []
    for i, row in enumerate(rows, start=1):
        rec = {"id": row["id"], "category": row["category"], "q": row["q"]}
        try:
            hits = retrieve_fn(row["q"])
            pairs = [(h.chunk.metadata.doc, h.chunk.metadata.section) for h in hits]
            rec.update(score_retrieval(row, pairs))
            gen = generate_fn(row["q"], hits)
            ans = gen["answer"]
            docs = [{"text": h.chunk.text} for h in hits]
            rec.update(
                answer=ans,
                gold_answer=row.get("gold_answer"),
                fact_recall=fact_recall(row.get("gold_answer", ""), ans, extract),
                ungrounded=ungrounded_fn(ans, docs),
                refused=REFUSAL_MARKER in ans.lower(),
                latency_s=gen.get("latency_s", 0.0),
                prompt_tokens=gen.get("prompt_tokens", 0),
                completion_tokens=gen.get("completion_tokens", 0),
            )
            if judge_fn:
                v = judge_fn(row, ans)
                rec.update(judge_correct=bool(v.get("correct")), judge_reason=v.get("reason", ""))
        except Exception as exc:  # one bad question must not kill the run
            rec["error"] = f"{type(exc).__name__}: {exc}"
        results.append(rec)
        print(f"[{i}/{len(rows)}] {row['id']:<8} " + ("ERROR " + rec["error"] if "error" in rec else
              f"hit@5={rec['recall@5']:.0f} facts={rec['fact_recall'] if rec['fact_recall'] is None else round(rec['fact_recall'], 2)}"
              + (f" judge={rec['judge_correct']}" if judge_fn else "")))
        if pause:
            time.sleep(pause)
    return results


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def summarize(results: list[dict]) -> dict[str, dict]:
    ok = [r for r in results if "error" not in r]
    cats = order_categories({r["category"] for r in ok})
    out = {}
    for c in cats:
        rs = ok if c == "ALL" else [r for r in ok if r["category"] == c]
        out[c] = {
            "n": len(rs),
            "recall@5": _mean([r["recall@5"] for r in rs]),
            "fact_recall": _mean([r["fact_recall"] for r in rs if r["fact_recall"] is not None]),
            "ungrounded_rate": _mean([1.0 if r["ungrounded"] else 0.0 for r in rs]),
            "refusal_rate": _mean([1.0 if r["refused"] else 0.0 for r in rs]),
            "avg_latency_s": _mean([r["latency_s"] for r in rs]),
            "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in rs),
        }
        judged = [r for r in rs if "judge_correct" in r]
        if judged:
            out[c]["judge_acc"] = _mean([1.0 if r["judge_correct"] else 0.0 for r in judged])
    return out


def print_summary(summary: dict[str, dict]) -> None:
    has_judge = any("judge_acc" in v for v in summary.values())
    cols = ["recall@5", "fact_recall", "ungrounded_rate", "refusal_rate", "avg_latency_s"] + (["judge_acc"] if has_judge else [])
    print(f"\n{'category':<12}{'n':>4}" + "".join(f"{c:>17}" for c in cols))
    for cat, v in summary.items():
        print(f"{cat:<12}{v['n']:>4}" + "".join(f"{v.get(c, float('nan')):>17.2f}" for c in cols))
    print(f"\ntotal tokens: {summary['ALL']['tokens']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", nargs="+", default=["eval/qa_set*.jsonl"])
    ap.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judge", action="store_true", help="LLM-as-judge correctness (extra call per question)")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between questions (free-tier limits)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    for name in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "google_genai"):
        logging.getLogger(name).setLevel(logging.ERROR)

    from llm.client import USAGE, get_client, load_prompt
    from llm.generator import generate_answer
    from retrieval.hybrid import get_retriever
    from security.numeric_guard import check, extract

    rows = [r for r in load_rows(args.qa) if r.get("gold_answer")]
    if args.limit:
        rows = rows[: args.limit]
    retriever = get_retriever()

    def judge(row: dict, answer: str) -> dict:
        prompt = (load_prompt("judge").replace("{question}", row["q"])
                  .replace("{gold}", row["gold_answer"]).replace("{answer}", answer))
        return get_client().generate_json(prompt)

    results = run_baseline(
        rows,
        retrieve_fn=lambda q: retriever.retrieve(q, top_k=args.k, mode=args.mode),
        generate_fn=generate_answer,
        extract=extract,
        ungrounded_fn=lambda ans, docs: check(ans, docs)["ungrounded"],
        judge_fn=judge if args.judge else None,
        pause=args.pause,
    )
    summary = summarize(results)
    print_summary(summary)
    errs = [r for r in results if "error" in r]
    if errs:
        print(f"\n{len(errs)} question(s) errored; see the JSON for details.")

    out = ROOT / "eval" / "results" / f"baseline_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "mode": args.mode, "k": args.k, "judge": args.judge,
        "llm_usage": vars(USAGE), "summary": summary, "details": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()