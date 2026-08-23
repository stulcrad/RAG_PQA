"""Evaluate retrieval quality and latency on a small hand-authored eval set.

Eval set format (data/eval.jsonl), one object per line:
    {"question": "...", "gold_ids": ["ParlaMint-CZ_..."]}

Retrieval-only on purpose: it needs no LLM, so it runs in seconds and does not
compete with the generation model for VRAM. Retrieval is also the upper bound on
answer quality - no prompt can recover a speech that was never retrieved.

    python evaluation.py                      # dense (the configured default)
    python evaluation.py --hybrid             # hybrid BM25+dense
    python evaluation.py --ablation           # both, side by side

# TODO(eval): larger auto-generated set - prompt the LLM to write a question answerable
#   only from a held-out speech; gold = that speech id. Scales to ~200 cases.
#   Caveat: questions inherit vocabulary from their source speech, so recall is optimistic.
# TODO(eval): groundedness / faithfulness - LLM-as-judge scoring whether each cited claim
#   is supported by the cited speech. Would catch the two defects seen in manual testing:
#   over-attributed citations and cross-speaker misattribution.
# TODO(eval): attribution precision - for speaker-scoped questions, the fraction of cited
#   sources actually spoken by the named MP. Cheap, needs no LLM, targets the failure above.
# TODO(eval): end-to-end answer latency (this measures retrieval only).
"""
import json
import time

from metrics import latency_stats, mrr, recall_at_k, reciprocal_rank
from pqa.config import load_config
from pqa.index import load_index
from pqa.retrieval import build_retriever


def _load_cases(eval_path: str) -> list[dict]:
    with open(eval_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _score(retriever, cases: list[dict], k: int) -> dict:
    """
    Run one retriever over every case and aggregate the metrics.
    
    Args:
        - retriever: a callable that takes a question and returns a list of hits
        - cases: a list of dicts, each with "question" and "gold_ids"
        - k: the number of top hits to consider for recall and MRR
    Returns:
        - A dict with the number of cases, recall@k, mrr@k, and retrieval latency statistics.
    """
    # Warm up first: the embedding model cold-loads on the first call and would
    # otherwise dominate the latency percentiles.
    retriever.invoke(cases[0]["question"])

    recalls, rrs, lats = [], [], []
    for case in cases:
        start = time.perf_counter()
        hits = retriever.invoke(case["question"])[:k]
        lats.append(time.perf_counter() - start)
        ids = [d.metadata["id"] for d in hits]
        recalls.append(recall_at_k(ids, case["gold_ids"], k))
        rrs.append(reciprocal_rank(ids, case["gold_ids"]))

    return {"n": len(cases), "k": k,
            f"recall@{k}": round(sum(recalls) / len(recalls), 3),
            f"mrr@{k}": round(mrr(rrs), 3),
            "retrieval_latency_s": {m: round(v, 4)
                                    for m, v in latency_stats(lats).items()}}


def evaluate_retrieval(eval_path: str, k: int | None = None,
                       hybrid: bool | None = None) -> dict:
    """
    Evaluate one retrieval setup. `hybrid=None` uses the configured default.
    
    Args:
        - eval_path: path to the eval set JSONL file
        - k: the number of top hits to consider for recall and MRR; if None, uses the configured default
        - hybrid: whether to use hybrid BM25+dense retrieval; if None, uses the configured default
    Returns:
        - A dict with the retrieval type, number of cases, recall@k, mrr@k, and retrieval latency statistics.
    """
    cfg = load_config()
    k = k or cfg.top_k
    hybrid = cfg.hybrid if hybrid is None else hybrid
    store, docs = load_index(cfg)
    retriever = build_retriever(store, docs, k, hybrid=hybrid)
    return {"retrieval": "hybrid (bm25+dense)" if hybrid else "dense", **_score(retriever, _load_cases(eval_path), k)}


def ablation(eval_path: str, k: int | None = None) -> list[dict]:
    """
    Compare dense vs hybrid on the same index, so the numbers are comparable.
    
    Args:
        - eval_path: path to the eval set JSONL file
        - k: the number of top hits to consider for recall and MRR; if None, uses the configured default
    Returns:
        - A list of dicts, each with the retrieval type, number of cases, recall@k, mrr@k,
          and retrieval latency statistics for both dense and hybrid retrieval.
    """
    cfg = load_config()
    k = k or cfg.top_k
    store, docs = load_index(cfg)          # load once, reuse for both setups
    cases = _load_cases(eval_path)
    results = []
    for hybrid in (False, True):
        if hybrid:
            for weights in ([0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0.0, 1.0]):
                retriever = build_retriever(store, docs, k, hybrid=hybrid, weights=weights)
                results.append({"retrieval": f"hybrid (bm25+dense) {weights}",
                                **_score(retriever, cases, k)})
        else:
            retriever = build_retriever(store, docs, k, hybrid=hybrid)
            results.append({"retrieval": "dense", **_score(retriever, cases, k)})
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate PQA retrieval.")
    ap.add_argument("eval_set", nargs="?", default="data/eval.jsonl")
    ap.add_argument("--k", type=int, default=None, help="Override top_k")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--hybrid", action="store_true", help="Evaluate hybrid BM25+dense")
    mode.add_argument("--ablation", action="store_true", help="Evaluate dense and hybrid side by side")
    args = ap.parse_args()

    if args.ablation:
        rows = ablation(args.eval_set, args.k)
        k = rows[0]["k"]
        print(f"{'retrieval':30s} {'recall@'+str(k):>9s} {'mrr@'+str(k):>8s} {'p50 s':>8s}")
        for r in rows:
            print(f"{r['retrieval']:30s} {r[f'recall@{k}']:9.3f} "
                  f"{r[f'mrr@{k}']:8.3f} {r['retrieval_latency_s']['p50']:8.4f}")
    else:
        result = evaluate_retrieval(args.eval_set, args.k,
                                    hybrid=True if args.hybrid else None)
        print(json.dumps(result, indent=2, ensure_ascii=False))
