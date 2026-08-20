"""Chunking strategy eval: 4 strategies x all 13 languages.
Per lang: fixed selected-passage pool, in-memory dense-only index (mirrors prod build),
recall@20 (token overlap >= 0.6 vs gold) + answerable rate on val queries, same queries for all 4.
"""
from __future__ import annotations
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_passage, _strategy_names
from src.embedder import Embedder
from src.indexer import Index
from src.retrieval import retrieve

LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "ur"]
LANG_FILES = {"as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal",
              "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "ta": "tam", "ur": "urd"}


def tokenize(t: str) -> set[str]:
    return {w for w in t.lower().split() if w.isalpha()}


def overlap_ok(chunk: str, gold: str) -> bool:
    c, g = tokenize(chunk), tokenize(gold)
    if not c or not g:
        return False
    return len(c & g) / len(c | g) >= 0.6


def recall_at_20(hits, gold_texts: list[str]) -> bool:
    for h in hits:
        if any(overlap_ok(h.text, g) for g in gold_texts):
            return True
    return False


def run_lang(lang: str, pool_size: int, n_queries: int, strategies: list[str],
             embedder: Embedder) -> dict:
    from datasets import load_dataset
    fname = f"validation/{LANG_FILES[lang]}val.parquet"
    ds = load_dataset("ai4bharat/MSMARCO-XI", data_files=fname, split="train", streaming=True)
    it = iter(ds)

    rows = []
    for _ in range(n_queries):
        try:
            rows.append(next(it))
        except StopIteration:
            break

    pool: list[tuple[str, str]] = []  # (passage_text, qid)
    seen: set[str] = set()
    for row in rows:
        for s, t in zip(row["passages"]["is_selected"], row["passages"]["Translated_passages"]):
            if s == 1 and t.strip() and t not in seen and len(pool) < pool_size:
                seen.add(t)
                pool.append((t, str(row["query_id"])))
    if len(pool) < 200:
        extra = iter(ds)
        for row in extra:
            for s, t in zip(row["passages"]["is_selected"], row["passages"]["Translated_passages"]):
                if s == 1 and t.strip() and t not in seen and len(pool) < pool_size:
                    seen.add(t)
                    pool.append((t, str(row["query_id"])))
            if len(pool) >= pool_size:
                break

    golds = [[t for t in row["passages"]["Translated_passages"]
              if t.strip() and t in seen] for row in rows]

    out = {"lang": lang, "queries": len(rows), "pool_passages": len(pool), "strategies": {}}
    for strat in strategies:
        t0 = time.time()
        chunks = []
        for i, (t, qid) in enumerate(pool):
            cs = chunk_passage(str(i), t, strat, lang=lang, qid=qid, selected=1, qtype="")
            chunks.extend(cs)
        ix = Index(lang, embedder)
        for i in range(0, len(chunks), 512):
            b = chunks[i:i + 512]
            ix.add([c["text"] for c in b], [c["meta"] for c in b],
                   embedder.encode([c["text"] for c in b]), index_toks=False)
        build_s = time.time() - t0

        hit = 0
        for row, gold in zip(rows, golds):
            q = row["query"]
            if not q.strip():
                continue
            vec = embedder.encode_one(q)
            res = retrieve(ix, vec, q, n=20)
            if recall_at_20(res, gold):
                hit += 1
        n = len(rows)
        out["strategies"][strat] = {
            "chunks": len(chunks),
            "build_s": round(build_s, 1),
            "recall_at_20": hit / n,
            "answerable": hit / n,
        }
        print(f"  {lang}/{strat}: {len(chunks)} chunks, recall@{20}={hit}/{n} "
              f"({hit/n:.3f}), {build_s:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default=None, help="single lang (default: all 13)")
    ap.add_argument("--pool", type=int, default=4000, help="selected passages per lang")
    ap.add_argument("--queries", type=int, default=300, help="val queries per lang")
    args = ap.parse_args()

    embedder = Embedder()
    langs = [args.lang] if args.lang else LANGS
    results = []
    for lang in langs:
        print(f"{lang}: collecting pool({args.pool}) + {args.queries} queries", flush=True)
        t0 = time.time()
        try:
            r = run_lang(lang, args.pool, args.queries, _strategy_names, embedder)
        except Exception as e:
            print(f"{lang}: FAILED {e}", flush=True)
            continue
        results.append(r)
        print(f"{lang}: done in {time.time()-t0:.0f}s", flush=True)

    if not results:
        print("no results")
        return

    olds = {s: [] for s in _strategy_names}
    for r in results:
        for s, d in r["strategies"].items():
            olds[s].append(d["recall_at_20"])
    print("\n=== SUMMARY (mean recall@20 per strategy) ===")
    for s in _strategy_names:
        print(f"  {s:10s} mean={statistics.mean(olds[s]):.4f}  min={min(olds[s]):.3f} "
              f"max={max(olds[s]):.3f}  per-lang: " +
              ", ".join(f"{r['lang']}={r['strategies'][s]['recall_at_20']:.3f}" for r in results))
    best = max(_strategy_names, key=lambda s: statistics.mean(olds[s]))
    print(f"\nWINNER: {best}")
    import json
    Path("data/eval_chunking.jsonl").parent.mkdir(exist_ok=True)
    with open("data/eval_chunking.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()