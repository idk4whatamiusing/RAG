"""Bench (GOA-6): T2 latency + gold recall on validation queries (default 75, Hindi)."""
from __future__ import annotations
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from src.embedder import Embedder  # noqa: E402
from src.pipeline import Corpus, Pipeline  # noqa: E402


def load_val(path: Path, n: int, seed: int = 42) -> list[dict]:
    pf = pq.ParquetFile(path)
    rows = []
    for batch in pf.iter_batches(batch_size=1024, columns=["query", "query_id", "passages"]):
        for r in batch.to_pylist():
            texts = r["passages"]["Translated_passages"]
            sel = r["passages"]["is_selected"]
            gold = [t for t, s in zip(texts, sel) if s == 1 and t.strip()]
            if gold:
                rows.append({"query": r["query"], "qid": str(r["query_id"]), "gold": gold})
        if len(rows) >= n * 20:
            break
    rng = random.Random(seed)
    return rng.sample(rows, n)


def recall_at_k(hits, gold_texts: list[str], k: int) -> float:
    if not gold_texts:
        return 0.0
    toks = [set(g.lower().split()[:200]) for g in gold_texts]
    found = 0
    for h in hits[:k]:
        ht = set(h.text.lower().split()[:200])
        if any(len(t & ht) / max(1, len(t)) >= 0.6 for t in toks):
            found += 1
    return found / len(gold_texts)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--lang", default=None, help="force language (default: LID)")
    ap.add_argument("--val", default="data/parquet/hinval.parquet")
    args = ap.parse_args()

    embedder = Embedder()
    corpus = Corpus(Path("data/index"), embedder)
    print(f"languages: {corpus.languages()} | chunks: {sum(p.count for p in corpus.parts.values())}")
    pipe = Pipeline(corpus, embedder)

    rows = load_val(Path(args.val), args.n)
    print(f"bench: {len(rows)} queries, forced lang={args.lang or 'lid'}")

    lats, recalls, refused = [], [], 0
    t0 = time.perf_counter()
    pipe.run("मैनहट्टन परियोजना क्या है?", lang="hi")
    pipe.run("टेलीफोन का आविष्कार किसने किया?", lang="hi")
    print(f"warmup: {(time.perf_counter()-t0)*1000:.1f}ms")

    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        resp = pipe.run(r["query"], lang=args.lang)
        ms = (time.perf_counter() - t0) * 1000
        lats.append(ms)
        vec = embedder.encode_one(r["query"])
        hits = corpus.retrieve(vec, r["query"], args.lang or "hi", n=20)
        recalls.append(recall_at_k(hits, r["gold"], k=20))
        if resp.refused:
            refused += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(rows)}] {ms:.1f}ms refused={resp.refused}")

    lats.sort()
    p = lambda q: lats[min(len(lats) - 1, int(q * len(lats)))]
    print(f"\nT2 (transcript-in -> answer): n={len(lats)}")
    print(f"  P50: {p(0.50):.1f}ms  P70: {p(0.70):.1f}ms  P100: {p(1.00):.1f}ms")
    print(f"  mean: {statistics.fmean(lats):.1f}ms  min: {lats[0]:.1f}ms")
    print(f"  gold recall@citations: mean={statistics.fmean(recalls):.3f}  refused: {refused}/{len(rows)}")


if __name__ == "__main__":
    main()