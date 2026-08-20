"""Bench (GOA-6): T2 latency + gold recall on validation queries (default 75, Hindi).
--rds: run against the live pgvector corpus (box only; downloads val parquets from HF)."""
from __future__ import annotations
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from src.embedder import Embedder  # noqa: E402
from src.pipeline import Corpus, Pipeline  # noqa: E402

LANG_FILES = {"as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal",
              "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "ta": "tam", "ur": "urd"}
HF_BASE = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/validation/{code}val.parquet"


def _val_path(lang: str, cache: str) -> Path:
    path = Path(cache) / f"{LANG_FILES[lang]}val.parquet"
    if not path.exists():
        print(f"  downloading {path.name} from HF ...", flush=True)
        urllib.request.urlretrieve(HF_BASE.format(code=LANG_FILES[lang]), path)
    return path


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
    ap.add_argument("--rds", action="store_true", help="bench against live pgvector corpus")
    ap.add_argument("--langs", default="hi,bn,mr,ta,ur", help="languages for --rds (comma sep)")
    ap.add_argument("--val-cache", default="data/parquet")
    args = ap.parse_args()

    embedder = Embedder()

    if args.rds:
        from src.sqlstore import rds_connection_factory, SqlCorpus
        corpus = SqlCorpus(rds_connection_factory())
    else:
        corpus = Corpus(Path("data/index"), embedder)
    print(f"languages: {corpus.languages()} | chunks: {sum(p.count for p in corpus.parts.values())}")
    pipe = Pipeline(corpus, embedder)

    if args.rds:
        langs = [l for l in args.langs.split(",") if l]
        per_lang: dict[str, dict] = {}
        for lang in langs:
            path = _val_path(lang, args.val_cache)
            rows = load_val(path, args.n)
            lats, recalls, refused = [], [], 0
            for i, r in enumerate(rows):
                t0 = time.perf_counter()
                resp = pipe.run(r["query"], lang=lang)
                ms = (time.perf_counter() - t0) * 1000
                lats.append(ms)
                vec = embedder.encode_one(r["query"])
                hits = corpus.retrieve(vec, r["query"], lang, n=20)
                recalls.append(recall_at_k(hits, r["gold"], k=20))
                if resp.refused:
                    refused += 1
                if i % 50 == 0:
                    print(f"  [{lang} {i}/{len(rows)}] {ms:.1f}ms refused={resp.refused}", flush=True)
            lats.sort()
            p = lambda q: lats[min(len(lats) - 1, int(q * len(lats)))]
            per_lang[lang] = {"n": len(lats), "P50": p(0.50), "P70": p(0.70), "P100": p(1.00),
                              "mean": statistics.fmean(lats), "recall": statistics.fmean(recalls),
                              "refused": refused}
            print(f"{lang}: P50={p(0.50):.1f} P70={p(0.70):.1f} P100={p(1.00):.1f} "
                  f"mean={statistics.fmean(lats):.1f} recall@20={statistics.fmean(recalls):.3f} "
                  f"refused={refused}/{len(lats)}", flush=True)
        print("\n=== --rds summary ===")
        for lang, d in per_lang.items():
            print(f"  {lang:2s} P50={d['P50']:.1f} P70={d['P70']:.1f} P100={d['P100']:.1f} "
                  f"mean={d['mean']:.1f} recall@20={d['recall']:.3f} refused={d['refused']}/{d['n']}")
        return

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