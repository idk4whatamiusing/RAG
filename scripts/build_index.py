"""Build the index. --smoke: single language, few queries. default: full 15-language subsample."""
from __future__ import annotations
import argparse
import os
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402

from src.chunking import chunk_passage, _strategy_names  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.indexer import Index, build_partition  # noqa: E402

LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur", "hi2"]
LANG_FILES = {"as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal",
              "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "ta": "tam",
              "te": "tel", "ur": "urd"}
QUERY_CAP = {"hi": 778638}  # full file for hi; others stay default
DEFAULT_CAP = 30000


def iter_rows(lang: str, split: str = "train", local: bool = False):
    split_dir = "train" if split == "train" else "validation"
    fname = f"{split_dir}/{LANG_FILES[lang]}{'train' if split == 'train' else 'val'}.parquet"
    if not local:
        from datasets import load_dataset  # lazy: not needed for --local
        ds = load_dataset("ai4bharat/MSMARCO-XI", data_files=fname, split="train", streaming=True)
        return iter(ds)
    path = Path(os.environ.get("PARQUET_DIR", "data/parquet")) / fname.split("/")[-1]
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=1024, columns=["query", "query_id", "query_type", "passages"]):
        for row in batch.to_pylist():
            yield row


def collect(lang: str, cap: int, strategy: str, split: str = "train", selected_only: bool = True, local: bool = False, batch: int = 25000):
    buffer: list[dict] = []
    qcount = 0
    total = 0
    seen: set[str] = set()
    t_last = time.time()
    for row in iter_rows(lang, split, local):
        if qcount and qcount % 50_000 == 0:
            print(f"  collect: {qcount} queries, {total:,} chunks, "
                  f"{len(seen):,} unique, {(time.time()-t_last):.0f}s/50k", flush=True)
            t_last = time.time()
        sel = [s for s in row["passages"]["is_selected"]]
        texts = row["passages"]["Translated_passages"]
        if selected_only:
            for i, (s, t) in enumerate(zip(sel, texts)):
                if s == 1 and t.strip() and t not in seen:
                    seen.add(t)
                    buffer.extend(chunk_passage(str(i), t, strategy, lang=lang,
                                                qid=str(row["query_id"]), selected=1,
                                                qtype=row.get("query_type", "")))
        else:
            for i, t in enumerate(texts):
                if t.strip() and t not in seen:
                    seen.add(t)
                    buffer.extend(chunk_passage(str(i), t, strategy, lang=lang,
                                                qid=str(row["query_id"]),
                                                selected=int(sel[i]) if i < len(sel) else 0,
                                                qtype=row.get("query_type", "")))
        qcount += 1
        if qcount >= cap or total >= 600_000:
            break
        if len(buffer) >= batch:
            total += len(buffer)
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one language, few queries")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--strategy", choices=_strategy_names, default="native")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--all", action="store_true", help="build all 14 language partitions")
    ap.add_argument("--local", action="store_true", help="read from data/parquet instead of HF streaming")
    args = ap.parse_args()

    embedder = Embedder()
    out = Path(args.out)

    if args.smoke:
        chunks = [c for b in collect(args.lang, args.queries, args.strategy, local=args.local) for c in b]
        print(f"{args.lang}: {len(chunks)} chunks from {args.queries} queries")
        ix = build_partition(args.lang, chunks, embedder)
        ix.save(out / args.lang)
        print("saved:", out / args.lang)
        return

    langs = list(LANG_FILES) if args.all else [args.lang]
    for lang in langs:
        cap = QUERY_CAP.get(lang, DEFAULT_CAP)
        t0 = time.time()
        ix = Index(lang, embedder)
        total = 0
        for batch in collect(lang, cap, args.strategy, local=args.local):
            total += len(batch)
            ix.add([c["text"] for c in batch], [c["meta"] for c in batch],
                   embedder.encode([c["text"] for c in batch]), index_toks=False)
            print(f"{lang}: {total} chunks embedded, {time.time()-t0:.0f}s", flush=True)
        if total == 0:
            print(f"{lang}: 0 chunks, skip")
            continue
        t2 = time.time()
        ix.save(out / lang)
        print(f"{lang}: saved in {time.time()-t2:.1f}s ({out / lang})")
        print(f"{lang}: {total} chunks total {time.time()-t0:.0f}s")
    print("done")


if __name__ == "__main__":
    main()