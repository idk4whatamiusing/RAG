"""Smoke: index a small partition, run known queries, print hits + timings + P50."""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedder import Embedder  # noqa: E402
from src.pipeline import Corpus, Pipeline  # noqa: E402

QUERIES = [
    "what was the immediate impact of the success of the manhattan project?",
    "manhattan project immediate impact",
    "who invented the telephone",
    "states of india with their capitals",
]


def main():
    embedder = Embedder()
    corpus = Corpus(Path("data/index"), embedder)
    print("languages:", corpus.languages(), "| total chunks:",
          sum(corpus.parts[l].count for l in corpus.languages()))
    pipe = Pipeline(corpus, embedder)
    lats = []
    for q in QUERIES:
        t0 = time.perf_counter()
        resp = pipe.run(q)
        total = (time.perf_counter() - t0) * 1000
        lats.append(total)
        print(f"\nQ: {q}")
        print("  ->", resp.to_json())
        print(f"  total: {total:.1f}ms")
    lats.sort()
    n = len(lats)
    p50 = lats[n // 2]
    p100 = lats[-1]
    print(f"\nP50: {p50:.1f}ms  P100: {p100:.1f}ms  (n={n})")


if __name__ == "__main__":
    main()