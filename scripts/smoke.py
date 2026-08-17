"""Smoke: run Hindi queries through the full pipeline; print latency percentiles."""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedder import Embedder  # noqa: E402
from src.pipeline import Corpus, Pipeline  # noqa: E402

QUERIES = [
    "मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?",
    "टेलीफोन का आविष्कार किसने किया?",
    "भारत के राज्य और उनकी राजधानियाँ",
    "द्वितीय विश्व युद्ध कब समाप्त हुआ?",
    "पानी का रासायनिक सूत्र क्या है?",
    "सौरमंडल का सबसे बड़ा ग्रह कौन सा है?",
]


def main():
    embedder = Embedder()
    corpus = Corpus(Path("data/index"), embedder)
    langs = corpus.languages()
    print("languages:", langs, "| chunks:",
          sum(corpus.parts[l].count for l in langs))
    pipe = Pipeline(corpus, embedder)
    lats = []
    for q in QUERIES:
        t0 = time.perf_counter()
        resp = pipe.run(q)
        total = (time.perf_counter() - t0) * 1000
        lats.append(total)
        print(f"\nQ: {q}")
        print(f"  -> {resp.to_json()}")
        print(f"  total: {total:.1f}ms")
    lats.sort()
    n = len(lats)
    print(f"\nP50: {lats[n // 2]:.1f}ms  P100: {lats[-1]:.1f}ms  (n={n})")


if __name__ == "__main__":
    main()