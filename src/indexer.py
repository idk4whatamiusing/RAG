"""Index build: per-language FAISS (exact cosine, fp32) + BM25 + metadata parquet."""
from __future__ import annotations
import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.chunking import _tokenize
from src.embedder import Embedder


class Index:
    """One language partition: dense (faiss IP, normalized = cosine) + sparse (bm25) + metadata."""

    def __init__(self, lang: str, embedder: Embedder):
        self.lang = lang
        self.embedder = embedder
        self.texts: list[str] = []
        self.meta: list[dict] = []
        self.index: faiss.Index | None = None
        self._bk = None
        self._toks: list[list[str]] = []

    @property
    def count(self) -> int:
        return len(self.texts)

    def add(self, texts: list[str], meta: list[dict], vectors: np.ndarray):
        if not texts:
            return
        if self.index is None:
            self.index = faiss.IndexFlatIP(int(vectors.shape[1]))
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.texts.extend(texts)
        self.meta.extend(meta)
        self._toks.extend(_tokenize(t) for t in texts)
        self._bk = None

    def _bm(self) -> BM25Okapi:
        if self._bk is None:
            self._bk = BM25Okapi(self._toks)
        return self._bk

    def search_dense(self, query_vec: np.ndarray, k: int = 50) -> list[tuple[int, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        scores, idx = self.index.search(np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32),
                                        min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def search_sparse(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        if not self._toks:
            return []
        scores = self._bm().get_scores(_tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(int(j), float(scores[j])) for j in order if scores[j] > 0]

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(path / "faiss.index"))
        with open(path / "texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)
        pd.DataFrame(self.meta).to_parquet(path / "meta.parquet", index=False)
        (path / "lang.json").write_text(json.dumps({"lang": self.lang, "count": self.count}))

    @classmethod
    def load(cls, path: Path, embedder: Embedder):
        ix = cls(path.name, embedder)
        ix.lang = json.loads((path / "lang.json").read_text())["lang"]
        ix.texts = pickle.loads((path / "texts.pkl").read_bytes())
        ix.meta = pd.read_parquet(path / "meta.parquet").to_dict("records")
        ix._toks = [_tokenize(t) for t in ix.texts]
        f = path / "faiss.index"
        if f.exists():
            ix.index = faiss.read_index(str(f))
        return ix


def build_partition(lang: str, chunks: list[dict], embedder: Embedder) -> Index:
    t0 = time.time()
    ix = Index(lang, embedder)
    if not chunks:
        return ix
    ix.add([c["text"] for c in chunks], [c["meta"] for c in chunks],
           embedder.encode([c["text"] for c in chunks], batch=32))
    return ix