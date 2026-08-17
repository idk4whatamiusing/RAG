"""Hybrid retrieval: dense + sparse, RRF fusion (k=60), is_selected boost, language routing."""
from __future__ import annotations
from dataclasses import dataclass, field

RRF_K = 60
SELECTED_BOOST = 0.15


@dataclass
class Hit:
    doc: int
    lang: str
    text: str
    meta: dict
    score: float
    selected: bool = False


def rrf_fuse(*rankings: list[tuple[int, float]], n: int = 20) -> list[tuple[int, float]]:
    agg: dict[int, float] = {}
    for rk in rankings:
        for rank, (doc, _s) in enumerate(rk):
            agg[doc] = agg.get(doc, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(agg.items(), key=lambda kv: -kv[1])[:n]


def retrieve(index, query_vec, query_text: str, n: int = 20) -> list[Hit]:
    dense = index.search_dense(query_vec, k=50)
    sparse = index.search_sparse(query_text, k=50)
    fused = rrf_fuse(dense, sparse, n=n)
    out = []
    for doc, score in fused:
        meta = index.meta[doc]
        sel = int(meta.get("selected", 0)) == 1
        if sel:
            score += SELECTED_BOOST
        out.append(Hit(doc=doc, lang=index.lang, text=index.texts[doc], meta=meta,
                       score=score, selected=sel))
    return out