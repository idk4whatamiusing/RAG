"""Chunking strategies (issue GOA-3). All four built; winner chosen by eval (GOA-4)."""
from __future__ import annotations
import re

FIXED_TOK = 128
FIXED_OVERLAP = 16
LONG_PASSAGE_TOK = 200

_strategy_names = ("native", "fixed", "semantic", "metadata")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?।॥…])\s+", text.strip())
    return [p for p in parts if p]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", text.lower())


def chunk_passage(pid: str, text: str, strategy: str,
                  lang: str = "", qid: str = "", selected: int = 0,
                  qtype: str = "") -> list[dict]:
    """Return chunk records: {chunk_id, text, strategy, meta{lang,qid,passage_idx,selected,query_type,chunk_id}}."""
    if strategy == "native":
        parts = [text]
    elif strategy == "fixed":
        toks = _tokenize(text)
        if len(toks) <= LONG_PASSAGE_TOK:
            parts = [text]
        else:
            parts = [" ".join(toks[i:i + FIXED_TOK])
                     for i in range(0, len(toks), FIXED_TOK - FIXED_OVERLAP)]
    elif strategy == "semantic":
        parts = _sentences(text)
        # merge short consecutive sentences, split on big gaps
        merged, buf = [], ""
        for s in parts:
            if not buf:
                buf = s
            elif len(buf) + len(s) < 220:
                buf += " " + s
            else:
                merged.append(buf)
                buf = s
        if buf:
            merged.append(buf)
        parts = merged or [text]
    elif strategy == "metadata":
        parts = [text]
    else:
        raise ValueError(f"unknown strategy {strategy}")

    out = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        cid = f"{lang}:{qid}:{pid}:{i}"
        out.append({
            "chunk_id": cid,
            "text": part,
            "strategy": strategy,
            "meta": {
                "lang": lang, "qid": qid, "passage_idx": pid, "chunk_id": i,
                "selected": int(selected), "query_type": qtype, "strategy": strategy,
            },
        })
    return out