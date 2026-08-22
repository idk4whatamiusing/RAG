"""Harness: typed stages, timeouts, retries, guardrails G2-G5, structured output (issue GOA-5)."""
from __future__ import annotations
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from src.indexer import Index
from src.embedder import Embedder
from src.retrieval import retrieve, Hit

TAU_OFFTOPIC = float(os.environ.get("TAU_OFFTOPIC", "0.30"))
TAU_GROUND = float(os.environ.get("TAU_GROUND", "0.35"))
TAU_ANSWER = float(os.environ.get("TAU_ANSWER", "0.28"))
STAGE_TIMEOUT = float(os.environ.get("STAGE_TIMEOUT", "2.0"))
SYNTH_BUDGET_MS = float(os.environ.get("SYNTH_BUDGET_MS", "150"))  # groq cap off the critical path

TOXIC_TERMS = ("kill you", "fuck", "bhenchod", "madarchod", "randi", "chutiya", "suar", "harami",
               "tumhe mar", "tera baap", "abe chaman", "spoiler", "shoot someone", "bomb", "katl")


@dataclass
class StageResult:
    stage: str
    ok: bool
    ms: float
    detail: dict = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    confidence: float
    citations: list[dict] = field(default_factory=list)
    grounded: bool = True
    path: str = "extractive"


@dataclass
class StructuredResponse:
    answer: str = ""
    confidence: float = 0.0
    citations: list[dict] = field(default_factory=list)
    language: str = ""
    refused: bool = False
    reason: str = ""
    path: str = "extractive"
    stage_latencies: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _retry(fn, attempts: int = 3, base: float = 0.1, timeout: float = STAGE_TIMEOUT):
    last = None
    for i in range(attempts):
        t0 = time.perf_counter()
        try:
            return fn(), (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            last = e
            if e.code < 500 and e.code != 429:
                raise
            time.sleep(base * (2 ** i) + random.uniform(0, base))
        except Exception as e:
            last = e
            time.sleep(base * (2 ** i) + random.uniform(0, base))
    raise last


def run_stage(stage: str, fn, timeout: float = STAGE_TIMEOUT) -> StageResult:
    t0 = time.perf_counter()
    try:
        fn()
        return StageResult(stage, True, (time.perf_counter() - t0) * 1000)
    except Exception as e:
        return StageResult(stage, False, (time.perf_counter() - t0) * 1000,
                           {"error": f"{type(e).__name__}: {e}"})


class GuardG2:
    """Toxicity: fast blocklist on partials (abort), full check on final. Real onnx classifier later."""

    @staticmethod
    def check(text: str) -> tuple[bool, str]:
        low = text.lower()
        for w in TOXIC_TERMS:
            if w in low:
                return True, f"unsafe input flagged ({w})"
        return False, ""


class GuardG3:
    """Language detect -> route. fastText lid.176 if importable, else script heuristic."""

    # fastText 176-lang labels we index; everything Indic-not-mapped -> "indic",
    # everything else -> "eng_Latn" (falls back to largest partition).
    CORPUS = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "ta", "te", "ur"}

    @staticmethod
    def detect(text: str) -> str:
        try:
            lang, _ = GuardG3._model().predict(text.replace("\n", " ")[:1000])
            code = str(lang[0]).replace("__label__", "")
            if code in GuardG3.CORPUS:
                return code
            return "indic" if _has_indic(text) else "eng_Latn"
        except Exception:
            return "indic" if _has_indic(text) else "eng_Latn"

    _lid = None

    @classmethod
    def _model(cls):
        if cls._lid is None:
            import fasttext
            cls._lid = fasttext.load_model(_lid_path())
        return cls._lid


def _lid_path() -> str:
    return str(Path(os.environ.get("LID_MODEL", "models/lid.176.bin")))


def _has_indic(text: str) -> bool:
    return any(0x0900 <= ord(c) <= 0x0DFF for c in text[:500])


class Corpus:
    def __init__(self, index_dir: Path, embedder: Embedder):
        self.parts: dict[str, Index] = {}
        for d in index_dir.iterdir():
            if (d / "lang.json").exists():
                self.parts[d.name] = Index.load(d, embedder)
        self.order = sorted(self.parts)

    def languages(self) -> list[str]:
        return list(self.order)

    def retrieve(self, query_vec: np.ndarray, query_text: str, lang: str, n: int = 20) -> list[Hit]:
        if lang in self.parts:
            return retrieve(self.parts[lang], query_vec, query_text, n=n)
        best = max(self.parts.values(), key=lambda p: p.count)
        return retrieve(best, query_vec, query_text, n=n)


def _extract_answer(query: str, hits: list[Hit]) -> tuple[str, float]:
    import re
    q_toks = set(re.findall(r"\w+", query.lower()))
    best, best_score = "", 0.0
    for h in hits[:5]:
        for s in re.split(r"(?<=[.!?।॥…])", h.text):
            s = s.strip()
            if len(s) < 12 or len(s) > 220:
                continue
            overlap = len(q_toks & set(re.findall(r"\w+", s.lower())))
            score = overlap / max(1, len(q_toks)) - 0.05 * len(s) / 220
            if score > best_score:
                best, best_score = s, score
    return best, best_score


def _groq_answer(query: str, hits: list[Hit]) -> tuple[str, bool]:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("no GROQ_API_KEY")
    ctx = "\n\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits[:6]))
    prompt = ("Answer ONLY from the context passages below, in the same language as the question. "
              "If the answer is not in the context, say 'NOT_IN_CONTEXT'. Return JSON {\"answer\": \"...\"}.\n\n"
              f"Context:\n{ctx}\n\nQuestion: {query}")
    body = json.dumps({
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 120,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    })
    req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",
                                 data=body.encode(), method="POST")
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=STAGE_TIMEOUT) as resp:
        data = json.load(resp)
    out = json.loads(data["choices"][0]["message"]["content"])
    ans = out.get("answer", "")
    grounded = "NOT_IN_CONTEXT" not in ans
    return ans, grounded


class Pipeline:
    def __init__(self, corpus: Corpus, embedder: Embedder):
        self.corpus = corpus
        self.embedder = embedder
        self.timings: dict[str, float] = {}
        self._gq = ThreadPoolExecutor(max_workers=1)

    def _time(self, name: str, fn):
        t0 = time.perf_counter()
        out = fn()
        self.timings[name] = (time.perf_counter() - t0) * 1000
        return out

    def run(self, query: str, lang: str | None = None) -> StructuredResponse:
        self.timings = {}
        resp = StructuredResponse(language=lang or "")

        if not query.strip():
            return StructuredResponse(refused=True, reason="no-speech", language=lang or "")

        if lang is None:
            lang = self._time("g3_lid", lambda: GuardG3.detect(query))
            resp.language = lang
        if lang not in self.corpus.parts and lang not in ("indic", "eng_Latn"):
            return StructuredResponse(refused=True, reason=f"language-not-supported: {lang}",
                                      language=lang)

        toxic, why = GuardG2.check(query)
        if toxic:
            return StructuredResponse(refused=True, reason=why, language=lang)

        vec = self._time("embed", lambda: self.embedder.encode_one(query))
        hits = self._time("retrieve", lambda: self.corpus.retrieve(vec, query, lang))
        if not hits:
            return StructuredResponse(refused=True, reason="empty-retrieval", language=lang)

        top = hits[0]
        if top.sim < TAU_OFFTOPIC:
            return StructuredResponse(refused=True, reason="off-topic", confidence=top.sim,
                                      language=lang, citations=[_cit(top)])

        ans, conf = self._time("extract", lambda: _extract_answer(query, hits[:5]))
        resp.answer, resp.confidence, resp.citations = ans, conf, [_cit(h) for h in hits[:3]]
        resp.path = "extractive"
        resp.stage_latencies = dict(self.timings)

        if conf < TAU_ANSWER:
            fut = self._gq.submit(lambda: _groq_answer(query, hits))
            try:
                synth, grounded = fut.result(timeout=SYNTH_BUDGET_MS / 1000)
                if grounded and _grounded_check(synth, hits):
                    resp.answer, resp.path, resp.confidence = synth, "synthesis", max(conf, 0.4)
            except Exception:
                fut.cancel()
                if conf <= 0.0:
                    return StructuredResponse(refused=True, reason="not-in-knowledge-base",
                                              language=lang, stage_latencies=dict(self.timings))
        if not resp.answer:
            return StructuredResponse(refused=True, reason="not-in-knowledge-base", language=lang,
                                      stage_latencies=dict(self.timings))
        resp.stage_latencies = dict(self.timings)
        return resp


def _cit(h: Hit) -> dict:
    return {"lang": h.lang, "qid": h.meta.get("qid"), "passage_idx": h.meta.get("passage_idx"),
            "selected": int(h.meta.get("selected", 0)), "chunk_id": h.meta.get("chunk_id"),
            "snippet": (h.text[:240] + ("…" if len(h.text) > 240 else ""))}


def _grounded_check(text: str, hits: list[Hit], tau: float = TAU_GROUND) -> bool:
    import re
    toks = set(re.findall(r"\w+", text.lower()))
    if len(toks) < 3:
        return True
    for h in hits:
        overlap = len(toks & set(re.findall(r"\w+", h.text.lower())))
        if overlap / len(toks) >= tau:
            return True
    return len(hits) == 0