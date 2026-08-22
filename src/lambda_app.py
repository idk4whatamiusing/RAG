"""Lambda (Function URL) + local uvicorn: POST /query -> Pipeline.run (GOA-7)."""
from __future__ import annotations
import asyncio
import base64
import hashlib
import json
import os
import random
import tempfile
import time
import urllib.request as _ur
import zipfile
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import boto3

from src.embedder import Embedder
from src.pipeline import Corpus, Pipeline
from src.sqlstore import SqlCorpus, rds_connection_factory

BUCKET = os.environ.get("INDEX_BUCKET", "voice-rag-index")
PREFIX = os.environ.get("INDEX_PREFIX", "index")
_LOCAL = Path(os.environ.get("INDEX_DIR", "data/index"))

app = FastAPI()
_embedder = _corpus = _pipe = None
_s3 = None


def _load():
    global _embedder, _corpus, _pipe
    if _pipe is not None:
        return
    t0 = time.perf_counter()
    _embedder = Embedder()
    if os.environ.get("INDEX_MODE") == "rds":
        _corpus = SqlCorpus(rds_connection_factory())
    elif _LOCAL.exists() and any(_LOCAL.iterdir()):
        _corpus = Corpus(_LOCAL, _embedder)
    else:
        dest = Path(tempfile.mkdtemp(prefix="ragidx-"))
        _s3 = boto3.client("s3", region_name="ap-south-1")
        objs = _s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/")["Contents"]
        for o in objs:
            key = o["Key"]
            rel = key[len(PREFIX) + 1:]
            if rel.endswith("/"):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            _s3.download_file(BUCKET, key, str(out))
        _corpus = Corpus(dest, _embedder)
    _pipe = Pipeline(_corpus, _embedder)
    print(f"init {time.perf_counter()-t0:.1f}s langs={_corpus.languages()}")


@app.get("/health")
async def health():
    _load()
    return {"ok": True, "langs": list(_corpus.languages()) if _corpus else []}


@app.post("/query")
async def query(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    text = (body.get("query") or "").strip()
    if not text:
        return JSONResponse({"error": "empty query"}, status_code=400)
    t0 = time.perf_counter()
    _load()
    lang = body.get("lang")
    kb = body.get("kb")  # GOA-17 G1: "goa" | "docs" brains force their partition
    if kb in ("goa", "docs"):
        lang = kb
    resp = _pipe.run(text, lang=lang, history=body.get("history"))
    out = json.loads(resp.to_json())
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return JSONResponse(out)


# --- /bench (GOA-17 L3): live evaluation on stage ---

_BENCH_FALLBACK = [
    "मैनहट्टन परियोजना क्या है?", "टेलीफोन का आविष्कार किसने किया?", "आइंस्टीन को नोबेल पुरस्कार कब मिला?",
    "कंप्यूटर किसने बनाया?", "पहली विश्व युद्ध कब हुआ?", "चंद्रमा पर पहला कदम किसने रखा?",
    "पृथ्वी का सबसे बड़ा महासागर कौन सा है?", "प्रकाश की गति कितनी है?", "जल का रासायनिक सूत्र क्या है?",
    "डीएनए की खोज किसने की?", "वायुमंडल का सबसे पुराना स्तर कौन सा है?", "बिजली कैसे बनती है?",
    "सूर्य का तापमान कितना है?", "ग्रांट कैन्यन कहाँ है?", "ताजमहल किसने बनवाया?",
    "मंगल ग्रह पर कितने मून्स हैं?", "समुद्र का खारापन क्यों है?", "क्यूरेटिव साइंस क्या है?",
    "पेनेसिलिन की खोज किसने की?", "वेटिंग रूम में लोग क्यों इंतज़ार करते हैं?", "सूरज की ऊर्जा कहाँ से आती है?",
    "कार्बन डाइऑक्साइड क्या है?", "विश्व की सबसे ऊँची चोटी कौन सी है?", "भारत ने आज़ादी कब पाई?",
    "मोबाइल फोन का आविष्कार किसने किया?",
]

_bench_qs: list[str] | None = None


def _bench_queries() -> list[str]:
    global _bench_qs
    if _bench_qs is not None:
        return _bench_qs
    out: list[str] = []
    path = Path("data_eval/hinval.parquet")
    if path.exists():
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512, columns=["query"]):
            for row in batch.to_pylist():
                q = (row["query"] or "").strip()
                if q:
                    out.append(q)
            if len(out) >= 2000:
                break
        _bench_qs = random.Random(42).sample(out, 25)
    else:
        _bench_qs = list(_BENCH_FALLBACK)
    return _bench_qs


@app.post("/bench")
async def bench():
    _load()
    qs = _bench_queries()
    lats: list[float] = []
    rows = []
    for q in qs:
        t0 = time.perf_counter()
        r = _pipe.run(q, lang="hi")
        ms = (time.perf_counter() - t0) * 1000
        lats.append(ms)
        rows.append({"q": q[:80], "ms": round(ms, 1), "refused": r.refused})
    s = sorted(lats)
    n = len(s)
    p = lambda q_: s[min(n - 1, int(q_ * n))]
    return JSONResponse({
        "n": n, "P50": round(p(0.5), 1), "P70": round(p(0.7), 1), "P100": round(p(1.0), 1),
        "mean": round(sum(lats) / n, 1), "queries": rows})


# --- TTS (GOA-16/17): tiered chain — ElevenLabs (paid) -> Sarvam AI (Indic-native) -> Edge (free) ---

TTS_MODEL = os.environ.get("ELEVEN_TTS_MODEL", "eleven_flash_v2_5")
VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
SARVAM_KEY = os.environ.get("SARVAM_API_KEY", "")
SARVAM_SPEAKER = os.environ.get("SARVAM_SPEAKER", "shubh")
_TTS_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_TTS_CACHE_MAX = int(os.environ.get("TTS_CACHE_MAX", "32"))

# Sarvam TTS covers 11 BCP-47 Indic codes; as/ur/sa/ne fall through to Edge.
SARVAM_BCP: dict[str, str] = {
    "bn": "bn-IN", "en": "en-IN", "eng_Latn": "en-IN", "gu": "gu-IN", "hi": "hi-IN",
    "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "or": "od-IN", "pa": "pa-IN",
    "ta": "ta-IN", "te": "te-IN", "indic": "hi-IN", "goa": "hi-IN",
}


def _tts_key(text: str) -> str:
    return hashlib.sha1(f"{VOICE_ID}|{TTS_MODEL}|{text}".encode()).hexdigest()


def _tts_open(text: str):
    """Open the ElevenLabs stream; raises (401/402/upstream) before any byte flows."""
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        raise RuntimeError("missing ELEVENLABS_API_KEY")
    body = json.dumps({"text": text, "model_id": TTS_MODEL}).encode()
    req = _ur.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
        "?output_format=mp3_44100_128",
        data=body, method="POST")
    req.add_header("xi-api-key", key)
    req.add_header("Content-Type", "application/json")
    return _ur.urlopen(req, timeout=20)


def _sarvam_synth(text: str, lang: str) -> bytes:
    """Sarvam AI bulbul:v3 (Indic-native voices) — fires when ElevenLabs is burned/exhausted."""
    code = SARVAM_BCP.get(lang)
    if not code or not SARVAM_KEY:
        raise LookupError(f"no sarvam path for {lang}")
    body = json.dumps({"text": text, "language_code": code, "model": "bulbul:v3",
                       "output_audio_codec": "mp3", "speaker": SARVAM_SPEAKER,
                       "pace": 1, "temperature": 0.6}).encode()
    req = _ur.Request("https://api.sarvam.ai/text-to-speech", data=body, method="POST")
    req.add_header("api-subscription-key", SARVAM_KEY)
    req.add_header("Content-Type", "application/json")
    r = _ur.urlopen(req, timeout=30)
    data = json.loads(r.read())
    b64 = data["audios"][0]
    return base64.b64decode(b64)


# Free fallback voice (no key): Microsoft Edge neural TTS over websockets.
EDGE_VOICES: dict[str, str] = {
    "as": "hi-IN-MadhurNeural", "bn": "bn-IN-TanishaaNeural", "gu": "gu-IN-NiranjanNeural",
    "hi": "hi-IN-MadhurNeural", "kn": "kn-IN-GaganNeural", "ml": "ml-IN-SobhanaNeural",
    "mr": "mr-IN-AarohiNeural", "ne": "ne-NP-SagarNeural", "or": "or-IN-SubhasiniNeural",
    "pa": "pa-IN-?Neural", "sa": "hi-IN-MadhurNeural", "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural", "ur": "ur-IN-SalmanNeural", "en": "en-IN-PrabhatNeural",
    "eng_Latn": "en-IN-PrabhatNeural", "indic": "hi-IN-MadhurNeural",
    "goa": "hi-IN-MadhurNeural", "docs": "en-IN-PrabhatNeural",
}
EDGE_DEFAULT = "en-IN-PrabhatNeural"


def _edge_voice(lang: str) -> str | None:
    v = EDGE_VOICES.get(lang, "")
    if "?" in v:
        return None
    return v or EDGE_DEFAULT


async def _edge_tts(text: str, lang: str):
    """Yield MP3 chunks from the free Edge neural TTS (no key needed)."""
    import edge_tts
    voice = _edge_voice(lang)
    if not voice:
        raise LookupError(f"no edge voice for {lang}")
    tts = edge_tts.Communicate(text, voice)
    async for chunk in tts.stream():
        if chunk["type"] == "audio" and chunk.get("data"):
            yield chunk["data"]


def _cache_put(key: str, audio: bytes, source: str, mime: str = "audio/mpeg"):
    if len(audio) <= 512:
        return
    _TTS_CACHE[key] = {"audio": audio, "source": source, "mime": mime}
    _TTS_CACHE.move_to_end(key)
    while len(_TTS_CACHE) > _TTS_CACHE_MAX:
        _TTS_CACHE.popitem(last=False)


@app.post("/tts")
async def tts(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse({"error": "invalid json"}, status_code=400)
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    text = text[:600]
    lang = body.get("lang") or "hi"

    key = _tts_key(text)
    hit = _TTS_CACHE.get(key)
    if hit is not None:
        _TTS_CACHE.move_to_end(key)
        return Response(content=hit["audio"], media_type=hit["mime"],
                        headers={"X-Cache": "hit", "X-TTS-Source": hit["source"]})

    # Tier 1: ElevenLabs (premium when credits exist)
    try:
        up = await asyncio.to_thread(_tts_open, text)

        async def eleven_gen():
            buf = bytearray()
            complete = False
            try:
                while True:
                    chunk = await asyncio.to_thread(up.read, 4096)
                    if not chunk:
                        complete = True
                        break
                    buf.extend(chunk)
                    yield chunk
            finally:
                up.close()
                if complete:
                    _cache_put(key, bytes(buf), "eleven")

        return StreamingResponse(eleven_gen(), media_type="audio/mpeg",
                                 headers={"Cache-Control": "no-store", "X-Cache": "miss",
                                          "X-TTS-Source": "eleven"})
    except Exception:
        pass  # burned/exhausted -> next tier

    # Tier 2: Sarvam AI (Indic-native, fired when ElevenLabs is exhausted)
    try:
        audio = await asyncio.to_thread(_sarvam_synth, text, lang)
        if audio:
            _cache_put(key, audio, "sarvam")
            return Response(content=audio, media_type="audio/mpeg",
                            headers={"Cache-Control": "no-store", "X-Cache": "miss",
                                     "X-TTS-Source": "sarvam"})
    except Exception:
        pass  # missing key / unsupported lang / upstream -> final tier

    # Tier 3: Edge neural (free, universal)
    async def edge_gen():
        buf = bytearray()
        complete = False
        try:
            async for chunk in _edge_tts(text, lang):
                buf.extend(chunk)
                yield chunk
            complete = True
        finally:
            if complete:
                _cache_put(key, bytes(buf), "edge")

    return StreamingResponse(edge_gen(), media_type="audio/mpeg",
                             headers={"Cache-Control": "no-store", "X-Cache": "miss",
                                      "X-TTS-Source": "edge"})


from mangum import Mangum  # noqa: E402
handler = Mangum(app, lifespan="off")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
