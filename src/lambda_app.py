"""Lambda (Function URL) + local uvicorn: POST /query -> Pipeline.run (GOA-7)."""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
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
    resp = _pipe.run(text, lang=body.get("lang"))
    out = json.loads(resp.to_json())
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return JSONResponse(out)


# --- TTS (GOA-16): ElevenLabs flash streaming proxy with LRU cache ---

TTS_MODEL = os.environ.get("ELEVEN_TTS_MODEL", "eleven_flash_v2_5")
VOICE_ID = os.environ.get("ELEVEN_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
_TTS_CACHE: "OrderedDict[str, bytes]" = OrderedDict()
_TTS_CACHE_MAX = int(os.environ.get("TTS_CACHE_MAX", "32"))


def _tts_key(text: str) -> str:
    return hashlib.sha1(f"{VOICE_ID}|{TTS_MODEL}|{text}".encode()).hexdigest()


def _tts_open(text: str):
    """Open the ElevenLabs stream; raises (401/upstream) before any byte flows."""
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


# Free fallback voice (no key): Microsoft Edge neural TTS over websockets.
EDGE_VOICES: dict[str, str] = {
    "as": "hi-IN-MadhurNeural", "bn": "bn-IN-TanishaaNeural", "gu": "gu-IN-NiranjanNeural",
    "hi": "hi-IN-MadhurNeural", "kn": "kn-IN-GaganNeural", "ml": "ml-IN-SobhanaNeural",
    "mr": "mr-IN-AarohiNeural", "ne": "ne-NP-SagarNeural", "or": "or-IN-SubhasiniNeural",
    "pa": "pa-IN-?Neural", "sa": "hi-IN-MadhurNeural", "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural", "ur": "ur-IN-SalmanNeural", "en": "en-IN-PrabhatNeural",
    "eng_Latn": "en-IN-PrabhatNeural", "indic": "hi-IN-MadhurNeural",
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

    ok_text = text
    key = _tts_key(ok_text)
    hit = _TTS_CACHE.get(key)
    if hit is not None:
        _TTS_CACHE.move_to_end(key)
        return Response(content=hit, media_type="audio/mpeg", headers={"X-Cache": "hit"})

    async def gen():
        buf = bytearray()
        complete = False
        up = None
        try:
            # primary: ElevenLabs (needs account credits); fallback: Edge neural (free)
            try:
                up = await asyncio.to_thread(_tts_open, ok_text)
            except Exception:
                up = None
            if up is not None:
                while True:
                    chunk = await asyncio.to_thread(up.read, 4096)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    yield chunk
                complete = True
            else:
                async for chunk in _edge_tts(ok_text, lang):
                    buf.extend(chunk)
                    yield chunk
                complete = True
        except Exception:
            complete = False
        finally:
            if up is not None:
                try:
                    up.close()
                except Exception:
                    pass
            if complete and len(buf) > 512:
                _TTS_CACHE[key] = bytes(buf)
                _TTS_CACHE.move_to_end(key)
                while len(_TTS_CACHE) > _TTS_CACHE_MAX:
                    _TTS_CACHE.popitem(last=False)

    return StreamingResponse(gen(), media_type="audio/mpeg",
                             headers={"Cache-Control": "no-store", "X-Cache": "miss"})


from mangum import Mangum  # noqa: E402
handler = Mangum(app, lifespan="off")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
