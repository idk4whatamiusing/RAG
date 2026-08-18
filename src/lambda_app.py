"""Lambda (Function URL) + local uvicorn: POST /query -> Pipeline.run (GOA-7)."""
from __future__ import annotations
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import boto3

from src.embedder import Embedder
from src.pipeline import Corpus, Pipeline

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
    if _LOCAL.exists() and any(_LOCAL.iterdir()):
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


from mangum import Mangum  # noqa: E402
handler = Mangum(app, lifespan="off")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
