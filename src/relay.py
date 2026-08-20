"""WS relay: browser <-> ElevenLabs Scribe realtime, dumb byte pipe (GOA-8 relay)."""
from __future__ import annotations
import asyncio
import logging
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets

log = logging.getLogger("uvicorn.error")

UPSTREAM = os.environ.get(
    "SCRIBE_WS",
    "wss://api.elevenlabs.io/v1/speech-to-text/realtime",
)
QUERY = os.environ.get(
    "SCRIBE_QUERY",
    "model_id=scribe_v2_realtime&commit_strategy=manual&include_language_detection=true",
)
KEY = os.environ.get("ELEVENLABS_API_KEY", "")

app = FastAPI()


def _upstream_url() -> str:
    sep = "&" if "?" in UPSTREAM else "?"
    return f"{UPSTREAM}{sep}{QUERY}"


async def _relay(client: WebSocket, up, stats: dict):
    async def client_to_up():
        while True:
            msg = await client.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes") or msg.get("text")
            if data:
                stats["up"] += len(data)
            await up.send(data)

    async def up_to_client():
        while True:
            msg = await up.recv()
            stats["down"] += len(msg)
            if isinstance(msg, bytes):
                await client.send_bytes(msg)
            else:
                await client.send_text(msg)

    done, pending = await asyncio.wait(
        [asyncio.create_task(client_to_up()), asyncio.create_task(up_to_client())],
        return_when=asyncio.FIRST_EXCEPTION,
    )
    for t in pending:
        t.cancel()
    for t in done:
        t.result()


@app.websocket("/ws")
async def ws_proxy(client: WebSocket):
    await client.accept()
    if not KEY:
        await client.close(code=1011, reason="missing ELEVENLABS_API_KEY")
        return

    stats = {"up": 0, "down": 0}
    # ponytail: retry the Scribe leg 3x on transient closes (Scribe 1000s idle
    # sessions); only give up — surfacing the real reason — if it keeps dying
    try:
        for attempt in range(4):
            try:
                async with websockets.connect(
                    _upstream_url(),
                    additional_headers={"xi-api-key": KEY},
                    ping_interval=20,
                    ping_timeout=20,
                ) as up:
                    await _relay(client, up, stats)
                return
            except websockets.ConnectionClosed as e:
                rcvd = getattr(getattr(e, "rcvd", None), "code", None)
                rsn = getattr(getattr(e, "rcvd", None), "reason", "")
                log.warning("scribe close: code=%s reason=%r up=%dB down=%dB attempt=%d",
                            rcvd, rsn, stats["up"], stats["down"], attempt)
                if attempt == 3:
                    raise
                await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error("relay error: %s up=%dB down=%dB", e, stats["up"], stats["down"])
        await client.close(code=1011, reason=f"upstream error: {type(e).__name__}: {e}")