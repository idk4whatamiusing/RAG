"""WS relay: browser <-> ElevenLabs Scribe realtime, dumb byte pipe (GOA-8 relay)."""
from __future__ import annotations
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import websockets

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


@app.websocket("/ws")
async def ws_proxy(client: WebSocket):
    await client.accept()
    if not KEY:
        await client.close(code=1011, reason="missing ELEVENLABS_API_KEY")
        return
    try:
        async with websockets.connect(_upstream_url(), additional_headers={"xi-api-key": KEY},
                                      ping_interval=20, ping_timeout=20) as up:
            async def client_to_up():
                while True:
                    msg = await client.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    data = msg.get("bytes") or msg.get("text")
                    await up.send(data)
                    if msg["type"] == "websocket.disconnect":
                        break

            async def up_to_client():
                while True:
                    msg = await up.recv()
                    await client.send_bytes(msg if isinstance(msg, bytes) else msg.encode())

            import asyncio
            done, pending = await asyncio.wait(
                [asyncio.create_task(client_to_up()), asyncio.create_task(up_to_client())],
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for t in pending:
                t.cancel()
            for t in done:
                t.result()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await client.close(code=1011, reason=f"upstream error: {type(e).__name__}")
