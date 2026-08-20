import { useEffect, useRef, useState } from "react";
import "./App.css";

const WS_URL = import.meta.env.VITE_WS_URL || "/ws";

type Resp = {
  answer: string;
  confidence: number;
  citations: { lang: string; qid: string }[];
  language: string;
  refused: boolean;
  reason: string;
  path: string;
  latency_ms?: number;
  stage_latencies?: Record<string, number>;
};

function toB64(u8: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < u8.length; i++) bin += String.fromCharCode(u8[i]);
  return btoa(bin);
}

export default function App() {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [resp, setResp] = useState<Resp | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const micRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workRef = useRef<AudioWorkletNode | null>(null);
  const stoppingRef = useRef<{ ctx: AudioContext; ws: WebSocket } | null>(null);
  const lastPartialRef = useRef<number>(0);
  const watchdogRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stoppingRef2 = useRef(false);

  async function start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = new AudioContext({ sampleRate: 16000 });
    await ctx.audioWorklet.addModule("/pcm16.js");
    const src = ctx.createMediaStreamSource(stream);
    const work = new AudioWorkletNode(ctx, "pcm16");
    src.connect(work);
    const ws = new WebSocket(WS_URL);
    ws.onmessage = (e) => {
      let msg: any;
      try {
        msg = JSON.parse(e.data as string);
      } catch {
        return;
      }
      const t = msg.text ?? msg.committed_transcript ?? "";
      if (msg.message_type === "partial_transcript" && t) {
        setTranscript(t);
        lastPartialRef.current = performance.now();
      }
      if (msg.message_type === "committed_transcript" || msg.message_type === "committed_transcript_with_timestamps") {
        setTranscript(t);
        const stopper = stoppingRef.current;
        stoppingRef.current = null;
        if (watchdogRef.current) {
          clearInterval(watchdogRef.current);
          watchdogRef.current = null;
        }
        setListening(false);
        if (stopper) {
          try { stopper.ws.close(); } catch { /* noop */ }
          setTimeout(() => { try { stopper.ctx.close(); } catch { /* noop */ } }, 0);
        }
        if (t) ask(t);
      }
    };
    work.port.onmessage = (e) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            message_type: "input_audio_chunk",
            audio_base_64: toB64(new Uint8Array(e.data.data)),
            commit: false,
            sample_rate: 16000,
          })
        );
      }
    };
    wsRef.current = ws;
    ctxRef.current = ctx;
    micRef.current = src;
    workRef.current = work;
    stoppingRef2.current = false;
    setListening(true);
    lastPartialRef.current = 0;
    watchdogRef.current = setInterval(() => {
      if (stoppingRef2.current) return;
      if (lastPartialRef.current > 0 && performance.now() - lastPartialRef.current > 1400) {
        stop();
      }
    }, 250);
  }

  function stop() {
    if (stoppingRef2.current) return;
    stoppingRef2.current = true;
    if (watchdogRef.current) {
      clearInterval(watchdogRef.current);
      watchdogRef.current = null;
    }
    micRef.current?.disconnect();
    workRef.current?.disconnect();
    const ctx = ctxRef.current!;
    const ws = wsRef.current!;
    micRef.current = workRef.current = null;
    // flush trailing silence so the STT commits the utterance
    const silence = new Uint8Array(1600 * 2);
    let sent = 0;
    const timer = setInterval(() => {
      sent += 1;
      const last = sent >= 24;
      ws.send(
        JSON.stringify({
          message_type: "input_audio_chunk",
          audio_base_64: toB64(silence),
          commit: last,
          sample_rate: 16000,
        })
      );
      if (last) {
        clearInterval(timer);
        stoppingRef.current = { ctx, ws };
      }
    }, 60);
    setListening(false);
  }

  async function ask(text: string) {
    const t0 = performance.now();
    try {
      const r = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      const j = (await r.json()) as Resp;
      j.latency_ms = performance.now() - t0;
      setResp(j);
      if (!j.refused && j.answer) speak(j.answer);
    } catch {
      setResp({ answer: "", confidence: 0, citations: [], language: "", refused: true, reason: "network-error", path: "" });
    }
  }

  function speak(text: string) {
    if (!("speechSynthesis" in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = resp?.language || "hi-IN";
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  }

  useEffect(() => () => stop(), []);

  return (
    <main className="app">
      <h1>वॉइस RAG · Voice RAG</h1>
      <p className="sub">HH Goa 2026 · MSMARCO-XI · say a question in any Indic language</p>
      <button className={listening ? "mic on" : "mic"} onClick={listening ? stop : start}>
        {listening ? "■ Stop" : "● Speak"}
      </button>
      {transcript && <p className="transcript">🎤 {transcript}</p>}
      {resp && (
        <section className="answer">
          {resp.refused ? (
            <p className="refused">{resp.reason}</p>
          ) : (
            <>
              <p>{resp.answer}</p>
              <div className="meta">
                lang {resp.language} · {resp.path} · {(resp.latency_ms ?? 0).toFixed(0)}ms · conf{" "}
                {(resp.confidence * 100).toFixed(0)}%
              </div>
              {resp.citations.length > 0 && (
                <ul className="cites">
                  {resp.citations.map((c, i) => (
                    <li key={i}>{c.qid}</li>
                  ))}
                </ul>
              )}
            </>
          )}
        </section>
      )}
    </main>
  );
}