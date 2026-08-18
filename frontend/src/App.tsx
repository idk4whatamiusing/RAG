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

export default function App() {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [resp, setResp] = useState<Resp | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const micRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workRef = useRef<AudioWorkletNode | null>(null);

  async function start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = new AudioContext({ sampleRate: 16000 });
    await ctx.audioWorklet.addModule("/pcm16.js");
    const src = ctx.createMediaStreamSource(stream);
    const work = new AudioWorkletNode(ctx, "pcm16");
    src.connect(work);
    work.port.onmessage = (e) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(e.data.data);
      }
    };
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    ws.onmessage = (e) => {
      const msg = typeof e.data === "string" ? JSON.parse(e.data) : null;
      if (!msg) return;
      const seg = msg.text_segments?.[0];
      if (seg?.text) setTranscript((t) => (t ? t + " " : "") + seg.text.trim());
      if (msg.is_final || msg.type === "transcript_done") {
        setListening(false);
        ask(seg?.text || "");
      }
    };
    wsRef.current = ws;
    ctxRef.current = ctx;
    micRef.current = src;
    workRef.current = work;
    setListening(true);
  }

  function stop() {
    micRef.current?.disconnect();
    workRef.current?.disconnect();
    ctxRef.current?.close();
    wsRef.current?.close();
    micRef.current = ctxRef.current = workRef.current = null;
    wsRef.current = null;
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