import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import EvalsPage from "./EvalsPage";

const WS_URL = import.meta.env.VITE_WS_URL || "/ws";

type Resp = {
  answer: string;
  confidence: number;
  citations: { lang: string; qid: string; snippet?: string }[];
  language: string;
  refused: boolean;
  reason: string;
  path: string;
  latency_ms?: number;
  stage_latencies?: Record<string, number>;
};

type Mode = "idle" | "listening" | "thinking" | "speaking";

const CHIPS: { label: string; q: string }[] = [
  { label: "हिन्दी · टेलीफोन", q: "टेलीफोन का आविष्कार किसने किया?" },
  { label: "தமிழ் · தலைநகரம்", q: "சென்னை எந்த மாநிலத்தின் தலைநகரம்?" },
  { label: "اردو · دارالحکومت", q: "پاکستان کا دارالحکومت کیا ہے؟" },
  { label: "smalltalk", q: "hello goa how are you?" },
];

const BADGE_LANGS: Record<string, string> = {
  hi: "हिन्दी",
  ta: "தமிழ்",
  te: "తెలుగు",
  bn: "বাংলা",
  mr: "मराठी",
  gu: "ગુજરાતી",
  kn: "ಕನ್ನಡ",
  ml: "മലയാളം",
  pa: "ਪੰਜਾਬੀ",
  ur: "اردو",
  or: "ଓଡ଼ିଆ",
  as: "অসমীয়া",
  ne: "नेपाली",
  sa: "संस्कृतम्",
  en: "English",
};

const RMS_BARGE = 0.025;
const RMS_SKIP_MS = 300;

function toB64(u8: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < u8.length; i++) bin += String.fromCharCode(u8[i]);
  return btoa(bin);
}

export default function App() {
  const [mode, setMode] = useState<Mode>("idle");
  const [transcript, setTranscript] = useState("");
  const [resp, setResp] = useState<Resp | null>(null);
  const [statusMsg, setStatusMsg] = useState("click ● to speak");
  const [liveLang, setLiveLang] = useState("");
  const [v2v, setV2v] = useState<number | null>(null);
  const [autoL, setAutoL] = useState(true);

  const ctxRef = useRef<AudioContext | null>(null);
  const workRef = useRef<AudioWorkletNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const modeRef = useRef<Mode>("idle");
  const tCommitRef = useRef(0);
  const playSrcRef = useRef<AudioBufferSourceNode | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const watchdogRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastPartialRef = useRef(0);
  const bargeStreakRef = useRef(0);
  const playingSinceRef = useRef(0);
  const waveRef = useRef<number[]>([]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef(0);
  const listenTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const setModeBoth = useCallback((m: Mode) => {
    modeRef.current = m;
    setMode(m);
  }, []);

  /* ---------- waveform raf loop ---------- */
  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const g = cvs.getContext("2d")!;
    const draw = () => {
      const w = cvs.width;
      const h = cvs.height;
      g.clearRect(0, 0, w, h);
      const data = waveRef.current;
      if (data.length > 1) {
        const mid = h / 2;
        g.strokeStyle = modeRef.current === "listening" ? "var(--hhg-yellow)" : "var(--hhg-green)";
        g.lineWidth = 2;
        g.beginPath();
        const step = (w - 4) / (data.length - 1);
        data.forEach((v, i) => {
          const x = 2 + i * step;
          const y = mid - v * (h / 2 - 4);
          if (i === 0) g.moveTo(x, y);
          else g.lineTo(x, y);
        });
        g.stroke();
      }
      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  /* ---------- session bootstrap (once per page) ---------- */
  async function ensureSession() {
    if (ctxRef.current && workRef.current) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    const ctx = new AudioContext({ sampleRate: 16000 });
    await ctx.audioWorklet.addModule("/pcm16.js");
    const src = ctx.createMediaStreamSource(stream);
    const work = new AudioWorkletNode(ctx, "pcm16");
    src.connect(work);
    ctxRef.current = ctx;
    workRef.current = work;
    work.port.onmessage = (e) => onPcm(e.data);
  }

  /* ---------- pcm ingress: feed ws OR barge-in rms ---------- */
  const onPcm = (msg: any) => {
    const arr = new Int16Array(msg.data);
    const m = modeRef.current;
    if (m === "listening") {
      pushWave(arr);
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            message_type: "input_audio_chunk",
            audio_base_64: toB64(new Uint8Array(arr.buffer)),
            commit: false,
            sample_rate: 16000,
          })
        );
      }
    } else if (m === "speaking") {
      pushWave(arr);
      const now = performance.now();
      if (now - playingSinceRef.current > RMS_SKIP_MS) {
        if (rmsOf(arr) > RMS_BARGE) {
          bargeStreakRef.current += 1;
          if (bargeStreakRef.current >= 3) barge();
        } else {
          bargeStreakRef.current = 0;
        }
      }
    }
  };

  const pushWave = (arr: Int16Array) => {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += Math.abs(arr[i]);
    const w = waveRef.current;
    w.push(s / arr.length / 32768);
    if (w.length > 140) w.splice(0, w.length - 140);
  };

  function rmsOf(arr: Int16Array): number {
    let ss = 0;
    for (let i = 0; i < arr.length; i++) ss += arr[i] * arr[i];
    return Math.sqrt(ss / arr.length) / 32768;
  }

  /* ---------- ws lifecycle ---------- */
  const openWS = useCallback(
    () =>
      new Promise<WebSocket>((resolve, reject) => {
        const ws = new WebSocket(WS_URL);
        ws.onopen = () => {
          ws.onmessage = (e) => onWsMsg(e);
          ws.onclose = (e) => {
            if (modeRef.current === "listening") {
              setStatusMsg(`connection lost (${e.code})`);
              setModeBoth("idle");
            }
          };
          resolve(ws);
        };
        ws.onerror = () => reject(new Error("ws error"));
        wsRef.current = ws;
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  function onWsMsg(e: MessageEvent) {
    let msg: any;
    try {
      msg = JSON.parse(e.data as string);
    } catch {
      return;
    }
    const t = msg.text ?? msg.committed_transcript ?? "";
    const ld = msg.language_detection ?? msg.language;
    if (Array.isArray(ld) && ld.length) setLiveLang(ld[0].language || ld[0]);
    else if (typeof ld === "string" && ld) setLiveLang(ld);
    if (msg.message_type === "partial_transcript" && t) {
      setTranscript(t);
      lastPartialRef.current = performance.now();
    }
    if (msg.message_type === "committed_transcript" || msg.message_type === "committed_transcript_with_timestamps") {
      clearWatchdog();
      setLiveLang("");
      const text = t;
      setTranscript(text);
      tCommitRef.current = performance.now();
      closeWs();
      setModeBoth("thinking");
      setStatusMsg("got your question · thinking…");
      if (text) ask(text);
    }
  }

  const closeWs = () => {
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        /* noop */
      }
      wsRef.current = null;
    }
  };

  const clearWatchdog = () => {
    if (watchdogRef.current) {
      clearInterval(watchdogRef.current);
      watchdogRef.current = null;
    }
  };

  /* ---------- listen / commit ---------- */
  async function start() {
    try {
      await ensureSession();
      setTranscript("");
      setResp(null);
      setV2v(null);
      const ws = await openWS();
      wsRef.current = ws;
      setModeBoth("listening");
      setStatusMsg("listening · speak now");
      lastPartialRef.current = 0;
      watchdogRef.current = setInterval(() => {
        if (modeRef.current !== "listening") return;
        if (lastPartialRef.current > 0 && performance.now() - lastPartialRef.current > 1400) {
          commitCurrent();
        }
      }, 250);
    } catch {
      setModeBoth("idle");
      setStatusMsg("mic / speech service unavailable — use chips below");
    }
  }

  function commitCurrent() {
    if (modeRef.current !== "listening") return;
    clearWatchdog();
    const ws = wsRef.current;
    if (!ws) {
      setModeBoth("idle");
      return;
    }
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
      if (last) clearInterval(timer);
    }, 60);
    setModeBoth("thinking");
    setStatusMsg("finalizing…");
  }

  /* ---------- answer ---------- */
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
      if (!j.refused && j.answer) {
        setStatusMsg(`answered · ${(j.latency_ms).toFixed(0)}ms · speaking`);
        await speak(j.answer);
      } else {
        setStatusMsg(`refused · ${(j.latency_ms).toFixed(0)}ms · ${j.reason}`);
      }
      afterAnswer();
    } catch {
      setResp({ answer: "", confidence: 0, citations: [], language: "", refused: true, reason: "network-error", path: "" });
      setStatusMsg("query failed — network error");
      afterAnswer();
    }
  }

  async function speak(text: string) {
    setModeBoth("speaking");
    const ac = new AbortController();
    abortRef.current = ac;
    playingSinceRef.current = performance.now();
    bargeStreakRef.current = 0;
    try {
      const r = await fetch("/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang: resp?.language || "hi" }),
        signal: ac.signal,
      });
      if (!r.ok) throw new Error(`tts ${r.status}`);
      const buf = await r.arrayBuffer();
      const ctx = ctxRef.current;
      if (!ctx || ac.signal.aborted) return;
      const audio = await ctx.decodeAudioData(buf);
      if (ac.signal.aborted) return;
      const srcNode = ctx.createBufferSource();
      srcNode.buffer = audio;
      srcNode.connect(ctx.destination);
      srcNode.onended = () => {
        playSrcRef.current = null;
      };
      srcNode.start();
      playSrcRef.current = srcNode;
      ctx.resume().catch(() => {});
      setV2v(performance.now() - tCommitRef.current);
    } catch {
      if (ac.signal.aborted) return;
      if ("speechSynthesis" in window) {
        const u = new SpeechSynthesisUtterance(text);
        u.lang = (resp?.language === "eng_Latn" ? "en" : resp?.language || "hi") + "-IN";
        speechSynthesis.cancel();
        speechSynthesis.speak(u);
        await new Promise<void>((res) => {
          u.onend = () => res();
          u.onerror = () => res();
        });
      }
    }
  }

  function stopPlayback() {
    if (abortRef.current) {
      try {
        abortRef.current.abort();
      } catch {
        /* noop */
      }
      abortRef.current = null;
    }
    if (playSrcRef.current) {
      try {
        playSrcRef.current.stop();
      } catch {
        /* noop */
      }
      playSrcRef.current = null;
    }
    try {
      speechSynthesis.cancel();
    } catch {
      /* noop */
    }
    bargeStreakRef.current = 0;
  }

  /* ---------- barge-in ---------- */
  function barge() {
    if (modeRef.current !== "speaking") return;
    stopPlayback();
    setStatusMsg("interrupted · listening again");
    listenLater();
  }

  /* ---------- full-duplex: auto-return to listening ---------- */
  function afterAnswer() {
    listenLater();
  }

  function listenLater() {
    setModeBoth("idle");
    if (listenTimerRef.current) clearTimeout(listenTimerRef.current);
    listenTimerRef.current = setTimeout(() => {
      if (autoL) {
        start().catch(() => setStatusMsg("tap ● to speak again"));
      }
    }, 400);
  }

  /* ---------- manual stop ---------- */
  function stop() {
    clearWatchdog();
    stopPlayback();
    if (modeRef.current === "speaking") {
      setModeBoth("idle");
      setStatusMsg("stopped");
      return;
    }
    if (modeRef.current === "listening") commitCurrent();
    else if (modeRef.current === "thinking") {
      setModeBoth("idle");
      setStatusMsg("stopped");
    }
  }

  function chip(q: string) {
    closeWs();
    stopPlayback();
    setResp(null);
    setTranscript("");
    setV2v(null);
    setModeBoth("thinking");
    setStatusMsg("chip query…");
    tCommitRef.current = performance.now();
    ask(q);
  }

  if (window.location.hash === "#/evals") {
    return <EvalsPage onBack={() => (window.location.hash = "")} />;
  }

  const stages = resp?.stage_latencies || {};
  const stageOrder = ["g3_lid", "embed", "retrieve", "extract", "groq", "synth"];
  const stageTotal = Math.max(
    1,
    Object.values(stages).reduce((a: number, b: number) => a + b, 0)
  );
  const stageLabels: Record<string, string> = {
    g3_lid: "lid",
    embed: "embed",
    retrieve: "retrieve",
    extract: "extract",
    groq: "synth",
    synth: "synth",
  };
  const stageColors: Record<string, string> = {
    g3_lid: "var(--hhg-yellow)",
    embed: "var(--hhg-green)",
    retrieve: "var(--hhg-teal)",
    extract: "var(--hhg-brick)",
    groq: "var(--hhg-pink)",
    synth: "var(--hhg-pink)",
  };

  return (
    <main className="app">
      <div className="badges">
        <img className="goa" src="/assets/hhg/goa_hindi.svg" alt="गोवा" />
        <img className="clock" src="/assets/hhg/2-47.svg" alt="2:47 pm Studio" />
      </div>
      <h1>
        वॉइस <span>RAG</span>
      </h1>
      <p className="sub">HH Goa 2026 · MSMARCO-XI · ask in any Indic language</p>

      {liveLang && <div className="langbadge">detected: {BADGE_LANGS[liveLang] || liveLang}</div>}

      <button className={mode === "listening" ? "mic on" : "mic"} onClick={mode === "idle" || mode === "listening" ? (mode === "listening" ? stop : start) : stop}>
        {mode === "listening" ? "■ Stop" : mode === "speaking" ? "■ Interrupt" : "● Speak"}
      </button>

      <label className="autol">
        <input type="checkbox" checked={autoL} onChange={(e) => setAutoL(e.target.checked)} />
        keep listening after each answer
      </label>

      {transcript && <p className="transcript">🎤 {transcript}</p>}
      <canvas ref={canvasRef} className="wave" width={560} height={56} />

      <p className={`status ${mode === "speaking" ? "speaking" : mode}`}>
        {mode === "listening" && "● "}
        {mode !== "listening" && "✓ "}
        {statusMsg}
      </p>

      {resp && !resp.refused && v2v !== null && (
        <div className="v2v">voice→voice {(v2v).toFixed(0)}ms after you stopped talking</div>
      )}

      <div className="chips">
        {CHIPS.map((c) => (
          <button key={c.label} className="chip" onClick={() => chip(c.q)} title={c.q}>
            {c.label}
          </button>
        ))}
      </div>

      {resp && (
        <section className="answer">
          <img className="frame fr-top" src="/assets/hhg/138-frame-1948755142-54-27257.svg" alt="" />
          {resp.refused ? (
            <p className="refused">
              {resp.reason}
              {resp.language ? ` · lang ${resp.language}` : ""}
            </p>
          ) : (
            <>
              <p className="anstext">{resp.answer}</p>
              <div className="meta">
                lang {resp.language} · {resp.path} · {(resp.latency_ms ?? 0).toFixed(0)}ms · conf{" "}
                {(resp.confidence * 100).toFixed(0)}%
              </div>
              {Object.keys(stages).length > 0 && (
                <div className="hud">
                  {stageOrder
                    .filter((s) => stages[s] !== undefined)
                    .map((s) => (
                      <div
                        key={s}
                        className="hudseg"
                        style={{ width: `${(stages[s] / stageTotal) * 100}%`, background: stageColors[s] }}
                        title={`${stageLabels[s]} ${stages[s].toFixed(1)}ms`}
                      >
                        {stages[s] >= 15 ? `${stageLabels[s]} ${stages[s].toFixed(0)}` : ""}
                      </div>
                    ))}
                </div>
              )}
              {resp.citations.length > 0 && (
                <ul className="cites">
                  {resp.citations.map((c, i) => (
                    <details key={i}>
                      <summary>
                        [{i + 1}] {c.lang} · {c.qid}
                      </summary>
                      {c.snippet && <p className="snippet">{c.snippet}</p>}
                    </details>
                  ))}
                </ul>
              )}
            </>
          )}
          <img className="frame fr-bottom" src="/assets/hhg/138-frame-1948755142-54-27257.svg" alt="" />
        </section>
      )}

      <p className="foot">
        हैलो गोवा · hello goa · हिंदी · english · اردو · বাংলা ·{" "}
        <a href="#/evals" className="evalslink">
          evals →</a>
      </p>
    </main>
  );
}
