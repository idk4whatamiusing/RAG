import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import EvalsPage from "./EvalsPage";
import { QrModal, StageRings, BrainSelector, LangSelector, TraceBox, BenchBox, type StageInfo, type BenchResult } from "./components";

const WS_URL = import.meta.env.VITE_WS_URL || "/ws";

type Resp = {
  answer: string;
  confidence: number;
  citations: { lang: string; qid: string; snippet?: string; sentence?: string }[];
  language: string;
  refused: boolean;
  reason: string;
  path: string;
  latency_ms?: number;
  stage_latencies?: Record<string, number>;
  trace?: Record<string, unknown>;
  follow_up?: boolean;
};

type Mode = "idle" | "listening" | "thinking" | "speaking";

const CHIPS: { label: string; q: string; kb?: string; lang?: string }[] = [
  { label: "🍉 गोवा चर्च", q: "गोवा का सबसे पुराना चर्च कौन सा है?", kb: "goa" },
  { label: "कोंकणी · कसो आसा?", q: "कसो आसा म्हणजे काय?", kb: "goa" },
  { label: "हिन्दी → தமிழ்", q: "टेलीफोन का आविष्कार किसने किया?", lang: "ta" },
  { label: "guard: मुझे मारना है", q: "मुझे तुम्हें मारना है" },
  { label: "smalltalk", q: "hello goa how are you?" },
];

const INTERVIEW_Q = [
  "तुम्हारा नाम क्या है?",
  "तुम कौन सी भाषा बोलते हो?",
  "गोवा की कोई एक जगह मन में आती है?",
];

const BADGE_LANGS: Record<string, string> = {
  hi: "हिन्दी", ta: "தமிழ்", te: "తెలుగు", bn: "বাংলা", mr: "मराठी", gu: "ગુજરાતી",
  kn: "ಕನ್ನಡ", ml: "മലയാളം", pa: "ਪੰਜਾਬੀ", ur: "اردو", or: "ଓଡ଼ିଆ", as: "অসমীয়া",
  ne: "नेपाली", sa: "संस्कृतम्", en: "English", goa: "गोवा", docs: "bench",
};

const RMS_BARGE = 0.025;
const RMS_SKIP_MS = 300;
const STAGE_ORDER = [
  { id: "g3_lid", label: "lid" },
  { id: "embed", label: "embed" },
  { id: "retrieve", label: "retrieve" },
  { id: "extract", label: "extract" },
  { id: "synth", label: "synth" },
];
const STAGE_COLORS: Record<string, string> = {
  g3_lid: "var(--hhg-yellow)", embed: "var(--hhg-green)", retrieve: "var(--hhg-teal)",
  extract: "var(--hhg-brick)", groq: "var(--hhg-pink)", synth: "var(--hhg-pink)",
};

function toB64(u8: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < u8.length; i++) bin += String.fromCharCode(u8[i]);
  return btoa(bin);
}

export default function App() {
  const [mode, setMode] = useState<Mode>("idle");
  const [transcript, setTranscript] = useState("");
  const [resp, setResp] = useState<Resp | null>(null);
  const [statusMsg, setStatusMsg] = useState("निशा बोलत नाही? बोला — ask in any Indic language");
  const [liveLang, setLiveLang] = useState("");
  const [v2v, setV2v] = useState<number | null>(null);
  const [autoL, setAutoL] = useState(true);
  const [answerLang, setAnswerLang] = useState("auto");
  const [kb, setKb] = useState("main");
  const [qrOpen, setQrOpen] = useState(false);
  const [kIdx, setKIdx] = useState(0);
  const [bench, setBench] = useState<BenchResult | null>(null);
  const [benchBusy, setBenchBusy] = useState(false);
  const [memText, setMemText] = useState("");
  const [memMsg, setMemMsg] = useState("");
  const [lastMem, setLastMem] = useState<{ qid: string; text: string } | null>(null);
  const [ledger, setLedger] = useState({ q: 0, langs: {} as Record<string, number>, cites: 0, refused: 0, remembered: 0, forgotten: 0 });
  const [race, setRace] = useState<number[] | null>(null);
  

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
  const prevTurnRef = useRef<{ query: string; answer: string } | null>(null);
  const wordsRef = useRef<string[]>([]);
  const karaokeStartRef = useRef(0);
  const karaokeMsRef = useRef(0);
  const rafKRef = useRef(0);
  const spokenLangRef = useRef("");
  const interviewRef = useRef<{ step: number; answers: string[] } | null>(null);

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

  /* ---------- karaoke (G2): estimated word sync ---------- */
  useEffect(() => {
    return () => cancelAnimationFrame(rafKRef.current);
  }, []);
  const runKaraoke = useCallback(() => {
    const tick = () => {
      const el = performance.now() - karaokeStartRef.current;
      const frac = Math.min(1, el / Math.max(1, karaokeMsRef.current));
      const idx = Math.floor(frac * wordsRef.current.length);
      setKIdx(idx);
      if (modeRef.current === "speaking" && frac < 1) rafKRef.current = requestAnimationFrame(tick);
    };
    cancelAnimationFrame(rafKRef.current);
    rafKRef.current = requestAnimationFrame(tick);
  }, []);

  /* ---------- session bootstrap ---------- */
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

  /* ---------- pcm ingress ---------- */
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
      setTranscript(t);
      spokenLangRef.current = "";
      tCommitRef.current = performance.now();
      closeWs();
      if (interviewRef.current) {
        setStatusMsg("got it · …");
        void handleInterview(t);
        return;
      }
      setModeBoth("thinking");
      setStatusMsg("got your question · thinking…");
      if (t) ask(t);
    }
  }

  /* N2: inverted interview — निशा asks, the human answers */
  async function interviewStart() {
    stopPlayback();
    
    interviewRef.current = { step: 0, answers: [] };
    setResp(null);
    setStatusMsg("निशा पूछती है…");
    await startInterviewStep(0);
  }

  async function startInterviewStep(step: number) {
    interviewRef.current = { step, answers: interviewRef.current?.answers ?? [] };
    
    const q = INTERVIEW_Q[step];
    if (!q) {
      const a = interviewRef.current?.answers ?? [];
      const name = a[0] || "मित्र";
      const lang = a[1] || "कुछ भी";
      const place = a[2] || "गोवा";
      const summary = `देव बरे करूं ${name}. आप ${lang} बोलते हैं और ${place} आपको पसंद है.`;
      interviewRef.current = null;
      
      setStatusMsg(summary);
      await speak(summary, "hi");
      afterAnswer();
      return;
    }
    setStatusMsg(`सवाल ${step + 1}: ${q}`);
    await speak(q, "hi");
    if (interviewRef.current) {
      start().catch(() => setStatusMsg("tap ● to speak"));
    }
  }

  async function handleInterview(t: string) {
    const iv = interviewRef.current;
    if (!iv) return;
    const answers = [...iv.answers, t];
    interviewRef.current = { step: iv.step + 1, answers };
    
    await startInterviewStep(iv.step + 1);
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
  function chime() {
    const ctx = ctxRef.current;
    if (!ctx) return;
    try {
      const o = ctx.createOscillator();
      const g = ctx.createGain();
      o.frequency.value = 880;
      o.type = "sine";
      g.gain.value = 0.04;
      o.connect(g);
      g.connect(ctx.destination);
      o.start();
      o.stop(ctx.currentTime + 0.06);
    } catch {
      /* noop */
    }
  }

  async function ask(text: string) {
    const t0 = performance.now();
    try {
      const lang = answerLang === "auto" ? undefined : answerLang;
      const body: Record<string, unknown> = { query: text };
      if (lang) body.lang = lang;
      if (kb !== "main") body.kb = kb;
      if (prevTurnRef.current) body.history = [prevTurnRef.current];
      const r = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const j = (await r.json()) as Resp;
      j.latency_ms = performance.now() - t0;
      setResp(j);
      setLedger((l) => ({
        ...l,
        q: l.q + 1,
        langs: { ...l.langs, [j.language]: (l.langs[j.language] || 0) + 1 },
        cites: l.cites + (j.citations?.length || 0),
        refused: l.refused + (j.refused ? 1 : 0),
      }));
      if (lang) spokenLangRef.current = j.language;
      if (!j.refused && j.answer) {
        prevTurnRef.current = { query: text, answer: j.answer };
        setStatusMsg(`answered · ${(j.latency_ms).toFixed(0)}ms · speaking`);
        chime();
        await speak(j.answer, lang || j.language);
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

  async function speak(text: string, ttsLang: string) {
    setModeBoth("speaking");
    const ac = new AbortController();
    abortRef.current = ac;
    playingSinceRef.current = performance.now();
    bargeStreakRef.current = 0;
    wordsRef.current = text.split(/\s+/);
    setKIdx(0);
    try {
      const r = await fetch("/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, lang: ttsLang || "hi" }),
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
      karaokeStartRef.current = performance.now();
      karaokeMsRef.current = audio.duration * 1000;
      runKaraoke();
      setV2v(performance.now() - tCommitRef.current);
    } catch {
      if (ac.signal.aborted) return;
      cancelAnimationFrame(rafKRef.current);
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
    cancelAnimationFrame(rafKRef.current);
    bargeStreakRef.current = 0;
  }

  /* ---------- barge-in / full duplex ---------- */
  function barge() {
    if (modeRef.current !== "speaking") return;
    stopPlayback();
    setStatusMsg("interrupted · listening again");
    listenLater();
  }

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

  /* ---------- GOA-18: live memory, bench, race, ledger, report ---------- */
  async function runBench() {
    setBenchBusy(true);
    try {
      const r = await fetch("/bench", { method: "POST" });
      setBench((await r.json()) as BenchResult);
    } catch {
      /* keep old */
    }
    setBenchBusy(false);
  }

  async function tellNisha() {
    const t = memText.trim();
    if (!t) return;
    setMemMsg("याद रख रही हूँ…");
    try {
      const r = await fetch("/remember", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: t, kb: kb === "goa" || kb === "main" ? "goa" : kb }),
      });
      const j = await r.json();
      if (j.ok) {
        setLastMem({ qid: j.qid, text: t });
        setMemMsg(`याद रखा · ${j.qid} — अब इसके बारे में पूछो! (phone भी याद करेगी — scan me)`);
        setLedger((l) => ({ ...l, remembered: l.remembered + 1 }));
        chime();
        setMemText("");
      } else {
        setMemMsg(j.error || "remember failed");
      }
    } catch {
      setMemMsg("remember failed");
    }
  }

  async function forgetMem() {
    if (!lastMem) return;
    try {
      const r = await fetch("/forget", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ qid: lastMem.qid }),
      });
      const j = await r.json();
      setMemMsg(j.ok ? `भूल गई · ${lastMem.qid}` : "forget failed");
      if (j.ok) {
        setLedger((l) => ({ ...l, forgotten: l.forgotten + 1 }));
        setLastMem(null);
      }
      chime();
    } catch {
      /* noop */
    }
  }

  async function raceIt() {
    const q = "टेलीफोन का आविष्कार किसने किया?";
    setRace(null);
    const t1 = performance.now();
    await fetch("/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: q }) });
    const m1 = performance.now() - t1;
    const t2 = performance.now();
    await fetch("/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query: q }) });
    const m2 = performance.now() - t2;
    setRace([m1, m2]);
  }

  function reportCard() {
    const l = ledger;
    const langsTxt = Object.keys(l.langs).map((k) => BADGE_LANGS[k] || k).join(", ") || "—";
    const txt = `इस सेशन में ${l.q} सवाल · ${Object.keys(l.langs).length} भाषाएँ · ${l.cites} स्रोत · ${l.refused} refuse · ${l.remembered} याद · ${l.forgotten} भूल. भाषाएँ: ${langsTxt}.`;
    setStatusMsg(txt);
    chime();
    void speak(txt, "hi").then(() => listenLater());
  }

  function chip(c: { q: string; kb?: string; lang?: string }) {
    closeWs();
    stopPlayback();
    setKb(c.kb || "main");
    if (c.lang) setAnswerLang(c.lang);
    setResp(null);
    setTranscript("");
    setV2v(null);
    setModeBoth("thinking");
    setStatusMsg("chip query…");
    tCommitRef.current = performance.now();
    ask(c.q);
  }

  if (window.location.hash === "#/evals") {
    return <EvalsPage onBack={() => (window.location.hash = "")} />;
  }

  const stages = resp?.stage_latencies || {};
  const stageInfo: StageInfo[] = STAGE_ORDER.filter((s) => stages[s.id] !== undefined).map((s) => ({
    id: s.id,
    label: s.label,
    ms: stages[s.id],
    color: STAGE_COLORS[s.id] || "var(--hhg-yellow)",
  }));

  const crossLingual =
    answerLang !== "auto" && resp && !resp.refused && resp.language && BADGE_LANGS[resp.language];

  const words = wordsRef.current;

  return (
    <main className="app">
      <QrModal open={qrOpen} onClose={() => setQrOpen(false)} />
      <div className="badges">
        <img className="goa" src="/assets/hhg/goa_hindi.svg" alt="गोवा" />
        <img className="clock" src="/assets/hhg/2-47.svg" alt="2:47 pm Studio" />
      </div>
      <h1>
        वॉइस <span>RAG</span>
      </h1>
      <p className="sub">निशा · Goa desk — HH Goa 2026 · 13 भाषाएँ · full duplex · receipts</p>

      {liveLang && <div className="langbadge">detected: {BADGE_LANGS[liveLang] || liveLang}</div>}
      {crossLingual && <div className="langbadge xl">answering in {crossLingual}</div>}
      {kb !== "main" && <div className="langbadge kb">brain: {kb === "goa" ? "गोवा" : "bench docs"}</div>}

      <BrainSelector kb={kb} setKb={setKb} />
      <LangSelector lang={answerLang} setLang={setAnswerLang} />

      <button
        className={mode === "listening" ? "mic on" : "mic"}
        onClick={mode === "idle" || mode === "listening" ? (mode === "listening" ? stop : start) : stop}
      >
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
          <button key={c.label} className="chip" onClick={() => chip(c)} title={c.q}>
            {c.label}
          </button>
        ))}
        <button className="chip" onClick={interviewStart}>
          🎤 निशा पूछती है
        </button>
        <button className="chip" onClick={raceIt}>
          🏁 race
        </button>
        <button className="chip" onClick={reportCard}>
          🧾 report
        </button>
        <button className="chip" onClick={() => setQrOpen(true)}>
          📱 scan me
        </button>
      </div>

      <div className="rememberbox">
        <input
          value={memText}
          placeholder="निशा याद रखो — एक तथ्य…"
          onChange={(e) => setMemText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") tellNisha();
          }}
        />
        <button className="chip" onClick={tellNisha}>याद रखो</button>
        {lastMem && (
          <button className="chip" onClick={forgetMem}>🧹 भूल जाओ</button>
        )}
      </div>
      {(memMsg || lastMem) && (
        <p className="memnote">
          {memMsg}
          {lastMem ? " · memory is live everywhere — teach here, ask on the phone" : ""}
        </p>
      )}

      <BenchBox bench={bench} busy={benchBusy} onRun={runBench} />

      {race && (
        <div className="racebox">
          <div className="racetile">🖥 laptop <b>{race[0].toFixed(0)}ms</b></div>
          <div className="racetile">📱 phone <b>{race[1].toFixed(0)}ms</b></div>
        </div>
      )}

      <div className="ledger">
        <span>सवाल <b>{ledger.q}</b></span>
        <span>भाषाएँ <b>{Object.keys(ledger.langs).length}</b></span>
        <span>स्रोत <b>{ledger.cites}</b></span>
        <span>refused <b>{ledger.refused}</b></span>
        <span>याद <b>{ledger.remembered}</b></span>
        <span>भूल <b>{ledger.forgotten}</b></span>
      </div>

      {stageInfo.length > 0 && <StageRings stages={stageInfo} running={mode === "speaking"} />}

      {resp && (
        <section className="answer">
          <img className="frame fr-top" src="/assets/hhg/138-frame-1948755142-54-27257.svg" alt="" />
          {resp.refused ? (
            <>
              <p className="refused">
                {resp.reason}
                {resp.language ? ` · lang ${resp.language}` : ""}
              </p>
              {resp.trace && <TraceBox trace={resp.trace} />}
            </>
          ) : (
            <>
              {resp.follow_up && <div className="fbadge">follow-up · re-anchored to previous answer</div>}
              {mode === "speaking" && words.length > 1 ? (
                <p className="anstext karaoke">
                  {words.map((w, i) => (
                    <span key={i} className={i <= kIdx ? "kw on" : "kw"}>
                      {w}{" "}
                    </span>
                  ))}
                </p>
              ) : (
                <p className="anstext">{resp.answer}</p>
              )}
              <div className="meta">
                lang {resp.language} · {resp.path} · {(resp.latency_ms ?? 0).toFixed(0)}ms · conf{" "}
                {(resp.confidence * 100).toFixed(0)}%
              </div>
              {Object.keys(stages).length > 0 && (
                <div className="hud">
                  {STAGE_ORDER.filter((s) => stages[s.id] !== undefined).map((s) => (
                    <div
                      key={s.id}
                      className="hudseg"
                      style={{
                        width: `${(stages[s.id] / Math.max(1, Object.values(stages).reduce((a: number, b: number) => a + b, 0))) * 100}%`,
                        background: STAGE_COLORS[s.id] || "var(--hhg-yellow)",
                      }}
                      title={`${s.label} ${stages[s.id].toFixed(1)}ms`}
                    >
                      {stages[s.id] >= 15 ? `${s.label} ${stages[s.id].toFixed(0)}` : ""}
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
                      {c.snippet && (
                        <p className="snippet">
                          {c.sentence && c.snippet.includes(c.sentence) ? (
                            <>
                              {c.snippet.slice(0, c.snippet.indexOf(c.sentence))}
                              <mark className="sent">{c.sentence}</mark>
                              {c.snippet.slice(c.snippet.indexOf(c.sentence) + c.sentence.length)}
                            </>
                          ) : (
                            c.snippet
                          )}
                        </p>
                      )}
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
        हैलो गोवा · निशा का डेस्क · हिंदी · english · اردو · বাংলা · ಕೊಂಕಣಿ ·{" "}
        <a href="#/evals" className="evalslink">
          evals →
        </a>
      </p>
    </main>
  );
}
