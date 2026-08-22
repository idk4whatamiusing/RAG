export function QrModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  const url = window.location.origin;
  return (
    <div className="qrback" onClick={onClose}>
      <div className="qrmod" onClick={(e) => e.stopPropagation()}>
        <img src={`https://api.qrserver.com/v1/create-qr-code/?size=280x280&data=${encodeURIComponent(url)}`} alt="scan me" />
        <p>निशा · scan me, ask me — works on your phone too</p>
        <button className="chip" onClick={onClose}>close</button>
      </div>
    </div>
  );
}

export type StageInfo = { id: string; label: string; ms: number; color: string };

export function StageRings({ stages, running }: { stages: StageInfo[]; running: boolean }) {
  const total = stages.reduce((a, s) => a + s.ms, 0) || 1;
  return (
    <div className={`rings ${running ? "run" : ""}`}>
      {stages.map((s, i) => (
        <div
          key={s.id}
          className="ring"
          style={{ animationDelay: `${stages.slice(0, i).reduce((a, x) => a + x.ms, 0) * 60 / Math.max(1, total)}ms` }}
        >
          <span className="ringdot" style={{ background: s.color }} />
          <span className="ringlab">{s.label}</span>
          <span className="ringms">{s.ms ? s.ms.toFixed(0) : ""}</span>
        </div>
      ))}
    </div>
  );
}

export function BrainSelector({ kb, setKb }: { kb: string; setKb: (k: string) => void }) {
  const opts = [
    { id: "main", label: "मुख्य corpus" },
    { id: "goa", label: "गोवा brain" },
    { id: "docs", label: "bench docs" },
  ];
  return (
    <div className="selrow">
      {opts.map((o) => (
        <button key={o.id} className={`sel ${kb === o.id ? "on" : ""}`} onClick={() => setKb(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function LangSelector({ lang, setLang }: { lang: string; setLang: (l: string) => void }) {
  const opts = [
    { id: "auto", label: "आपकी भाषा" },
    { id: "hi", label: "हिन्दी" },
    { id: "ta", label: "தமிழ்" },
    { id: "ur", label: "اردو" },
    { id: "bn", label: "বাংলা" },
  ];
  return (
    <div className="selrow small">
      {opts.map((o) => (
        <button key={o.id} className={`sel ${lang === o.id ? "on" : ""}`} onClick={() => setLang(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function TraceBox({ trace }: { trace: Record<string, unknown> }) {
  return (
    <pre className="trace">
      {Object.entries(trace)
        .map(([k, v]) => `${k} = ${v}`)
        .join("\n")}
    </pre>
  );
}
