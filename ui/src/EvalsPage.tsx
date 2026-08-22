import { useEffect, useState } from "react";

type EvalRow = {
  lang: string;
  queries: number;
  pool_passages: number;
  strategies: Record<string, { chunks: number; build_s: number; recall_at_20: number; answerable: number }>;
};

const BENCH = [
  { lang: "hi", P50: 100.2, P70: 119.4, P100: 243.6, mean: 102.5, recall: 0.067, refused: "0/150" },
  { lang: "bn", P50: 14.8, P70: 16.1, P100: 114.8, mean: 16.0, recall: 0.000, refused: "0/150" },
  { lang: "mr", P50: 15.0, P70: 16.1, P100: 25.7, mean: 15.3, recall: 0.000, refused: "0/150" },
  { lang: "ta", P50: 15.2, P70: 16.1, P100: 28.1, mean: 15.6, recall: 0.000, refused: "0/150" },
  { lang: "ur", P50: 14.5, P70: 15.8, P100: 163.7, mean: 16.6, recall: 0.000, refused: "36/150" },
];

const STRATS = ["native", "fixed", "semantic", "metadata"];

export default function EvalsPage({ onBack }: { onBack: () => void }) {
  const [rows, setRows] = useState<EvalRow[] | null>(null);

  useEffect(() => {
    fetch("/evals-data.json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setRows)
      .catch(() => setRows(null));
  }, []);

  return (
    <main className="app evals">
      <div className="badges">
        <img className="goa" src="/assets/hhg/goa_hindi.svg" alt="गोवा" />
        <img className="clock" src="/assets/hhg/2-47.svg" alt="2:47 pm Studio" />
      </div>
      <h1>
        वॉइस <span>RAG</span> <em className="eyesub">…/evals</em>
      </h1>
      <button className="chip back" onClick={onBack}>
        ← back
      </button>

      <section className="evalsec">
        <h2>Chunking eval — 13 langs × 4 strategies</h2>
        <p className="evalsub">
          mean recall@20 (300 val queries / lang, 4000-passage pool, cross-split). Source:{" "}
          <code>data/eval_chunking.jsonl</code>
        </p>
        {rows ? (
          <table className="evtbl">
            <thead>
              <tr>
                <th>lang</th>
                {STRATS.map((s) => (
                  <th key={s}>{s}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.lang}>
                  <td>{r.lang}</td>
                  {STRATS.map((s) => {
                    const d = r.strategies[s];
                    const best = Math.max(...STRATS.map((x) => r.strategies[x].recall_at_20));
                    return (
                      <td key={s} className={d.recall_at_20 === best ? "best" : ""}>
                        {d.recall_at_20.toFixed(3)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="evalsub">(eval data not bundled — see data/eval_chunking.jsonl)</p>
        )}
        <p className="evalsub">
          verdict: <b>native</b> mean 0.1900 = metadata, &gt; fixed 0.1885, &gt;&gt; semantic 0.1382 →
          live corpus already native, no rebuild.
        </p>
      </section>

      <section className="evalsec">
        <h2>T2 bench — live pgvector (150 queries / lang on EC2)</h2>
        <table className="evtbl">
          <thead>
            <tr>
              <th>lang</th>
              <th>P50</th>
              <th>P70</th>
              <th>P100</th>
              <th>mean ms</th>
              <th>recall@20</th>
              <th>refused</th>
            </tr>
          </thead>
          <tbody>
            {BENCH.map((b) => (
              <tr key={b.lang}>
                <td>{b.lang}</td>
                <td>{b.P50}</td>
                <td>{b.P70}</td>
                <td>{b.P100}</td>
                <td>{b.mean}</td>
                <td>{b.recall.toFixed(3)}</td>
                <td>{b.refused}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="evalsub">
          warm-path P50/P70 &lt; 200ms everywhere (hi = 500k-chunk partition; P100 = first-query warmup +
          150ms synthesis budget). Docs: <code>docs/bench-results.md</code>,{" "}
          <code>docs/requirements.md</code>
        </p>
      </section>

      <p className="foot">टेलीफोन के बारे में पूछो · ask about the telephone · बोलिए</p>
    </main>
  );
}
