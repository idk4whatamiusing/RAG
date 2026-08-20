# Bench results — live RDS corpus (pgvector)

Run: `scripts/bench.py --rds --n 150 --langs hi,bn,mr,ta,ur` on EC2 (box), 2026-08-20,
against the live `/query` corpus (`chunks` table, 500k hi + 19.7k × 12 other langs).
Val queries streamed from `ai4bharat/MSMARCO-XI/validation/*val.parquet`, cross-split
(train-index vs holdout val queries — the production-relevant scenario).

| lang | P50 | P70 | P100 | mean | recall@20 | refused |
|---|---|---|---|---|---|---|
| hi | 100.2 | 119.4 | 243.6 | 102.5 | 0.067 | 0/150 |
| bn | 14.8 | 16.1 | 114.8 | 16.0 | 0.000 | 0/150 |
| mr | 15.0 | 16.1 | 25.7 | 15.3 | 0.000 | 0/150 |
| ta | 15.2 | 16.1 | 28.1 | 15.6 | 0.000 | 0/150 |
| ur | 14.5 | 15.8 | 163.7 | 16.6 | 0.000 | 36/150 |

Notes
- Warm-path P50/P70 well under 200 ms for every language (hi is the big 500k-chunk partition; others are 20k samples).
- P100 outliers = first-query warmup (fresh RDS connection via SSM+RDS describe) + the 150 ms synthesis budget on weak-confidence queries (`SYNTH_BUDGET_MS`).
- recall@20 = exact-gold token-overlap against held-out val passages while the index is built from train queries; same-question strategy comparison is in `data/eval_chunking.jsonl` (mean recall@20 native=0.1900 > fixed=0.1885 > semantic=0.1382; native == metadata == live corpus → no rebuild required).
- ur's 36 refusals = off-topic gate on the 30k-query ur sample (top sim < 0.30); the other four langs refused nothing — answer-and-ground via extractive path.
- Guardrail probes on live `/query` (outside bench): non-corpus speech → `not-in-knowledge-base`; Hindi question → answered with citations, conf 0.99.