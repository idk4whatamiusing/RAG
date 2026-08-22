# Demo script — निशा · वॉइस RAG (HH Goa 2026)

Live URL: https://15.206.175.100.sslip.io — deploy via SSM (ssh unreliable from CGNAT).

Persona: **निशा · Goa desk** — ask in any Indic language, answer spoken back (Sarvam AI
Indian voices when ElevenLabs is exhausted, Edge as the free umbrella). The Goa brain knows
Goa; the main brain knows the MSMARCO-XI corpus; the bench brain can talk about itself.

## 60-second flow

1. **Open the page.** Greeting: "निशा बोलत नाही? बोला". Chips row = the fallback insurance.
2. **Goa brain (the soul):** tap the `गोवा brain` selector, then ask out loud:
   "गोवा का सबसे पुराना चर्च कौन सा है?"
   - Answer comes from a hand-curated Goa corpus, cited, with the exact sentence highlighted.
3. **Konkani moment:** tap `कोंकणी · कसो आसा?` — the answer explains the phrase, in Konkani text via the Goa brain. Nobody else has Konkani.
4. **Cross-lingual flex (main brain):** tap `हिन्दी → தமிழ்` — a Hindi question answered in Tamil.
   Watch the "answering in தமிழ்" badge + the Tamil voice (Sarvam ta-IN). Language badge flips live while you speak.
5. **Barge-in:** while it's speaking, talk over it — it cuts and re-listens. Full duplex.
6. **Guardrail:** "hello goa how are you?" → clean refusal, and the trace shows the math
   (`answer-confidence = 0.0 < 0.28 → not-in-knowledge-base`).
7. **Bench it now:** the `⚡ /bench` — actually, the chips row hosts it? No: bench is a server
   endpoint behind the scenes; stage rings + HUD above the answer show lid/embed/retrieve/extract
   live, and the voice→voice number shows speech-end → first audio (~400-700ms).
8. **Receipts:** footer `evals →` opens the 13-lang × 4-strategy chunking table + P50/P70/P100.
   Ask the bench brain "what is the tamil recall?" (docs brain) — it answers from its own eval doc.
9. **Scan me:** QR → the live app on a judge's phone.

## Voice chain (no-code demo line)

`X-TTS-Source` header shows the live tier: eleven → sarvam → edge. The ElevenLabs account is
burned (402) so Sarvam AI bulbul:v3 (shubh) is the voice today — an Indian model speaking
Indian answers. Top up ElevenLabs anytime → it silently upgrades.

## Tips

- Headphones for barge-in (browser AEC handles speaker bleed; headphones are bulletproof).
- Warm the box first: hit /health once after any deploy, then /bench once (first call is cold).
- Repeat queries hit the LRU /tts cache — instant, zero credits.
- If the stage mic dies: all chips are pure HTTP.
