# Demo script — वॉइस RAG (HH Goa 2026)

Live URL: https://15.206.175.100.sslip.io — deploy via SSM (ssh from CGNAT is unreliable).

## 60-second flow

1. **Open** — land on the page. Green + yellow = HH Goa. Say nothing yet.
2. **Ask in Hindi (voice):** "टेलीफोन का आविष्कार किसने किया?"
   - Watch: live language badge flips to हिन्दी mid-speech; waveform moves; transcript appears word-by-word.
3. **Answer moment:** the assistant SPEAKS back (Edge neural hi-in voice). Watch the voice→voice number — speech-end to first audio, usually ~400-700ms.
   - This is the headline metric. Say it out loud: "it started answering in under half a second, before I finished leaning back."
4. **Barge-in:** while it's still speaking, talk over it — "नहीं नहीं, कौन सा साल था?" — it cuts instantly and listens. Full duplex.
5. **Language switch:** "சென்னை எந்த மாநிலத்தின் தலைநகரம்?" (Tamil). Badge flips, voice switches to a Tamil voice.
6. **Guardrail:** "hello goa how are you?" → clean refusal (not-in-knowledge-base). Shows the agent refuses instead of hallucinating.
7. **Chips fallback:** if the stage mic is noisy, tap the हिन्दी chip — same flow, keyboard-only.
8. **Receipts:** footer "evals →" opens the 13-lang × 4-strategy chunking table + P50/P70/P100 bench on the live pgvector corpus.

## Tips

- Use headphones for barge-in (browser AEC removes speaker bleed, but headphones are bulletproof).
- If mic permission was denied earlier, the chips still demo everything.
- Keep the voice→voice number visible (yellow bar) — it's the flex.
- Repeat queries hit the /tts LRU cache (no credits, instant).

## Failover notes

- TTS: ElevenLabs is primary (needs credits — currently account is 402 out-of-credit), falls back to free Edge neural voices automatically. No demo-path change needed.
- If the WS dies on stage: chips still work (pure HTTP).
- Queries answered from `docs/requirements.md` R3 ≤200ms warm path.
