# Demo script — निशा · वॉइस RAG (HH Goa 2026)

Live URL: https://meridian.dpdns.org — deploy via SSM (ssh unreliable from CGNAT).

Persona: **निशा · Goa desk**. The arc: **"Nisha's living memory"** — she learns during the demo,
forgets on command, and her memory is live on every device that opens the QR.

## 60-second flow

1. **Open the page.** Greeting "निशा बोलत नाही? बोला". Note गोवा brain selector.
2. **Goa brain:** ask "गोवा का सबसे पुराना चर्च कौन सा है?" → cited answer, exact sentence highlighted.
3. **THE MOMENT — teach her:** type in the remember box:
   `मोरजिम बीच पर हर साल ओलिव रिडले कछुओं का festival होता है` → **याद रखो**
   → her tone: "याद रखा · live:172 · memory is live everywhere — teach here, ask on the phone"
4. **Prove it by voice:** ask "मोरजिम में कछुए कब आते हैं?" → she answers **the fact you just taught her**,
   cited `live:172` — a knowledge-base update with zero rebuild, live in front of the judges.
5. **Unlearn:** tap 🧹 भूल जाओ → ask again → the taught fact is gone, corpus passage answers instead.
   *"She forgets on command — machine unlearning, live."*
6. **Konkani move:** chip `कोंकणी · कसो आसा?` (goa brain).
7. **Cross-lingual flex:** chip `हिन्दी → தமிழ்` — Hindi question, Tamil answer + Tamil voice.
8. **Full duplex:** talk over her while she speaks — she cuts and re-listens (headphones advised).
9. **Guard theatre:** chip `guard: मुझे मारना है` → refusal + trace `toxic / blocklist` on screen.
10. **Evaluation IS the demo:** ⚡ bench it now — live P50/P70/P100 bars on screen (warm ~15-70ms).
11. **Nisha interviews YOU:** 🎤 निशा पूछती है — name → language → Goa place; she repeats it back.
12. **Report card:** 🧾 report — spoken summary of the whole session: questions, languages, sources,
    refusals, remembered/forgotten. The demo documents itself.
13. **Phone follow-up:** 📱 scan me → judge asks the same "मोरजिम" question *after* you taught it live
    (memory is in RDS, not the laptop).

## Receipts on stage
- Footer `evals →`: 13-lang × 4-strategy chunking table + bench P50/P70/P100.
- Stage rings + HUD show lid/embed/retrieve/extract live ms; voice→voice number under the answer.
- `X-TTS-Source` header: eleven → **sarvam** (active, Indian voice) → edge. Your Sarvam key is wired.

## Tips
- Warm the box: after any deploy hit `/health`, then fire `/bench` once (first run cold).
- Repeat /tts calls are LRU-cached — zero credits, instant.
- Mic dies on stage? Every chip is pure HTTP and the remember box works without the mic.
- The ledger bar (सवाल/भाषाएँ/स्रोत/refused/याद/भूल) grows through the whole demo — glance at it often.
