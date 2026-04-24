# Clary — Approach and Failure Analysis

## Approach to the Reasoning Problem

**The core challenge** is that raw conversation text, naively concatenated, loses temporal signal. The sentence "my stomach hurt this week" in January means something different from the same sentence in March, and the gap matters when looking for delayed effects like telogen effluvium (calorie restriction → hair fall at 6 weeks).

### Pre-computed Temporal Metadata

Before any LLM call, `temporal_engine.py` converts each session timestamp into four pre-computed fields:
- **Week number** (1-indexed from user's first session)
- **Day delta** (calendar days since first session)
- **Absolute date and weekday**
- **Time-of-day label**: `LATE-NIGHT (11pm–4am)`, `MORNING`, `AFTERNOON`, `EVENING`

This means the LLM receives headers like `[Week 1 | Day 0 | 2026-01-05 Mon 23:14 | LATE-NIGHT]` rather than ISO timestamps. Pattern P1 (late-night eating → acidity) becomes structurally visible because all four acidity sessions share the `LATE-NIGHT` label. Pattern P3 (calorie restriction → hair fall) becomes explicit because the header shows "Day 0" and "Day 45" rather than two ISO strings requiring subtraction.

### Two-Phase Streaming

The system prompt instructs Claude to complete a 9-step reasoning chain first, then output JSON. The state machine in `pattern_detector.py` detects the ` ```json ` fence to split the stream into a live reasoning display and a final pattern card render. This satisfies the `trace` field requirement without a separate API call.

### No Chunking

All sessions per user fit in a single context window call (~5K tokens). Chunking would sever the P3 connection (Jan 8 diary → Feb 19 hair fall) because the two sessions would land in different chunks. The "chunking" decision is made at the extraction layer instead: structured timeline tuples replace raw message dumps.

### Negative Evidence Instruction

The system prompt explicitly asks for "sessions where trigger was present but symptom absent, or trigger absent and symptom absent." This is the mechanism that pushes P1 and P2 to `high` confidence — Claude can observe that Arjun's non-deadline sessions show no acidity and non-busy-week sessions show no headaches.

---

## Where the System Fails or Hallucinates

### P3 vs P8 Conflation (Meera)
P3 is "calorie restriction → hair fall at Day 41" and P8 is "calorie restriction → cascade (Day 1 dizziness → Week 5 fatigue/brain fog → Week 6 hair fall)." These share the same root cause and terminal symptom. Claude may output them as one merged pattern or omit the cascade framing of P8 entirely. The distinction requires recognizing that dizziness (Day 1), fatigue (Week 5), and hair fall (Week 6) are three separate downstream effects of the same trigger at different biological delays — a reasoning pattern that requires holding the root cause in mind across multiple sessions simultaneously.

### Telogen Effluvium Timeline
The dataset shows calorie restriction starting Jan 8 (Day 0) and hair fall first reported Feb 19 (Day 42). Medical literature puts telogen effluvium onset at 6–12 weeks (42–84 days), so Day 42 is at the early edge. The LLM may either accept the dataset's framing uncritically or incorrectly flag the 6-week delay as too short. The system does not inject medical knowledge, so this is a known blind spot.

### P6b — Sleep as Independent Cramp Driver (Priya)
Pattern P6b requires distinguishing two drivers of Priya's menstrual cramps: work stress (P6, confirmed across two months) and sleep deprivation as an *independent* driver (confirmed in Session 9 where stress was explicitly low but sleep was still poor and cramps were severe). This requires the model to hold a controlled-variable comparison across the full session history. In practice, Claude may conflate P6 and P6b into a single "stress + sleep → cramps" pattern, losing the independence claim that makes P6b a separate finding.

### P7 Cascade Ordering (Priya)
Pattern P7 is a three-step cascade: late-night screens (mid-Jan) → sleep deprivation → all-day fatigue (Week 5) → anxiety (Week 7) → worsened cramps (Month 3). The cascade only makes sense if the model correctly sequences five symptoms across seven weeks and identifies screens as the single upstream cause. The model may detect the individual symptoms without connecting them into a chain, outputting three separate `LOW` confidence patterns instead of one `HIGH` confidence cascade.

### False Positives
The system has no hardcoded pattern suppression. On Arjun's sessions, back pain (S03) and fatigue (S05) appear once each with plausible but weak triggers (sedentary work, late bedtime). Claude may elevate these to `MEDIUM` confidence patterns. The confidence rubric (3+ episodes = HIGH, 2 = MEDIUM, 1 = LOW) is stated in the prompt but is not enforced mechanically — it relies on the LLM self-applying the rule correctly.

### JSON Parsing Robustness
The state machine uses string search for ` ```json ` and ` ``` ` delimiters. If the model outputs the JSON block without fences (rare but possible), or inserts a code comment inside the block, the parser will either return the full raw text as an error or silently truncate. This is a known gap — a fallback `json.loads` pass on the full accumulated text would recover these cases but is not currently implemented.
