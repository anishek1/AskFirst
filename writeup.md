# Clary - Execution Writeup & Failure Analysis

This document addresses the mandatory submission requirements, detailing the reasoning approach, context strategy, honest failure analysis, and proposed future improvements.

Live app: https://askfirst-clary.streamlit.app/

## 1. Approach to the Reasoning Problem

### No Hardcoded Patterns & Dynamic Generation
Clary relies completely on dynamic analysis of the historical data log. No pre-defined symptom lists or medical rulesets are hardcoded into the application logic. The logic isolates chronological session data and forces the LLM to identify statistically significant sequences, such as triggers followed by symptoms, and to highlight negative evidence, such as triggers without symptoms.

### Accuracy Controls for Stable Pattern Finding
To keep pattern finding repeatable, the app uses deterministic decoding defaults (`temperature=0`, `top_p=1`, optional seed support where available) and a stricter evidence gate in the prompt. In practice, this means Clary prefers fewer but more defensible patterns over speculative ones. A candidate only survives if it has temporal direction, recurrence, and supporting evidence such as negative evidence or intervention response.

### Model Selection & Rationale
**Chosen LLM:** NVIDIA Nemotron-3-Nano-30B-A3B

Why:
- **Efficiency & Throughput:** Nemotron-3-Nano uses a 30B-parameter hybrid MoE architecture combining Mamba-2 and Transformer layers, but only activates about 3.5B parameters during token generation. That gives strong reasoning depth without high latency, which suits a streaming conversational UX.
- **In-Context Learning & Temporal Scope:** The temporal correlation rules, such as checking day deltas and negative evidence across longitudinal patient timelines, require strong context handling. Nemotron's context window is sufficient for the full per-user history.
- **Explicit Reasoning Traces:** Nemotron is used with streaming and extended thinking support so the UI can surface a live reasoning trace. The final pattern set is still constrained by deterministic decoding and post-hoc evidence gates, which is what keeps the output stable.

### Chunking and Context Management Strategy
**Strategy:** No chunking. Single-pass full context.

- **Why:** Naive RAG or fixed-window chunking breaks long-term temporal dependencies. For example, Pattern P3 in Meera's timeline requires linking a cause in Session 1 (Jan 8) to an effect in Session 6 (Feb 19). A 42-day gap would be easy to miss if the history were split across chunks. Because all sessions per patient fit within a modern context window, the system passes the entire chronological history in a single shot.
- **Context Management in Conversation:** Once the initial timeline is processed, it is embedded permanently in the system prompt. During multi-turn follow-ups, the user's conversational turns are appended, but the full health history remains fixed in the system memory. That lets the model answer localized follow-up questions without external retrieval.

### Temporal Pre-computation & The Reasoning Trace
Raw text limits an LLM's grasp of time. Before any LLM call, `temporal_engine.py` enriches each session with structural time tokens: `[Week X | Day Y | Date | Time-of-Day]`. The model is asked to output a full reasoning trace array before the final confidence score, but the app does not trust that trace by itself. The output is still filtered through the confidence rubric and evidence rules so that weak or duplicate patterns do not surface as final findings.

---

## 2. Where the System Fails or Hallucinates Confidently

A system optimized for pattern discovery naturally biases toward false positives. Here is an honest assessment of current failure modes:

### False Positives on Single-Session Symptoms
The main remaining failure mode is overfitting to a plausible but weak connection. The code now suppresses most of this by requiring stronger evidence, but a model can still propose a borderline low-confidence pattern if the timeline is only loosely supportive. That is the right tradeoff for the assignment because it keeps recall acceptable without letting every symptom mention become a pattern.

### Conflating Independent Health Drivers (Priya's Cramps)
Pattern P8 relies on identifying sleep deprivation as a completely independent driver of menstrual cramps, separate from work stress (P6). The system can still sometimes merge them into a broader stress-related explanation because both are correlated in the timeline. The current prompt reduces that tendency by asking the model to prefer one root-cause cascade over duplicated sub-patterns, but this remains a subtle reasoning edge case.

### Misinterpreting Cascade Temporal Delays (Telogen Effluvium)
Medical literature dictates a 6-12 week delay for stress-induced hair loss. Meera's hair fall at Day 42 sits exactly at the 6-week boundary. That is a useful borderline case because it shows why the app needs temporal labels plus an evidence gate, not keyword matching. The model still has judgment error risk at the boundary, so I treat this as a medically plausible but not perfectly certain cascade.

---

## 3. What Would Be Built Differently With More Time

1. **Multi-Pass Reasoning (Agentic Cascades):** Instead of a single-shot extraction, implement an agentic workflow where LLM Pass 1 extracts isolated symptom/trigger pairs, and LLM Pass 2 is solely tasked with constructing the causal chain/cascades from the isolated pairs. This would further reduce borderline merges like P6/P8.
2. **Native Structured Output APIs:** Currently, JSON extraction still uses string parsing (`_extract_json`). With more time, I would migrate all provider endpoints to native structured output enforcement where supported to reduce formatting failures.
3. **Automated Evaluation Harness:** Develop a ground-truth eval script measuring recall vs. precision against the 8 known, implanted patterns across the 3 synthetic users so prompt and threshold changes are judged quantitatively instead of by inspection.

