# Clary — Execution Writeup & Failure Analysis

This document addresses the mandatory submission requirements, detailing the reasoning approach, context strategy, honest failure analysis, and proposed future improvements.

## 1. Approach to the Reasoning Problem

### No Hardcoded Patterns & Dynamic Generation
Clary relies completely on dynamic analysis of the historical data log. No pre-defined symptom lists or medical rulesets (e.g., "if coffee then heartburn") are hardcoded into the application logic. The logic isolates chronological session data and forces the LLM to identify statistically significant sequences (triggers followed by symptoms) and highlight negative evidence (triggers without symptoms).

### Model Selection & Rationale
**Chosen LLM:** **NVIDIA Nemotron-3-Nano-30B-A3B**
**Why:**
- **Efficiency & Throughput:** Nemotron-3-Nano uses a 30B-parameter hybrid MoE architecture combining Mamba-2 and Transformer layers, but incredibly only activates ~3.5B parameters during token generation. This provides frontier-level reasoning depth with exceptionally low latency, which is essential for a fluid, streaming conversational UX.
- **In-Context Learning & Temporal Scope:** The temporal correlation rules (e.g., checking day-deltas and negative evidence across longitudinal patient timelines) demand intense contextual grasp. Nemotron's massive context window enables it to easily hold the entire 5K+ token chronological history in active memory without degradation.
- **Explicit Reasoning Traces:** Nemotron natively excels at multi-step logic and structured data processing. With extended thinking features enabled, it strictly follows the 8-step reasoning framework, successfully constructing deductive reasoning traces before outputting structured JSON patterns.

### Chunking and Context Management Strategy
**Strategy:** **No Chunking. Single-Pass Full Context.**
- **Why:** Naive RAG or fixed-window chunking breaks long-term temporal dependencies. For instance, Pattern P3 (Meera's calorie restriction → hair fall) requires linking a cause in Session 1 (Jan 8) to an effect in Session 6 (Feb 19). A 42-day gap would inevitably fall across chunk boundaries, effectively blinding the model to the causation. Because all sessions per patient easily fit within a modern context window (~5K tokens), the system passes the *entire* patient chronologic history in a single shot.
- **Context Management in Conversation:** Once the initial timeline is processed, it is embedded permanently in the **System Prompt**. During multi-turn follow-ups, the user's conversational turns are appended, but the full health history remains fixed in the system memory. This ensures the model can answer localized follow-up questions ("When did Priya first report cramps?") without external retrieval mechanisms or hallucinations.

### Temporal Pre-computation & The Reasoning Trace
Raw text limits an LLM's grasp of time. Before any LLM call, `temporal_engine.py` enriches each session with structural time tokens: `[Week X | Day Y | Date | Time-of-Day]`. The model is forced to explicitly output a full **reasoning trace** array before formulating its final confidence score to ensure we see *what* the system considered (e.g., verifying day-deltas and absent symptoms) before it reaches a conclusion.

---

## 2. Where the System Fails or Hallucinates Confidently

A system optimized for pattern discovery naturally biases toward false positives. Here is an honest assessment of current failure modes:

### False Positives on Single-Session Symptoms
Because the system is explicitly prompted to "find patterns," it will occasionally over-index on isolated events. For example, if Arjun reports back pain once after a long day at his desk (S03), the model will sometimes elevate this to a `MEDIUM` confidence pattern. It understands the cause-and-effect plausibility but fails to apply the strict statistical rigor (e.g., "this only happened once") without explicit code-level suppression.

### Conflating Independent Health Drivers (Priya's Cramps)
Pattern P8 relies on identifying sleep deprivation as a completely independent driver of menstrual cramps, separate from work stress (P6). The LLM often merges these into a single "Stress + Poor Sleep → Cramps" meta-pattern, hallucinating that they are inextricably linked despite data explicitly separating them (e.g., in Session 9). The system struggles to run controlled-variable comparisons across a dense session history.

### Misinterpreting Cascade Temporal Delays (Telogen Effluvium)
Medical literature dictates a 6–12 week delay for stress-induced hair loss. Meera's hair fall at Day 42 sits exactly at the 6-week boundary. The model will sometimes confidently claim the timeline is "too short" to be the root cause and downgrade the confidence improperly, or conversely, ignore the biological delay completely and treat it as a coincidental correlation.

---

## 3. What Would Be Built Differently With More Time

1. **Multi-Pass Reasoning (Agentic Cascades):** Instead of a single-shot extraction, implement an agentic workflow where LLM Pass 1 extracts isolated symptom/trigger pairs, and LLM Pass 2 is solely tasked with constructing the causal chain/cascades from the isolated pairs. This would resolve the P8 conflation issue.
2. **Native Structured Output APIs:** Currently, JSON extraction involves string parsing (`_extract_json`). With more time, I would migrate all provider endpoints to use native structured output enforcements (e.g., OpenAI `response_format` or Anthropic `tool_use`) to completely eliminate the risk of JSON parsing crashes due to hallucinated markdown.
3. **Controlled-Variable Prompt Architecture:** Introduce a mandatory "Negative Space" reasoning requirement where the model must explicitly locate a session where the trigger was *absent* before it is allowed to assign a `HIGH` confidence badge.
4. **Automated Evaluation Harness:** Develop a ground-truth eval script measuring recall vs. precision against the 8 known, implanted patterns across the 3 synthetic users to iterate on the system prompt systematically rather than anecdotally.
