# Clary — Health Pattern Analyst

Ask First AI Intern Assignment. A conversational Streamlit app that reads a patient's full conversation history, ingests each session chronologically with temporal metadata, and surfaces hidden health patterns through streaming causal reasoning.

Live app: https://askfirst-clary.streamlit.app/

---

## Assignment Coverage

- Streamlit interface with per-patient selection and a cross-patient population view.
- Session-by-session ingestion shown chronologically with week/day/time labels.
- Full temporal history embedded once in the system prompt; every follow-up retains full context.
- Streaming chat for both initial analysis and unlimited follow-up questions.
- JSON pattern output with confidence scores and one-line justifications.
- HTML report export — self-contained, dark-themed, shareable without the app running.
- Reasoning trace visible in the UI (expandable).
- File-based analysis cache — keyed by session hash, survives full page reloads.
- Plotly pattern timeline — scatter chart showing each pattern's session spread over time.
- Cross-patient population view — finds patterns shared across 2+ patients with a dedicated LLM call.
- No hardcoded pattern rules; patterns are generated dynamically from the user timeline.
- NVIDIA Nemotron via OpenAI-compatible NVIDIA NIM endpoints (default; any provider works).

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set your API key
```

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Select a patient or **🔬 All Patients** from the sidebar.

Hosted deployment: https://askfirst-clary.streamlit.app/

---

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `gemini`, `ollama` |
| `LLM_MODEL` | `nvidia/nemotron-3-nano-30b-a3b` | Model name within the chosen provider |
| `LLM_API_KEY` | — | API key for the provider |
| `LLM_BASE_URL` | NVIDIA NIM URL for NVIDIA models | Optional; for OpenAI-compatible endpoints |
| `LLM_MAX_TOKENS` | `16384` | Max output tokens per reasoning call |
| `LLM_TEMPERATURE` | `0` | Deterministic decoding for stable pattern finding |
| `LLM_TOP_P` | `1` | Nucleus sampling cap; keep `1` with temperature `0` |
| `LLM_SEED` | — | Optional seed for reproducible outputs |
| `LLM_THINKING_ENABLED` | `false` | Enable extended thinking (OpenAI-compatible providers with reasoning_content) |
| `LLM_REASONING_BUDGET` | `16384` | Token budget for reasoning traces (when thinking is enabled) |

**Example — NVIDIA NIM (nemotron with thinking):**
```env
LLM_PROVIDER=openai
LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b
LLM_API_KEY=nvapi-...
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MAX_TOKENS=16384
LLM_TEMPERATURE=0
LLM_TOP_P=1
LLM_SEED=
LLM_THINKING_ENABLED=true
LLM_REASONING_BUDGET=16384
```

**Example — Anthropic Claude:**
```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
LLM_API_KEY=sk-ant-...
```

No code changes are needed when switching providers — only `.env` changes.

---

## Demo Flow

### Single patient
1. Select Arjun, Meera, or Priya from the sidebar.
2. Watch sessions ingest in chronological order with temporal labels.
3. Review Clary's streamed pattern analysis, confidence scores, reasoning traces, and cited sessions.
4. Expand **📅 Pattern Timeline** to see each pattern's sessions plotted over time.
5. Download the generated JSON or HTML report, or regenerate if needed.
6. Ask follow-up questions; the full temporal history is available for the entire conversation.

### Population analysis
1. Click **🔬 All Patients** in the sidebar.
2. All three patients' sessions are ingested in sequence.
3. Clary runs a cross-patient analysis looking for patterns shared across 2+ users.
4. Cross-patient pattern cards show `users_affected`, temporal reasoning per patient, and a combined timeline coloured by patient.
5. Download the population JSON or HTML report.
6. Ask cross-patient follow-up questions.

Both modes cache their analysis to `.clary_cache/` — results survive page reloads without re-running the LLM.

---

## App UX Flow

```
Sidebar
  ├── Arjun / Meera / Priya  →  Single-patient mode
  └── 🔬 All Patients         →  Cross-patient population mode

Single-patient mode
  1. Animated session ingestion (W/D labels)
  2. Stream initial pattern analysis
     ├── Live reasoning trace (expandable)
     ├── Pattern cards (confidence, sessions, trace steps)
     └── 📅 Pattern Timeline (Plotly scatter, one row per pattern)
  3. ⬇ Download JSON | ⬇ Download HTML | ↺ Regenerate
  4. Chat input unlocked — unlimited follow-up questions
     Switch patients → independent chat state preserved

Cross-patient mode
  1. Animated ingestion for all patients in sequence
  2. Stream cross-patient analysis
     ├── CP pattern cards with users_affected badges
     └── 📅 Cross-Patient Timeline (dots coloured by patient)
  3. ⬇ Download JSON | ⬇ Download HTML
  4. Chat input for population-level follow-up questions
```

---

## LLM Choice Rationale

Default: **NVIDIA Nemotron-3-Nano-30B-A3B** (using the `openai` provider wrapper with NVIDIA NIM endpoints).

- **High-Efficiency Reasoning Architecture**: Leverages a 30B MoE hybrid structure (Mamba-2 + Transformer), activating only ~3.5B parameters per token. This provides deep diagnostic reasoning capabilities without the latency overhead of conventional large-scale models.
- **Multi-step temporal logic**: Nemotron's strong logic capabilities allow it to reliably execute the complex 8-step analysis checklist (symptom extraction → trigger identification → delta computation → negative evidence check → resolution check → cascade check → confidence rubric).
- **Extended Context Window**: Each user's full session history fits effortlessly in a single call. No chunking needed — chunking would destroy the cross-session signal required to find Pattern P3 (calorie restriction Jan 8 → hair fall Feb 19, a 42-day gap that falls between chunk boundaries).
- **Thinking & Streaming**: Completely supports extended reasoning ("thinking") modes and token-level streaming, which seamlessly powers the app's live animated causal reasoning trace UI.
- **Multi-turn capability**: The full prior conversation is available on every follow-up question — Clary can answer "tell me more about P3" while maintaining logical coherence.

---

## Architecture

```
app.py                     Streamlit UI — session ingestion animation,
                           streaming chat, pattern cards, Plotly timeline,
                           HTML/JSON export, cross-patient view,
                           file-based analysis cache

src/
  data_loader.py           JSON → Session / User dataclasses
  temporal_engine.py       Pre-computes week numbers, day deltas, time-of-day
                           labels before any LLM call
  chat_engine.py           System prompt builders (single-patient and
                           cross-patient), initial analysis prompts,
                           stream_response() wrapper
  llm_provider.py          Abstract LLMProvider + Anthropic / OpenAI / Gemini /
                           Ollama backends. All expose chat_stream()

.clary_cache/              File-based LLM result cache (gitignored).
                           Keyed by {uid}_{md5(session_ids+timestamps)}.json

Task/
  askfirst_synthetic_dataset.json   Read-only dataset (3 users, 27 sessions)
```

### Key Design Decisions

**Temporal pre-processing**
Raw timestamps are converted to `[Week N | Day D | YYYY-MM-DD Weekday HH:MM | TIME-OF-DAY]` labels before any LLM call. Pattern P1 (late-night eating → acidity) becomes structurally visible because all affected sessions share the `LATE-NIGHT` label. Pattern P3 (calorie restriction → hair fall at Day 42) is explicit because the headers show "Day 0" and "Day 42" rather than two ISO strings requiring subtraction.

**Timeline in system prompt, not in user turns**
The full chronological timeline is embedded once in the system prompt. Every follow-up question in the multi-turn conversation automatically has full temporal context without the user re-supplying data.

**File-based analysis cache**
The LLM result for each patient (and the population view) is saved to `.clary_cache/{uid}_{hash}.json` after the first run. The hash is computed from session IDs + timestamps, so it invalidates automatically if the dataset changes. On page reload the cache is checked before any animation runs.

**Multi-turn conversation state per patient**
`st.session_state.chats[user_id]` holds `system`, `messages`, `llm_messages`, and `done` independently for each patient. Switching between Arjun, Meera, Priya, and All Patients preserves each conversation independently.

**Provider abstraction**
All four providers implement `chat_stream(system, messages, max_tokens)`. The same provider code handles the initial one-shot analysis, the cross-patient analysis, and ongoing multi-turn chat. Switching providers requires only `.env` changes.

---

## Output Schema

### Single-patient patterns
```json
{
  "user_id": "USR001",
  "user_name": "Arjun",
  "patterns": [
    {
      "pattern_id": "P1",
      "title": "Late dinner (11pm+) → stomach acidity",
      "sessions_involved": ["USR001_S01", "USR001_S04", "USR001_S07", "USR001_S09"],
      "temporal_reasoning": "S01 (Jan 5, Day 0): late dinner 11:30pm → stomach burning same night.",
      "confidence": "high",
      "confidence_justification": "4 episodes all preceded by late dinner; acidity absent in 5 sessions without late eating",
      "trace": [
        "Symptom 'stomach burning' first appears in S01 (Jan 5, Day 0)",
        "Trigger 'late dinner 11pm+' present in S01, S04, S07, S09",
        "No acidity in 5 sessions without late eating — strong negative evidence"
      ]
    }
  ]
}
```

### Cross-patient patterns
```json
{
  "analysis_type": "cross_patient",
  "users_analyzed": ["USR001", "USR002", "USR003"],
  "cross_patient_patterns": [
    {
      "pattern_id": "CP1",
      "title": "Work stress → physical cramps",
      "users_affected": ["USR002 (Meera)", "USR003 (Priya)"],
      "sessions_involved": ["USR002_S03", "USR002_S07", "USR003_S02", "USR003_S05"],
      "temporal_reasoning": "Meera S03 (Jan 22, Day 17): work deadline → cramps same day. Priya S02 (Jan 15, Day 10): project deadline → cramps within 2 days.",
      "confidence": "medium",
      "confidence_justification": "2 patients, each with 2+ consistent episodes",
      "trace": [
        "Meera: work stress in S03, S07 → cramps within 0–1 days each time",
        "Priya: work deadline in S02, S05 → cramps within 0–2 days"
      ]
    }
  ]
}
```

---

## Dataset

3 synthetic user profiles · 27 conversations · January–March 2026 · 8 planted hidden patterns.

| User | ID | Sessions | Patterns |
|---|---|---|---|
| Arjun | USR001 | 9 | Late eating → acidity; Dehydration + busy periods → headaches |
| Meera | USR002 | 9 | Calorie restriction → hair fall (6-week delay); Dairy → acne |
| Priya | USR003 | 9 | High-carb lunch → blood sugar crash; Work stress → cramps; Screen use → sleep deprivation → anxiety cascade |
