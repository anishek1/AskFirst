# Clary — Health Pattern Analyst

Ask First AI Intern Assignment. A conversational Streamlit app that reads a patient's full conversation history, ingests each session chronologically with temporal metadata, and surfaces hidden health patterns through streaming causal reasoning.

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

Open `http://localhost:8501`. Select a patient from the sidebar. Clary will ingest sessions one by one, run a full pattern analysis, and then let you chat about the findings.

---

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, `gemini`, `ollama` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model name within the chosen provider |
| `LLM_API_KEY` | — | API key for the provider |
| `LLM_BASE_URL` | — | Optional; for OpenAI-compatible endpoints (NVIDIA NIM, etc.) |
| `LLM_MAX_TOKENS` | `16384` | Max output tokens per reasoning call |
| `LLM_THINKING_ENABLED` | `false` | Enable extended thinking (OpenAI-compatible providers with reasoning_content) |
| `LLM_REASONING_BUDGET` | `16384` | Token budget for reasoning traces (when thinking is enabled) |

**Example — NVIDIA NIM (nemotron with thinking):**
```env
LLM_PROVIDER=openai
LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b
LLM_API_KEY=nvapi-...
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MAX_TOKENS=16384
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

## App UX Flow

```
1. Select patient (Arjun / Meera / Priya) from sidebar
        │
2. Animated session ingestion
   Each session appears chronologically:
   [W01/D000] USR001_S01 · Jan 05 · LATE-NIGHT · sev:moderate · acidity, late_eating
   [W03/D014] USR001_S02 · Jan 19 · AFTERNOON  · sev:mild     · headache, dehydration
   ...
        │
3. Clary streams initial pattern analysis
   ├── Live reasoning trace (thinking tokens, expandable)
   └── Structured pattern cards (confidence scores, session citations, trace steps)
        │
4. Chat input unlocked — ask unlimited follow-up questions
   Full conversation history is sent to the LLM on every turn
   Switch patients → independent chat state is preserved
   Reset button → clears analysis for that patient
```

---

## LLM Choice Rationale

## LLM Choice Rationale

Default: **NVIDIA Nemotron-3-Nano-30B-A3B** (using the `openai` provider wrapper with NVIDIA NIM endpoints).

- **High-Efficiency Reasoning Architecture**: Leverages a 30B MoE hybrid structure (Mamba-2 + Transformer), activating only ~3.5B parameters per token. This provides deep diagnostic reasoning capabilities without the latency overhead of conventional large-scale models.
- **Multi-step temporal logic**: Nemotron's strong logic capabilities allow it to reliably execute the complex 8-step analysis checklist (symptom extraction → trigger identification → delta computation → negative evidence check → resolution check → cascade check → confidence rubric).
- **Extended Context Window**: Each user's full session history fits effortlessly in a single call. No chunking needed—chunking would destroy the cross-session signal required to find Pattern P3 (calorie restriction Jan 8 → hair fall Feb 19, a 42-day gap that falls between chunk boundaries).
- **Thinking & Streaming:** Completely supports extended reasoning ("thinking") modes and token-level streaming, which seamlessly powers the app's live animated causal reasoning trace UI.
- **Multi-turn capability**: Adding `chat_stream(messages)` ensures the full prior conversation is available on every follow-up question — Clary can answer "tell me more about P3" while maintaining logical coherence.

---

## Architecture

```
app.py                     Streamlit UI — session ingestion animation,
                           streaming chat interface, pattern card rendering

src/
  data_loader.py           JSON → Session / User dataclasses (unchanged)
  temporal_engine.py       Pre-computes week numbers, day deltas, time-of-day
                           labels before any LLM call (unchanged)
  chat_engine.py           [NEW] System prompt builder (embeds full temporal
                           timeline), initial analysis prompt, stream_response()
  llm_provider.py          Abstract LLMProvider + Anthropic / OpenAI / Gemini /
                           Ollama backends. All providers now implement both
                           stream_text() (single-turn) and chat_stream()
                           (multi-turn with message history)
  pattern_detector.py      Legacy streaming state machine (kept for reference)
  prompts.py               Legacy prompt templates (kept for reference)

Task/
  askfirst_synthetic_dataset.json   Read-only dataset (3 users, 27 sessions)
```

### Key Design Decisions

**Temporal pre-processing (unchanged from v1)**
Raw timestamps are converted to `[Week N | Day D | YYYY-MM-DD Weekday HH:MM | TIME-OF-DAY]` labels before any LLM call. Pattern P1 (late-night eating → acidity) becomes structurally visible because all affected sessions share the `LATE-NIGHT` label. Pattern P3 (calorie restriction → hair fall at Day 42) is explicit because the headers show "Day 0" and "Day 42" rather than two ISO strings requiring subtraction.

**Timeline in system prompt, not in user turns**
The full chronological timeline is embedded once in the system prompt. This means every follow-up question in the multi-turn conversation automatically has full temporal context without the user having to re-supply data. The `system` key is set when a user is first selected and never changes.

**Session-by-session ingestion animation**
Each session is displayed with its pre-computed temporal label as it is "inserted" into Clary's index (~120ms delay per session). This is a visual representation of the temporal indexing that underpins the reasoning — not just a loading screen.

**Multi-turn conversation state per patient**
`st.session_state.chats[user_id]` holds `system`, `messages` (display), `llm_messages` (LLM wire format), and `done` flag independently for each patient. Switching between Arjun, Meera, and Priya preserves each conversation independently.

**Provider abstraction**
All four providers implement `chat_stream(system, messages, max_tokens)`. `stream_text()` is now a thin wrapper that calls `chat_stream()` with a single-element messages list. This means the same provider code handles both the initial one-shot analysis and the ongoing multi-turn chat.

---

## Output Schema

```json
{
  "user_id": "USR001",
  "user_name": "Arjun",
  "patterns": [
    {
      "pattern_id": "P1",
      "title": "Late dinner (11pm+) → stomach acidity",
      "sessions_involved": ["USR001_S01", "USR001_S04", "USR001_S07", "USR001_S09"],
      "temporal_reasoning": "S01 (Jan 5, Day 0): late dinner 11:30pm → stomach burning same night. S04 (Jan 28, Day 23): late dinner 11pm during deadline → same burning. No acidity in S03 (Jan 19, Day 14) when late eating absent.",
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

---

## Dataset

3 synthetic user profiles · 27 conversations · January–March 2026 · 8 planted hidden patterns.

| User | ID | Sessions | Patterns |
|---|---|---|---|
| Arjun | USR001 | 9 | Late eating → acidity; Dehydration + busy periods → headaches |
| Meera | USR002 | 9 | Calorie restriction → hair fall (6-week delay); Dairy → acne |
| Priya | USR003 | 9 | High-carb lunch → blood sugar crash; Work stress → cramps; Screen use → sleep deprivation → anxiety cascade |
