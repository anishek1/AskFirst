# Clary — Health Pattern Detector

Ask First AI Intern Assignment. Detects hidden health patterns across a user's conversation history with explicit temporal reasoning and confidence-scored JSON output.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set LLM_API_KEY
```

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Click any user button. The reasoning trace streams live, followed by pattern cards with confidence scores and session citations.

## Configuration

All LLM settings are in `.env`:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, `gemini`, `ollama` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model name within the provider |
| `LLM_API_KEY` | — | API key for the chosen provider |
| `LLM_BASE_URL` | — | Optional; for OpenAI-compatible endpoints |
| `LLM_MAX_TOKENS` | `8000` | Max output tokens per analysis call |

To use a different provider, change `LLM_PROVIDER` and set the matching key — no code changes needed.

## LLM Choice Rationale

Default: **Claude claude-sonnet-4-6** (`anthropic` provider).

- Multi-step temporal reasoning: the 9-step reasoning chain (symptom extraction → trigger identification → delta computation → negative evidence check) benefits from a model that follows structured instructions reliably.
- Large context window: each user's full session history (~5K tokens) fits in one call. No chunking needed, which preserves cross-session signal (e.g., Pattern P3: calorie restriction Jan 8 → hair fall Feb 19, a 41-day gap that chunking would destroy).
- Native streaming: `client.messages.stream()` yields text chunks for live display.
- Structured JSON output: Claude reliably wraps output in ` ```json ``` ` fences as instructed.

## Architecture

```
app.py                    Streamlit UI — two-phase streaming display
src/
  data_loader.py          JSON → Session/User dataclasses
  temporal_engine.py      Pre-computes week numbers, day deltas, time-of-day labels
  prompts.py              System prompt + 9-step reasoning template + JSON schema
  llm_provider.py         Abstract LLMProvider + Anthropic / OpenAI / Gemini / Ollama
  pattern_detector.py     Streaming state machine (REASONING → JSON_BUFFERING → DONE)
Task/
  askfirst_synthetic_dataset.json   Read-only dataset
```

**Key design decision — temporal pre-processing**: Raw timestamps are converted to `[Week N | Day D | YYYY-MM-DD | TIME-OF-DAY]` labels before being sent to the LLM. This makes patterns like "late-night eating correlates with acidity" immediately readable without requiring the model to do date arithmetic mid-reasoning.

## Output Schema

```json
{
  "user_id": "USR001",
  "user_name": "Arjun",
  "patterns": [
    {
      "pattern_id": "P1",
      "title": "Late dinner (11pm+) -> stomach acidity",
      "sessions_involved": ["USR001_S01", "USR001_S04", "USR001_S07", "USR001_S09"],
      "temporal_reasoning": "...",
      "confidence": "high",
      "confidence_justification": "4 episodes all preceded by late dinner; acidity absent in sessions without late eating",
      "trace": ["Step 1: ...", "..."]
    }
  ]
}
```
