# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set LLM_API_KEY and LLM_PROVIDER

# Run the app
streamlit run app.py
```

There are no tests or linting configurations in this project.

## Architecture

**Clary** is a Streamlit app that analyzes health patterns from multi-session patient data using LLM reasoning.

```
app.py              Streamlit UI — session ingestion, streaming render, pattern cards,
                    Plotly timeline, HTML/JSON export, cross-patient view, file cache
src/
  data_loader.py    JSON → Session/User dataclasses
  temporal_engine.py  Pre-computes [Week N | Day D | date | TIME-OF-DAY] labels per session
  chat_engine.py    Single-patient + cross-patient prompt builders; stream_response() wrapper
  llm_provider.py   Abstract LLMProvider + Anthropic / OpenAI / Gemini / Ollama backends
.clary_cache/       File-based LLM result cache (gitignored); {uid}_{md5hash}.json per patient
Task/
  askfirst_synthetic_dataset.json  Read-only dataset (3 users, 27 sessions)
```

### Data flow

1. `data_loader.load_dataset()` parses the JSON into `User` + `Session` dataclasses.
2. `temporal_engine.build_temporal_timeline(user)` converts raw timestamps into structured `[Week N | Day D | …]` headers for every session — this pre-labeling runs before any LLM call and is the core of why temporal pattern detection works.
3. The timeline is embedded **once** in the system prompt via `chat_engine.build_system_prompt()`, so every follow-up message in the multi-turn conversation automatically has full temporal context.
4. `llm_provider.get_provider()` reads `LLM_PROVIDER` from `.env` and instantiates the matching backend. All four providers implement `chat_stream(system, messages, max_tokens)` yielding `("thinking"|"content", chunk)` tuples.
5. `app.py` drives everything: animated session ingestion, initial streamed analysis, pattern card rendering, and multi-turn chat — all keyed per patient in `st.session_state.chats[uid]`.

### Provider abstraction rule

**Never hardcode a specific LLM or provider.** All provider selection happens through `.env`. The `LLMProvider` ABC in `src/llm_provider.py` is the only extension point — add a new backend by subclassing it and wiring it into `get_provider()`. Code changes are never needed to switch providers.

### State management

`st.session_state.chats` is a dict keyed by `user_id`. Each entry holds:
- `system` — the full system prompt (built once, never changes mid-session)
- `messages` — display dicts including `role`, `content`, `thinking`, and `type`
- `llm_messages` — wire format sent to the LLM (`[{"role": ..., "content": ...}]`)
- `done` — flag that separates the initial analysis phase from the ongoing chat phase

Switching patients in the sidebar preserves independent state for each patient.

### LLM configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `gemini`, `ollama` |
| `LLM_MODEL` | `nvidia/nemotron-3-nano-30b-a3b` | Model name for the chosen provider |
| `LLM_API_KEY` | — | Required |
| `LLM_BASE_URL` | NVIDIA NIM URL for NVIDIA models | For OpenAI-compatible endpoints |
| `LLM_MAX_TOKENS` | `16384` | Max output tokens |
| `LLM_TEMPERATURE` | `0` | Keep at 0 for stable pattern finding |
| `LLM_TOP_P` | `1` | Nucleus sampling cap |
| `LLM_SEED` | — | Optional; for reproducible outputs |
| `LLM_THINKING_ENABLED` | `false` | Extended thinking via `reasoning_content` |
| `LLM_REASONING_BUDGET` | `16384` | Token budget for reasoning traces |

### Output schema

The initial analysis always requests JSON inside ` ```json ``` ` fences. The schema is defined in `chat_engine.INITIAL_ANALYSIS_TEMPLATE` and validated in `app._parse_pattern_json()`. Required fields per pattern: `pattern_id`, `title`, `sessions_involved`, `temporal_reasoning`, `confidence` (`high`/`medium`/`low`), `confidence_justification`, `trace`.
