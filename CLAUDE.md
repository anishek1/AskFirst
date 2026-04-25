# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI intern assignment for **Ask First**, a health clarity platform. The goal is to build a reasoning system called **Clary** that detects hidden health patterns across a user's conversation history with temporal awareness.

The dataset lives in `Task/askfirst_synthetic_dataset.json` — 3 synthetic user profiles (Arjun, Meera, Priya), 27 total conversations spanning January–March 2026, with **8 hidden patterns** planted across the profiles.

## What to Build

A runnable app (Streamlit preferred for demo clarity) that does two things:

1. **Cross-conversation pattern detection** — reason across sessions with time-awareness. A symptom 10 weeks *after* a lifestyle change is causally different from one 10 weeks *before* it.
2. **Confidence-scored output in JSON** — each pattern must include a confidence level (`high/medium/low`) with a one-line justification. All responses must support streaming.

Output must be **JSON format only** for pattern results — no prose summaries.

## Running the App

Once built, the app will be run with:
```bash
streamlit run app.py
# or
python main.py  # if CLI-based
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Architecture Decisions to Make

**LLM choice**: Must be documented with rationale. NVIDIA Nemotron-3-Nano-30B-A3B (via `openai` compatible SDK for NVIDIA NIM) is a strong default given the multi-step reasoning efficiency, MoE architecture, and structured output needs.
**Context management strategy** (critical — graded equally with output quality):
- The full conversation history per user fits in a single context window, but naive concatenation loses temporal signal.
- Preferred approach: structured chronological summaries with explicit timestamps before each LLM reasoning call, not raw message dumps.
- Each pattern detection call should include: session ID, timestamp, key symptoms/events extracted, and relative week number.

**Chunking strategy**:
- Per-user: all sessions in one reasoning pass (dataset is small enough)
- Temporal indexing: pre-process sessions into `(week_number, event_type, detail)` tuples before sending to LLM

**Reasoning trace**: Must be shown — log intermediate reasoning steps, not just the final pattern. A `trace` field alongside each pattern in the JSON output satisfies this.

## Dataset Structure

```
users[].conversations[].{
  session_id, timestamp, user_message,
  clary_questions, user_followup,
  clary_response, severity, tags
}
```

Key fields for temporal reasoning: `timestamp` (ISO 8601), `tags` (pre-labeled categories), `severity`.

## Known Patterns in Dataset (for validation)

These are the planted patterns to find — use these to validate detection coverage:

| # | User | Pattern |
|---|------|---------|
| 1 | Arjun (USR001) | Late-night eating during work deadlines → recurring stomach acidity (Sessions 1, 4, 7, 9) |
| 2 | Arjun (USR001) | Busy work periods + low water intake → afternoon headaches (Sessions 2, 6, 8) |
| 3 | Meera (USR002) | Severe calorie restriction (Jan 8, ~700 cal/day) → hair fall 6–7 weeks later (Session 6, Feb 19) — telogen effluvium delay |
| 4 | Meera (USR002) | High dairy intake → cheek/jawline acne flares; confirmed by elimination and reintroduction |
| 5 | Priya (USR003) | High-carb lunch (rice-heavy) → afternoon blood sugar crash at 2–4pm (Sessions 1, 3); resolved with protein addition (Session 6) |
| 6 | Priya (USR003) | Elevated work stress → worsened pre-menstrual cramps (Sessions 2, 5) |
| 7 | Priya (USR003) | Chronic late-night screen use (7+ weeks) → sleep deprivation → baseline anxiety (Session 7) |
| 8 | Priya (USR003) | Sleep deprivation (independent of work stress) → severe cramps — confirmed in Session 9 when stress was low but sleep still poor |

## Output Schema (required)

```json
{
  "user_id": "USR001",
  "patterns": [
    {
      "pattern_id": "P1",
      "title": "Late eating → stomach acidity",
      "sessions_involved": ["USR001_S01", "USR001_S04", "USR001_S07", "USR001_S09"],
      "temporal_reasoning": "...",
      "confidence": "high",
      "confidence_justification": "Pain in 4 sessions, all within 12h of late dinner during deadline; no pain in sessions without late eating",
      "trace": ["step 1: ...", "step 2: ..."]
    }
  ]
}
```

## Submission Requirements

- GitHub repo with working code
- `README.md` covering setup and how to run
- One-page writeup (can be `writeup.md`) covering:
  1. Approach to the reasoning problem
  2. Where the system fails or hallucinates — honest failure analysis is weighted as heavily as finding patterns
