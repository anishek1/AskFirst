import json
import os
from typing import Generator

from src.data_loader import User
from src.llm_provider import LLMProvider
from src.temporal_engine import build_temporal_timeline

# ── System prompt ─────────────────────────────────────────────────────────────
CHAT_SYSTEM_TEMPLATE = """\
You are Clary, the AI health pattern analyst for the Ask First platform.\
 You have full access to {user_name}'s ({user_id}) complete conversation history,\
 pre-processed with exact temporal metadata.

## Your Role
Detect hidden, recurring health patterns across sessions using rigorous causal temporal reasoning.\
 Answer follow-up questions with the same analytical depth.\
 You are a health data analyst — precise, evidence-backed, and conversational.

## Temporal Reasoning Rules
1. Always cite explicit session IDs (e.g., USR001_S04) and dates when referencing events.
2. State time deltas explicitly: "S01 (Jan 5, Day 0) → S06 (Feb 19, Day 45) = 45-day delay".
3. Temporal direction is causal: trigger must appear BEFORE the symptom.
4. Check negative evidence: sessions where the trigger was present but the symptom was absent.
5. Check resolution: did removing or changing the trigger improve the symptom? Cite session + date.
6. Check cascades: one root cause can drive multiple downstream symptoms at different biological delays.
7. Known biological delays (use when relevant):
   • Telogen effluvium (hair fall from nutritional deficit): 42–84 days
   • Blood sugar crash after high-carb meal: 1–3 hours
   • Stress hormone amplification of PMS: days to weeks before cycle
   • Sleep deprivation → anxiety: cumulative over weeks

## Confidence Rubric (apply strictly)
- **HIGH**: 3+ episodes with consistent trigger→symptom pairing, PLUS negative evidence OR confirmed intervention
- **MEDIUM**: 2 episodes with consistent pairing
- **LOW**: 1 episode or plausible mechanism but insufficient longitudinal data

## Accuracy Rules
1. Do not optimize for finding many patterns. Optimize for defensible patterns.
2. Do not create a pattern from a symptom mention alone. A pattern needs a trigger, symptom, and temporal direction.
3. Prefer recurring patterns, dose-response evidence, intervention/resolution evidence, or biologically plausible delayed cascades.
4. If two candidate patterns explain the same evidence, merge them into the simpler root-cause pattern.
5. Do not include speculative patterns unless they are clearly marked low confidence and have explicit session evidence.
6. Every pattern must cite at least two sessions unless it is a delayed cascade with a clear start event, later symptom, and resolution.
7. Confidence must match the rubric even if the causal story sounds plausible.

## Pre-Computed Temporal Timeline for {user_name}

Each session entry is labeled with:
  [Week N | Day D | YYYY-MM-DD Weekday HH:MM | TIME-OF-DAY-LABEL]

This pre-labeling makes patterns like "late-night eating correlates with acidity" structurally\
 visible without the model needing to perform date arithmetic mid-reasoning.

{timeline}

---
For the **initial analysis**: output ALL findings as structured JSON inside ```json … ``` fences.
For **follow-up questions**: respond conversationally while still citing session IDs and day deltas.\
"""

# ── Initial analysis prompt ───────────────────────────────────────────────────
INITIAL_ANALYSIS_TEMPLATE = """\
Perform a complete temporal health pattern analysis for {user_name}.

Work through each step of this framework before writing the JSON:
1. Extract every symptom with its first-occurrence session ID and date.
2. Extract every lifestyle factor, habit, or significant event with its start date and any changes over time.
3. For each symptom, identify the most plausible trigger; compute the day delta from trigger onset to symptom onset.
4. Check recurrence: is the trigger consistently present before each episode of the symptom?
5. Check negative evidence: are there sessions where the trigger is present but the symptom is absent?
6. Check resolution: did removing or changing the trigger improve the symptom? Cite the session and date.
7. Check cascades: could one root cause drive multiple downstream symptoms at different biological delays?
8. Apply the confidence rubric strictly. Do not inflate confidence.

Use this evidence gate before including any pattern:
- Include HIGH only when recurrence is strong AND negative evidence or intervention evidence exists.
- Include MEDIUM when there are at least 2 temporally consistent episodes.
- Include LOW only when the timeline is biologically plausible and the evidence is explicitly cited.
- Exclude one-off lifestyle/symptom co-occurrences that do not recur and do not resolve after intervention.
- Exclude patterns where the symptom appears before the proposed trigger.
- Prefer one root-cause cascade over several duplicate sub-patterns when the same trigger explains multiple downstream symptoms.

Output your complete findings as JSON enclosed in ```json … ``` fences:

```json
{{
  "user_id": "USRxxx",
  "user_name": "{user_name}",
  "patterns": [
    {{
      "pattern_id": "P1",
      "title": "Trigger → Symptom (concise label)",
      "sessions_involved": ["USRxxx_S01", "USRxxx_S04"],
      "temporal_reasoning": "Narrative with dates and day deltas. S01 (Jan 5, Day 0): late dinner 11:30 pm → stomach burning same night. S04 (Jan 28, Day 23): late dinner 11 pm during deadline → same burning. No acidity in S03 (Jan 19, Day 14) when late eating absent.",
      "confidence": "high",
      "confidence_justification": "4 episodes all preceded by late dinner; acidity absent in 5 sessions without late eating",
      "trace": [
        "Symptom 'stomach burning' first appears in S01 (Jan 5, Day 0)",
        "Trigger 'late dinner 11 pm+' present in S01, S04, S07, S09; absent in S02, S03, S05, S06, S08",
        "Symptom onset within hours of trigger each time — same-night causation",
        "No acidity in 5 sessions without late eating — strong negative evidence"
      ]
    }}
  ]
}}
```\
"""


# ── Public API ────────────────────────────────────────────────────────────────

def build_system_prompt(user: User) -> str:
    """Build the full system prompt for a user, embedding the temporal timeline."""
    timeline = build_temporal_timeline(user)
    return CHAT_SYSTEM_TEMPLATE.format(
        user_name=user.name,
        user_id=user.user_id,
        timeline=timeline,
    )


def build_initial_user_message(user: User) -> str:
    """Build the first user turn that triggers the initial pattern analysis."""
    return INITIAL_ANALYSIS_TEMPLATE.format(user_name=user.name)


# ── Cross-patient prompts ─────────────────────────────────────────────────────

CROSS_PATIENT_SYSTEM_TEMPLATE = """\
You are Clary, the AI health pattern analyst for the Ask First platform.\
 You have access to ALL patients' complete conversation histories, each pre-processed\
 with exact temporal metadata.

## Your Role
Find health patterns that recur across MULTIPLE patients — population-level signals\
 that no single patient's history alone could reveal.\
 Answer follow-up questions with the same analytical depth.

## Temporal Reasoning Rules
1. Always cite patient name + session ID (e.g., Meera USR002_S03) and date.
2. State time deltas per patient: "Meera S03 (Jan 22, Day 17) → S05 (Feb 10, Day 36) = 19-day lag".
3. Temporal direction is causal: trigger must precede symptom in EACH patient's data.
4. Check negative evidence within each patient's timeline separately.
5. Note differences: same pattern but different lag, severity, or context across patients.
6. Known biological delays: telogen effluvium 42–84 d; blood sugar crash 1–3 h;\
 stress → PMS days–weeks; sleep deprivation → anxiety cumulative over weeks.

## Confidence Rubric
- **HIGH**: 3+ patients, OR 2 patients each with 2+ consistent episodes PLUS negative evidence
- **MEDIUM**: 2 patients each with at least 1 consistent episode
- **LOW**: 2 patients with 1 episode each, or plausible mechanism with limited data

## Pre-Computed Temporal Timelines

{timelines}

---
For the **initial analysis**: output ALL cross-patient findings as JSON inside ```json … ``` fences.
For **follow-up questions**: respond conversationally, citing patient names, session IDs, and day deltas.\
"""

CROSS_PATIENT_ANALYSIS_TEMPLATE = """\
Perform a cross-patient population pattern analysis across all {n_users} patients: {user_names}.

Goal: identify trigger→symptom patterns that recur across 2 or more patients.

Steps:
1. For each patient list dominant recurring symptoms and their triggers.
2. Find triggers or symptoms appearing in 2+ patients' timelines.
3. For each candidate: verify temporal direction holds in EACH patient. Cite sessions.
4. Note differences in lag, severity, or resolution across patients.
5. Apply the confidence rubric strictly. Do not inflate confidence.
6. Merge overlapping patterns into a single root-cause pattern when possible.

Evidence gate: every session cited must actually contain the trigger or symptom. \
Do not include speculative patterns without explicit session evidence from ≥2 patients.

Output JSON in ```json … ``` fences:

```json
{{
  "analysis_type": "cross_patient",
  "users_analyzed": {user_ids},
  "cross_patient_patterns": [
    {{
      "pattern_id": "CP1",
      "title": "Trigger → Symptom (concise)",
      "users_affected": ["USR002 (Meera)", "USR003 (Priya)"],
      "sessions_involved": ["USR002_S03", "USR002_S07", "USR003_S02", "USR003_S05"],
      "temporal_reasoning": "Cross-patient narrative. Meera S03 (Jan 22, Day 17): work deadline → cramps same day. Priya S02 (Jan 15, Day 10): project deadline → cramps within 2 days.",
      "confidence": "medium",
      "confidence_justification": "2 patients, each 2+ consistent episodes; same trigger (acute work stress) preceding cramps within 0–2 days",
      "trace": [
        "Meera: work stress in S03, S07 → cramps within 0–1 days each time",
        "Priya: work deadline in S02, S05 → cramps within 0–2 days",
        "Lag difference: Meera same-day, Priya up to 2 days — plausible stress hormone pathway"
      ]
    }}
  ]
}}
```\
"""


def build_all_patients_system_prompt(users: list) -> str:
    sections = []
    for user in users:
        timeline = build_temporal_timeline(user)
        sections.append(f"### {user.name} ({user.user_id})\n\n{timeline}")
    combined = "\n\n---\n\n".join(sections)
    return CROSS_PATIENT_SYSTEM_TEMPLATE.format(timelines=combined)


def build_cross_patient_user_message(users: list) -> str:
    names    = ", ".join(u.name for u in users)
    ids_json = json.dumps([u.user_id for u in users])
    return CROSS_PATIENT_ANALYSIS_TEMPLATE.format(
        n_users=len(users),
        user_names=names,
        user_ids=ids_json,
    )


def stream_response(
    system_prompt: str,
    messages: list[dict],
    provider: LLMProvider,
    max_tokens: int | None = None,
) -> Generator[tuple[str, str], None, None]:
    """
    Thin wrapper that calls provider.chat_stream and converts errors to events.

    Yields:
      ("thinking", chunk)  — live reasoning token
      ("content",  chunk)  — response token
      ("error",    msg)    — exception message
    """
    if max_tokens is None:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "16384"))
    try:
        yield from provider.chat_stream(system_prompt, messages, max_tokens)
    except Exception as exc:
        yield ("error", str(exc))
