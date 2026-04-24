from typing import Generator
import os

from src.data_loader import User
from src.llm_provider import LLMProvider
from src.temporal_engine import build_temporal_timeline
from src.prompts import SYSTEM_PROMPT, REASONING_PROMPT_TEMPLATE, JSON_SCHEMA_INSTRUCTION


def _extract_json(text: str) -> str | None:
    """Pull the JSON block out of ```json ... ``` fences, or return raw text if it starts with {."""
    if "```json" in text:
        start = text.index("```json") + len("```json")
        remainder = text[start:]
        end = remainder.index("```") if "```" in remainder else len(remainder)
        return remainder[:end].strip()
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped
    return None


def stream_pattern_detection(
    user: User,
    provider: LLMProvider,
) -> Generator[tuple[str, str], None, None]:
    """
    Streams pattern detection for one user.

    Yields:
      ("reasoning", chunk)          — thinking/reasoning text chunk (live)
      ("reasoning_complete", full)  — full thinking text after stream ends
      ("complete", json_str)        — parsed JSON string
      ("error", message)            — on failure
    """
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "16384"))

    timeline = build_temporal_timeline(user)
    sessions = sorted(user.sessions, key=lambda s: s.timestamp)
    date_range = (
        f"{sessions[0].timestamp.strftime('%b %d, %Y')} - "
        f"{sessions[-1].timestamp.strftime('%b %d, %Y')}"
    )

    user_prompt = REASONING_PROMPT_TEMPLATE.format(
        user_name=user.name,
        user_id=user.user_id,
        onboarding_notes=user.onboarding_notes,
        session_count=len(user.sessions),
        date_range=date_range,
        timeline=timeline,
        json_schema=JSON_SCHEMA_INSTRUCTION,
    )

    thinking_text = ""
    content_buffer = ""

    try:
        for chunk_type, chunk in provider.stream_text(SYSTEM_PROMPT, user_prompt, max_tokens):
            if chunk_type == "thinking":
                thinking_text += chunk
                yield ("reasoning", chunk)
            elif chunk_type == "content":
                content_buffer += chunk

        # Emit reasoning_complete so the UI can finalise the reasoning display
        if thinking_text:
            yield ("reasoning_complete", thinking_text)

        # Extract JSON from the content stream
        clean_json = _extract_json(content_buffer)
        if clean_json:
            yield ("complete", clean_json)
        else:
            yield ("error", "No JSON found in model response:\n" + content_buffer)

    except Exception as exc:
        yield ("error", str(exc))
