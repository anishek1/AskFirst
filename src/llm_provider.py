"""
llm_provider.py — Abstract LLM provider with Anthropic / OpenAI / Gemini / Ollama backends.

Each provider exposes one method:
  chat_stream  — multi-turn, accepts message history list

Yields: ("thinking", chunk) | ("content", chunk)
"""

from abc import ABC, abstractmethod
from typing import Generator
import os


class LLMProvider(ABC):
    @abstractmethod
    def chat_stream(
        self, system: str, messages: list[dict], max_tokens: int
    ) -> Generator[tuple[str, str], None, None]:
        """
        Multi-turn streaming.
        messages = [{"role": "user"|"assistant", "content": str}, ...]
        Yields ("thinking"|"content", chunk).
        """


# ── Anthropic ─────────────────────────────────────────────────────────────────
class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def chat_stream(
        self, system: str, messages: list[dict], max_tokens: int
    ) -> Generator[tuple[str, str], None, None]:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield ("content", text)


# ── OpenAI / OpenAI-compatible (NVIDIA NIM, etc.) ────────────────────────────
class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        thinking_enabled: bool = False,
        reasoning_budget: int = 16384,
    ):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._thinking_enabled = thinking_enabled
        self._reasoning_budget = reasoning_budget

    def chat_stream(
        self, system: str, messages: list[dict], max_tokens: int
    ) -> Generator[tuple[str, str], None, None]:
        openai_messages = [{"role": "system", "content": system}] + messages
        kwargs: dict = dict(
            model=self._model,
            max_tokens=max_tokens,
            messages=openai_messages,
            stream=True,
            temperature=1,
            top_p=1,
        )
        if self._thinking_enabled:
            kwargs["extra_body"] = {
                "reasoning_budget": self._reasoning_budget,
                "chat_template_kwargs": {"enable_thinking": True},
            }

        response = self._client.chat.completions.create(**kwargs)
        for chunk in response:
            if not chunk.choices:
                continue
            # Some providers (NVIDIA, DeepSeek) surface reasoning in a separate field
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if reasoning:
                yield ("thinking", reasoning)
            content = chunk.choices[0].delta.content
            if content:
                yield ("content", content)


# ── Google Gemini ─────────────────────────────────────────────────────────────
class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model_name = model

    def chat_stream(
        self, system: str, messages: list[dict], max_tokens: int
    ) -> Generator[tuple[str, str], None, None]:
        import google.generativeai as genai

        model = genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system,
            generation_config={"max_output_tokens": max_tokens},
        )
        # Convert to Gemini's history format (all messages except the last)
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        last_content = messages[-1]["content"] if messages else ""
        chat = model.start_chat(history=history)
        for chunk in chat.send_message(last_content, stream=True):
            if chunk.text:
                yield ("content", chunk.text)


# ── Ollama ────────────────────────────────────────────────────────────────────
class OllamaProvider(LLMProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def chat_stream(
        self, system: str, messages: list[dict], max_tokens: int
    ) -> Generator[tuple[str, str], None, None]:
        import json
        import urllib.request

        all_messages = [{"role": "system", "content": system}] + messages
        payload = json.dumps({
            "model": self._model,
            "messages": all_messages,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if line.strip():
                    obj = json.loads(line)
                    text = obj.get("message", {}).get("content", "")
                    if text:
                        yield ("content", text)


# ── Factory ───────────────────────────────────────────────────────────────────
def get_provider() -> LLMProvider:
    provider_name = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")
    base_url = os.getenv("LLM_BASE_URL") or None
    thinking = os.getenv("LLM_THINKING_ENABLED", "false").lower() == "true"
    reasoning_budget = int(os.getenv("LLM_REASONING_BUDGET", "16384"))

    if provider_name == "anthropic":
        return AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-6")
    elif provider_name == "openai":
        model = model or "nvidia/nemotron-3-nano-30b-a3b"
        if base_url is None and model.startswith("nvidia/"):
            base_url = "https://integrate.api.nvidia.com/v1"
        return OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            thinking_enabled=thinking,
            reasoning_budget=reasoning_budget,
        )
    elif provider_name == "gemini":
        return GeminiProvider(api_key=api_key, model=model or "gemini-1.5-pro")
    elif provider_name == "ollama":
        return OllamaProvider(
            model=model or "llama3",
            base_url=base_url or "http://localhost:11434",
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider_name}'. "
            "Must be one of: anthropic, openai, gemini, ollama"
        )
