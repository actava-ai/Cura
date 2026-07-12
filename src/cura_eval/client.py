"""The one model-agnostic seam: an OpenAI-compatible chat-completions client.

Point it at any server that speaks the OpenAI chat-completions dialect — the Cura API
(https://inference.actava.ai/v1), OpenAI, OpenRouter, vLLM, SGLang, Ollama, LM Studio, ….
Reasoning traces are captured from either `message.reasoning_content` (Cura/Kimi/DeepSeek
convention) or `message.reasoning` (OpenRouter convention) and never silently dropped:
saved rows carry them as `reasoning_content` so collected trajectories stay SFT-trainable.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChatResult:
    """One model turn: the final answer, its optional reasoning trace, and any tool calls.

    `tool_calls` is a list of plain OpenAI-shaped dicts:
    `{"id": ..., "type": "function", "function": {"name": ..., "arguments": "<json str>"}}`.
    """

    content: str
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def assistant_message(self) -> dict[str, Any]:
        """This turn as a chat message (reasoning lifted to `reasoning_content`, never inline)."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning:
            msg["reasoning_content"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


def _make_openai_client(*, base_url: str | None, api_key: str, timeout: float | None):
    """Lazy import so offline tests never need the openai package at collection time."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _plain_tool_calls(raw: Any) -> list[dict[str, Any]]:
    """SDK tool-call objects -> plain OpenAI-shaped dicts (JSON-serializable)."""
    calls: list[dict[str, Any]] = []
    for tc in raw or []:
        if isinstance(tc, dict):
            calls.append(tc)
            continue
        fn = tc.function
        calls.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": fn.name, "arguments": fn.arguments},
            }
        )
    return calls


class ChatClient:
    """Bounded-concurrency, retrying wrapper over an OpenAI-compatible endpoint.

    Args:
        model: model id as the server knows it (e.g. ``actava/cura-soar``).
        base_url: server base URL; ``None`` = the official OpenAI API.
        api_key: bearer token. Falls back to ``api_key_env`` (then to ``"EMPTY"`` for
            local servers that ignore auth).
        api_key_env: name of the environment variable holding the key.
        client: injected AsyncOpenAI-compatible client (tests / advanced use).
        concurrency: max in-flight requests through this client.
        max_retries: attempts per request (exponential backoff between attempts).
        extra_body: provider-specific body forwarded verbatim on every request
            (e.g. OpenRouter ``provider``/``reasoning`` routing, Cura ``thinking``).
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        client: Any | None = None,
        concurrency: int = 16,
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
        timeout: float | None = 600.0,
    ):
        if client is None:
            key = api_key or (os.environ.get(api_key_env, "") if api_key_env else "")
            if not key:
                if api_key_env:
                    raise ValueError(
                        f"no API key: set {api_key_env} (or pass api_key=). "
                        "For keyless local servers pass api_key='EMPTY'."
                    )
                key = "EMPTY"
            client = _make_openai_client(base_url=base_url, api_key=key, timeout=timeout)
        self._client = client
        self.model = model
        self.base_url = base_url
        self._sem = asyncio.Semaphore(concurrency)
        self._max_retries = max(1, max_retries)
        self._extra_body = extra_body or None

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if stop:
            kwargs["stop"] = stop
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        last_error: Exception | None = None
        async with self._sem:
            for attempt in range(self._max_retries):
                try:
                    resp = await self._client.chat.completions.create(**kwargs)
                    msg = resp.choices[0].message
                    reasoning = (
                        getattr(msg, "reasoning_content", None)
                        or getattr(msg, "reasoning", None)
                        or None
                    )
                    return ChatResult(
                        content=msg.content or "",
                        reasoning=reasoning,
                        tool_calls=_plain_tool_calls(getattr(msg, "tool_calls", None)),
                    )
                except Exception as e:  # transient API/infra error: back off and retry
                    last_error = e
                    if attempt < self._max_retries - 1:
                        await asyncio.sleep(min(2**attempt, 10))
            assert last_error is not None
            raise last_error
