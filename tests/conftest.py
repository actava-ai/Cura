"""Offline test doubles: an AsyncOpenAI-shaped fake client (no network, no keys)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def make_message(
    content: str | None = "",
    *,
    reasoning_content: str | None = None,
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """An SDK-shaped chat message (attribute access, like the openai SDK objects)."""
    tcs = None
    if tool_calls:
        tcs = [
            SimpleNamespace(
                id=tc.get("id", f"call_{i}"),
                type="function",
                function=SimpleNamespace(
                    name=tc["function"]["name"], arguments=tc["function"]["arguments"]
                ),
            )
            for i, tc in enumerate(tool_calls)
        ]
    msg = SimpleNamespace(content=content, tool_calls=tcs)
    if reasoning_content is not None:
        msg.reasoning_content = reasoning_content
    if reasoning is not None:
        msg.reasoning = reasoning
    return msg


def make_response(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    """AsyncOpenAI-shaped fake: pops scripted messages (or calls a handler) per request.

    ``script`` is either a list of messages (popped in order; the last one repeats) or a
    callable ``(kwargs) -> message``. Every request's kwargs are recorded on ``.calls``.
    """

    def __init__(self, script: Any):
        self._script = script
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if callable(self._script):
            msg = self._script(kwargs)
        else:
            msg = self._script[0] if len(self._script) == 1 else self._script.pop(0)
        if isinstance(msg, Exception):
            raise msg
        return make_response(msg)
