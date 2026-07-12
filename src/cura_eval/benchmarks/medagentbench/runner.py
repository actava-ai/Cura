"""Drive one native-tool-calling episode per MedAgentBench task and grade it.

Needs a running FHIR server (read-only):

    docker run -p 8080:8080 jyxsu6/medagentbench:latest
"""

from __future__ import annotations

import asyncio
from typing import Any

from cura_eval.benchmarks.medagentbench import refsol
from cura_eval.benchmarks.medagentbench.fhir import verify_fhir_server
from cura_eval.benchmarks.medagentbench.loaders import (
    MABExample,
    build_system_prompt,
    build_user_message,
    load_medagentbench,
)
from cura_eval.benchmarks.medagentbench.tools import TOOL_SPECS, EpisodeState, execute_tool_call
from cura_eval.client import ChatClient
from cura_eval.rows import breakdown

_NO_TOOL_NUDGE = (
    "Your last message contained no tool call. Act through the provided tools "
    "(fhir_get, fhir_post) and call finish(answer) exactly once when you are done."
)


async def _episode(
    ex: MABExample,
    candidate: ChatClient,
    *,
    api_base: str,
    max_round: int,
    max_tokens: int,
    temperature: float | None,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(api_base)},
        {"role": "user", "content": build_user_message(ex)},
    ]
    state = EpisodeState()
    error: str | None = None
    async with sem:
        for _ in range(max_round):
            try:
                result = await candidate.complete(
                    messages, tools=TOOL_SPECS, max_tokens=max_tokens, temperature=temperature
                )
            except Exception as e:  # episode-level resilience: grade what we have
                error = str(e)[:200]
                break
            messages.append(result.assistant_message())
            if not result.tool_calls:
                messages.append({"role": "user", "content": _NO_TOOL_NUDGE})
                continue
            for tc in result.tool_calls:
                obs = await execute_tool_call(tc, state, api_base)
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": obs})
            if state.finished:
                break

    correct, gold = await asyncio.to_thread(refsol.grade, ex.raw, state, api_base)
    score: dict[str, Any] = {
        "correct": bool(correct),
        "task_type": ex.task_type,
        "gold": gold,
        "finished": state.finished,
        "n_posts": len(state.posts),
    }
    if error:
        score["error"] = error
    # Full multi-turn history (drop a trailing unanswered nudge); top-level `tools` so a
    # saved episode is re-trainable as tool-use SFT data.
    if (
        messages
        and messages[-1].get("role") == "user"
        and messages[-1]["content"] == _NO_TOOL_NUDGE
    ):
        messages = messages[:-1]
    return {
        "task_id": ex.id,
        "messages": messages,
        "tools": TOOL_SPECS,
        "response": state.result,
        "score": score,
    }


async def run_medagentbench(
    candidate: ChatClient,
    *,
    fhir_api_base: str,
    version: str = "v1",
    limit: int | None = None,
    max_round: int = 8,
    max_tokens: int = 8192,
    temperature: float | None = 0.0,
    concurrency: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not verify_fhir_server(fhir_api_base):
        raise RuntimeError(
            f"FHIR server unreachable at {fhir_api_base}; start it with: "
            "docker run -p 8080:8080 jyxsu6/medagentbench:latest"
        )
    examples = load_medagentbench(version, limit)
    sem = asyncio.Semaphore(concurrency)
    rows = await asyncio.gather(
        *[
            _episode(
                ex,
                candidate,
                api_base=fhir_api_base,
                max_round=max_round,
                max_tokens=max_tokens,
                temperature=temperature,
                sem=sem,
            )
            for ex in examples
        ]
    )
    rows = list(rows)
    corrects = [1.0 if r["score"]["correct"] else 0.0 for r in rows]
    summary = {
        "benchmark": "medagentbench",
        "score": sum(corrects) / len(corrects) if corrects else 0.0,
        "n": len(rows),
        "n_correct": int(sum(corrects)),
        "per_task_type": breakdown(
            [(r["score"]["task_type"], 1.0 if r["score"]["correct"] else 0.0) for r in rows]
        ),
        "note": (
            "Native tool-calling protocol; not directly comparable to published "
            "MedAgentBench scores (upstream uses a strict GET/POST/FINISH text protocol)."
        ),
    }
    return rows, summary
