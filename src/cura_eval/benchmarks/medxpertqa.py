"""MedXpertQA (text / mm): board-exam-style MCQ, graded by exact letter match.

Judge-free: the grader is the deterministic ``cura_eval.mcq.extract_letter``. The MM
subset sends base64 ``image_url`` parts, so it needs a vision-capable endpoint.
"""

from __future__ import annotations

import asyncio
from typing import Any

from cura_eval.client import ChatClient
from cura_eval.loaders import MXQExample, load_medxpertqa
from cura_eval.mcq import extract_letter
from cura_eval.rows import assistant_message, breakdown, response_row, strip_images


async def _one(
    ex: MXQExample,
    candidate: ChatClient,
    *,
    max_tokens: int,
    temperature: float | None,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        completion = await candidate.complete(
            ex.messages, max_tokens=max_tokens, temperature=temperature
        )
    # Prefer the final answer; fall back to the reasoning trace when the answer is empty
    # (e.g. a reasoning model that ran out of budget mid-answer).
    chosen = extract_letter(completion.content, ex.valid_letters[-1]) or extract_letter(
        completion.reasoning or "", ex.valid_letters[-1]
    )
    correct = chosen == ex.answer
    return response_row(
        strip_images(ex.messages),
        assistant_message(completion.content, completion.reasoning),
        example_id=ex.task_id,
        id_key="task_id",
        response=completion.content,
        score={
            "correct": correct,
            "chosen": chosen,
            "gold": ex.answer,
            "max_letter": ex.valid_letters[-1],
            "medical_task": ex.medical_task,
            "body_system": ex.body_system,
        },
    )


async def run_medxpertqa(
    candidate: ChatClient,
    *,
    subset: str = "text",
    limit: int | None = None,
    max_tokens: int = 8192,
    temperature: float | None = 0.0,
    concurrency: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = load_medxpertqa(subset, limit)  # type: ignore[arg-type]
    sem = asyncio.Semaphore(concurrency)
    rows = await asyncio.gather(
        *[
            _one(ex, candidate, max_tokens=max_tokens, temperature=temperature, sem=sem)
            for ex in examples
        ]
    )
    rows = list(rows)
    corrects = [1.0 if r["score"]["correct"] else 0.0 for r in rows]
    summary = {
        "benchmark": f"medxpertqa_{subset}",
        "score": sum(corrects) / len(corrects) if corrects else 0.0,
        "n": len(rows),
        "n_correct": int(sum(corrects)),
        "per_medical_task": breakdown(
            [(r["score"]["medical_task"], 1.0 if r["score"]["correct"] else 0.0) for r in rows]
        ),
        "per_body_system": breakdown(
            [(r["score"]["body_system"], 1.0 if r["score"]["correct"] else 0.0) for r in rows]
        ),
    }
    return rows, summary
