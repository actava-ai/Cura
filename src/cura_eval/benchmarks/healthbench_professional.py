"""HealthBench Professional: professional-context health tasks (525-example test split)
graded against per-example rubrics, with a length-adjusted headline score."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from cura_eval import judge as judge_mod
from cura_eval.client import ChatClient
from cura_eval.loaders import HBPExample, load_hbp
from cura_eval.rows import assistant_message, breakdown, mean, response_row

log = logging.getLogger(__name__)

DIMS = ("use_case", "type", "difficulty", "specialty")


async def _one(
    ex: HBPExample,
    candidate: ChatClient,
    *,
    judge_client: Any,
    judge_model: str,
    judge_reasoning_effort: str,
    max_tokens: int,
    temperature: float | None,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        completion = await candidate.complete(
            ex.conversation, max_tokens=max_tokens, temperature=temperature
        )
        try:
            graded = await judge_mod.grade_hbp(
                ex.conversation,
                completion.content,
                rubric_items=ex.rubric_items,
                client=judge_client,
                model=judge_model,
                reasoning_effort=judge_reasoning_effort,
            )
            raw, adjusted, judgments, judge_error = (
                graded["raw"],
                graded["adjusted"],
                graded["judgments"],
                False,
            )
        except Exception as e:
            log.warning("judge failure for %s: %s", ex.task_id, e)
            raw, adjusted, judgments, judge_error = 0.0, 0.0, [], True
    return response_row(
        ex.conversation,
        assistant_message(completion.content, completion.reasoning),
        example_id=ex.task_id,
        id_key="id",
        response=completion.content,
        score={
            "raw_reward": raw,
            "adjusted_reward": adjusted,
            "judgments": judgments,
            "use_case": ex.use_case,
            "type": ex.type,
            "difficulty": ex.difficulty,
            "specialty": ex.specialty,
            "judge_error": judge_error,
        },
    )


async def run_healthbench_professional(
    candidate: ChatClient,
    *,
    judge_client: Any,
    judge_model: str = judge_mod.HBP_DEFAULT_JUDGE,
    judge_reasoning_effort: str = "low",
    limit: int | None = None,
    max_tokens: int = 32000,
    temperature: float | None = 0.0,
    concurrency: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = load_hbp(limit)
    sem = asyncio.Semaphore(concurrency)
    rows = await asyncio.gather(
        *[
            _one(
                ex,
                candidate,
                judge_client=judge_client,
                judge_model=judge_model,
                judge_reasoning_effort=judge_reasoning_effort,
                max_tokens=max_tokens,
                temperature=temperature,
                sem=sem,
            )
            for ex in examples
        ]
    )
    rows = list(rows)
    scored = [r for r in rows if not r["score"]["judge_error"]]
    summary: dict[str, Any] = {
        "benchmark": "healthbench_professional",
        "score": mean([r["score"]["adjusted_reward"] for r in scored]),
        "overall_raw": mean([r["score"]["raw_reward"] for r in scored]),
        "n": len(rows),
        "n_judge_errors": len(rows) - len(scored),
    }
    for dim in DIMS:
        summary[f"per_{dim}"] = breakdown(
            [(r["score"][dim], r["score"]["adjusted_reward"]) for r in scored]
        )
    return rows, summary
