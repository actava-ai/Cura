"""HealthBench (hard / regular / consensus): open-ended health conversations graded
against physician-written rubrics by an LLM judge."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from cura_eval import judge as judge_mod
from cura_eval.client import ChatClient
from cura_eval.loaders import HBExample, load_healthbench
from cura_eval.rows import assistant_message, breakdown, mean, response_row

log = logging.getLogger(__name__)


async def _one(
    ex: HBExample,
    candidate: ChatClient,
    *,
    judge_client: Any,
    judge_model: str,
    max_tokens: int,
    temperature: float | None,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    async with sem:
        completion = await candidate.complete(
            ex.prompt, max_tokens=max_tokens, temperature=temperature
        )
        try:
            graded = await judge_mod.grade_healthbench(
                ex.prompt,
                completion.content,
                criteria=ex.criteria,
                points=ex.points,
                client=judge_client,
                model=judge_model,
            )
            reward, judgments, judge_error = graded["reward"], graded["judgments"], False
        except Exception as e:
            log.warning("judge failure for %s: %s", ex.task_id, e)
            reward, judgments, judge_error = 0.0, [], True
    return response_row(
        ex.prompt,
        assistant_message(completion.content, completion.reasoning),
        example_id=ex.task_id,
        id_key="task_id",
        response=completion.content,
        score={
            "reward": reward,
            "judgments": judgments,
            "axes": ex.axes,
            "theme": ex.theme,
            "judge_error": judge_error,
        },
    )


def _axis_breakdown(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Per-axis earned/total point ratio across all judged rubric items."""
    earned: dict[str, float] = {}
    total: dict[str, float] = {}
    for row in rows:
        # strict=False on purpose: a judge-error row has judgments=[] while axes is non-empty.
        for j, axis in zip(row["score"]["judgments"], row["score"]["axes"], strict=False):
            if j["points"] > 0:
                total[axis] = total.get(axis, 0.0) + j["points"]
                if j["criteria_met"]:
                    earned[axis] = earned.get(axis, 0.0) + j["points"]
    return {a: earned.get(a, 0.0) / t for a, t in sorted(total.items()) if t}


async def run_healthbench(
    candidate: ChatClient,
    *,
    judge_client: Any,
    judge_model: str = judge_mod.HB_DEFAULT_JUDGE,
    variant: str = "hard",
    limit: int | None = None,
    max_tokens: int = 8192,
    temperature: float | None = 0.0,
    concurrency: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = load_healthbench(variant, limit)
    sem = asyncio.Semaphore(concurrency)
    rows = await asyncio.gather(
        *[
            _one(
                ex,
                candidate,
                judge_client=judge_client,
                judge_model=judge_model,
                max_tokens=max_tokens,
                temperature=temperature,
                sem=sem,
            )
            for ex in examples
        ]
    )
    rows = list(rows)
    scored = [r for r in rows if not r["score"]["judge_error"]]
    summary = {
        "benchmark": f"healthbench_{variant}",
        "score": mean([r["score"]["reward"] for r in scored]),
        "n": len(rows),
        "n_judge_errors": len(rows) - len(scored),
        "per_axis": _axis_breakdown(scored),
        "per_theme": breakdown([(r["score"]["theme"], r["score"]["reward"]) for r in scored]),
    }
    return rows, summary
