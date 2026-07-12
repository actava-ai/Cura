"""Fan AgentClinic episodes out over a dataset and aggregate diagnostic accuracy."""

from __future__ import annotations

import asyncio
from typing import Any

from cura_eval.benchmarks.agentclinic.doctor import Doctor, build_doctor_system_prompt
from cura_eval.benchmarks.agentclinic.episode import EpisodeResult, run_episode
from cura_eval.benchmarks.agentclinic.npc import MeasurementAgent, PatientAgent
from cura_eval.benchmarks.agentclinic.scenarios import load_scenarios, mcq_match
from cura_eval.client import ChatClient
from cura_eval.rows import mean


async def run_agentclinic(
    candidate: ChatClient,
    *,
    judge_client: Any,
    judge_model: str = "gpt-5.5",
    dataset: str = "MedQA",
    limit: int | None = None,
    total_inferences: int = 20,
    max_tokens: int = 8192,
    temperature: float | None = 0.0,
    concurrency: int = 8,
    image_request: bool = False,
    preserve_thinking: bool = False,
    max_image_size: int = 1024,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenarios = load_scenarios(dataset)
    if limit is not None:
        scenarios = scenarios[:limit]
    sem = asyncio.Semaphore(concurrency)
    progress = {"n": 0, "total": len(scenarios)}

    async def one(idx: int, scenario: Any) -> dict[str, Any]:
        async with sem:
            patient = PatientAgent(scenario, judge_client, judge_model)
            measurement = MeasurementAgent(scenario, judge_client, judge_model)
            doctor = Doctor(
                build_doctor_system_prompt(scenario, total_inferences, image_request),
                candidate,
                max_tokens=max_tokens,
                temperature=temperature,
                preserve_thinking=preserve_thinking,
            )
            try:
                res = await run_episode(
                    scenario,
                    dataset,
                    doctor=doctor,
                    patient_agent=patient,
                    measurement_agent=measurement,
                    judge_client=judge_client,
                    judge_model=judge_model,
                    total_inferences=total_inferences,
                    image_request=image_request,
                    max_image_size=max_image_size,
                )
            except Exception as exc:  # episode-level resilience: never crash the run
                res = EpisodeResult(
                    0.0,
                    [],
                    {
                        "stopped": "error",
                        "error": str(exc)[:200],
                        "gold": scenario.diagnosis_information(),
                    },
                )
            progress["n"] += 1
            print(
                f"  [{progress['n']}/{progress['total']}] {dataset}:{idx} "
                f"reward={res.reward} turns={res.logs.get('n_turns')} "
                f"stopped={res.logs.get('stopped')}",
                flush=True,
            )
            return {
                "task_id": f"{dataset}:{idx}",
                "dataset": dataset,
                "reward": res.reward,
                "turns": res.turns,
                "logs": {
                    **res.logs,
                    "mcq_match": mcq_match(res.logs.get("diagnosis", ""), scenario),
                },
            }

    rows = list(await asyncio.gather(*[one(i, s) for i, s in enumerate(scenarios)]))
    summary = {
        "benchmark": f"agentclinic_{dataset}",
        "score": mean([r["reward"] for r in rows]),
        "n": len(rows),
        "n_correct": sum(1 for r in rows if r["reward"] > 0),
        "n_errors": sum(1 for r in rows if r["logs"].get("stopped") == "error"),
        "mean_doctor_turns": mean([float(r["logs"].get("n_turns") or 0) for r in rows]),
    }
    return rows, summary
