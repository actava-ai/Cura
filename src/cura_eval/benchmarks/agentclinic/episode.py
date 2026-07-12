"""One AgentClinic episode — a faithful async transcription of the upstream main loop.

Routing (DIAGNOSIS READY / REQUEST TEST), the final-question nudge, and moderator grading
follow upstream's exact control flow. One intentional deviation: image gating uses
``dataset.startswith("NEJM")`` (vs upstream's ``== "NEJM"``) so it also covers NEJM_Ext.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cura_eval.benchmarks.agentclinic.images import ImageFetchError, fetch_image, image_data_uri
from cura_eval.benchmarks.agentclinic.npc import (
    MeasurementAgent,
    PatientAgent,
    compare_results,
)


@dataclass
class EpisodeResult:
    reward: float
    turns: list[dict] = field(default_factory=list)
    logs: dict = field(default_factory=dict)


def _image_uri_for(
    scenario: Any, dataset: str, doctor_dialogue: str, image_request: bool, max_image_size: int
) -> str | None:
    if not dataset.startswith("NEJM"):
        return None
    requested = ("REQUEST IMAGES" in doctor_dialogue) if image_request else True
    if not requested:
        return None
    return image_data_uri(fetch_image(scenario.image_url), max_image_size)


async def run_episode(
    scenario: Any,
    dataset: str,
    *,
    doctor: Any,
    patient_agent: PatientAgent,
    measurement_agent: MeasurementAgent,
    judge_client: Any,
    judge_model: str,
    total_inferences: int,
    image_request: bool = False,
    max_image_size: int = 1024,
) -> EpisodeResult:
    turns: list[dict] = []
    pi_dialogue = ""
    doctor_dialogue = ""
    reward, stopped, diagnosis = 0.0, "budget", ""
    try:
        for inf_id in range(total_inferences):
            if inf_id == total_inferences - 1:
                pi_dialogue += "This is the final question. Please provide a diagnosis.\n"
            image_uri = _image_uri_for(
                scenario, dataset, doctor_dialogue, image_request, max_image_size
            )
            turn = await doctor.step(pi_dialogue, image_uri)
            doctor_dialogue = turn.utterance
            turns.append({"role": "doctor", "content": turn.utterance, "reasoning": turn.reasoning})

            if "DIAGNOSIS READY" in doctor_dialogue:
                diagnosis = doctor_dialogue
                correct = await compare_results(
                    judge_client, judge_model, doctor_dialogue, scenario.diagnosis_information()
                )
                reward = 1.0 if correct else 0.0
                stopped = "diagnosis"
                break
            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = await measurement_agent.inference_measurement(doctor_dialogue)
                turns.append({"role": "measurement", "content": pi_dialogue})
                patient_agent.add_hist(pi_dialogue)
            else:
                pi_dialogue = await patient_agent.inference_patient(doctor_dialogue)
                turns.append({"role": "patient", "content": pi_dialogue})
                measurement_agent.add_hist(pi_dialogue)
    except Exception as exc:
        # Any turn-level failure (image fetch, context overflow, transient API) ends the
        # episode but preserves the partial trajectory — completed turns are valid data.
        kind = "image" if isinstance(exc, ImageFetchError) else "sampling"
        return EpisodeResult(
            0.0,
            turns,
            {
                "stopped": "error",
                "error": f"{kind}: {str(exc)[:200]}",
                "gold": scenario.diagnosis_information(),
                "n_turns": sum(1 for t in turns if t["role"] == "doctor"),
            },
        )

    return EpisodeResult(
        reward,
        turns,
        {
            "diagnosis": diagnosis,
            "gold": scenario.diagnosis_information(),
            "n_turns": sum(1 for t in turns if t["role"] == "doctor"),
            "stopped": stopped,
        },
    )
