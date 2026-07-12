"""The three AgentClinic NPCs — patient, measurement reader, moderator — on the judge endpoint.

Prompts are upstream AgentClinic's verbatim (MIT). Reasoning-class judge models get a
per-role ``reasoning_effort`` (patient role-play needs none; the measurement reader needs
depth to map free-text test requests onto the case's available findings); non-reasoning
models get ``temperature=0.0``.
"""

from __future__ import annotations

from typing import Any

from cura_eval.judge import is_reasoning_model

_EFFORT = {"patient": "none", "measurement": "high", "moderator": "low"}


async def _npc_chat(client: Any, model: str, role: str, system_prompt: str, prompt: str) -> str:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if is_reasoning_model(model):
        effort = _EFFORT[role]
        if effort != "none":
            kwargs["reasoning_effort"] = effort
    else:
        kwargs["temperature"] = 0.0
    resp = await client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


class PatientAgent:
    """Simulated patient: answers in dialogue, never reveals the disease explicitly."""

    def __init__(self, scenario: Any, client: Any, model: str):
        self._client = client
        self._model = model
        self.agent_hist = ""
        self.symptoms = scenario.patient_information()

    def system_prompt(self) -> str:
        base = (
            "You are a patient in a clinic who only responds in the form of dialogue. "
            "You are being inspected by a doctor who will ask you questions and will perform "
            "exams on you in order to understand your disease. Your answer will only be 1-3 "
            "sentences in length."
        )
        symptoms = (
            f"\n\nBelow is all of your information. {self.symptoms}. \n\n Remember, you must "
            "not reveal your disease explicitly but may only convey the symptoms you have in "
            "the form of dialogue if you are asked."
        )
        return base + symptoms

    async def inference_patient(self, question: str) -> str:
        prompt = (
            f"\nHere is a history of your dialogue: {self.agent_hist}\n "
            f"Here was the doctor response: {question}"
            "Now please continue your dialogue\nPatient: "
        )
        answer = await _npc_chat(self._client, self._model, "patient", self.system_prompt(), prompt)
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def add_hist(self, hist_str: str) -> None:
        self.agent_hist += hist_str + "\n\n"


class MeasurementAgent:
    """Test-result reader: returns the case's findings for a requested exam."""

    def __init__(self, scenario: Any, client: Any, model: str):
        self._client = client
        self._model = model
        self.agent_hist = ""
        self.information = scenario.exam_information()

    def system_prompt(self) -> str:
        base = (
            "You are an measurement reader who responds with medical test results. "
            'Please respond in the format "RESULTS: [results here]"'
        )
        presentation = (
            f"\n\nBelow is all of the information you have. {self.information}. \n\n "
            "If the requested results are not in your data then you can respond with "
            "NORMAL READINGS."
        )
        return base + presentation

    async def inference_measurement(self, question: str) -> str:
        prompt = (
            f"\nHere is a history of the dialogue: {self.agent_hist}\n "
            f"Here was the doctor measurement request: {question}"
        )
        answer = await _npc_chat(
            self._client, self._model, "measurement", self.system_prompt(), prompt
        )
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer

    def add_hist(self, hist_str: str) -> None:
        self.agent_hist += hist_str + "\n\n"


async def compare_results(client: Any, model: str, diagnosis: str, correct_diagnosis: str) -> bool:
    """Moderator: does the doctor's diagnosis name the same disease as gold?"""
    system = (
        "You are responsible for determining if the corrent diagnosis and the doctor diagnosis "
        "are the same disease. Please respond only with Yes or No. Nothing else."
    )
    prompt = (
        f"\nHere is the correct diagnosis: {correct_diagnosis}\n "
        f"Here was the doctor dialogue: {diagnosis}\nAre these the same?"
    )
    verdict = await _npc_chat(client, model, "moderator", system, prompt)
    return verdict.strip().lower() == "yes"
