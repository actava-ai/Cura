"""AgentClinic scenario data: fetched from the upstream repo, parsed into thin wrappers.

The four scenario JSONL files live in the upstream AgentClinic repo (MIT). They are
downloaded on first use into the cura_eval cache (override the ref with
``AGENTCLINIC_REV``, or point ``AGENTCLINIC_DIR`` at a local clone to skip downloads).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from cura_eval.loaders import cache_root

_UPSTREAM = "https://raw.githubusercontent.com/SamuelSchmidgall/AgentClinic/{rev}/{name}"

DATA_FILES = {
    "MedQA": "agentclinic_medqa.jsonl",
    "MedQA_Ext": "agentclinic_medqa_extended.jsonl",
    "NEJM": "agentclinic_nejm.jsonl",
    "NEJM_Ext": "agentclinic_nejm_extended.jsonl",
}

DATASETS = tuple(DATA_FILES)


class OSCEScenario:
    """MedQA / MedQA_Ext case: structured OSCE examination dict."""

    def __init__(self, scenario_dict: dict[str, Any]):
        self.scenario_dict = scenario_dict
        osce = scenario_dict["OSCE_Examination"]
        self.tests = osce["Test_Results"]
        self.diagnosis = osce["Correct_Diagnosis"]
        self.patient_info = osce["Patient_Actor"]
        self.examiner_info = osce["Objective_for_Doctor"]
        self.physical_exams = osce["Physical_Examination_Findings"]
        self.image_url = None

    def patient_information(self):
        return self.patient_info

    def examiner_information(self):
        return self.examiner_info

    def exam_information(self):
        exams = dict(self.physical_exams)
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self):
        return self.diagnosis


class NEJMScenario:
    """NEJM / NEJM_Ext image-challenge case."""

    def __init__(self, scenario_dict: dict[str, Any]):
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = next(a["text"] for a in scenario_dict["answers"] if a["correct"])
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self):
        return self.patient_info

    def examiner_information(self):
        return "What is the most likely diagnosis?"

    def exam_information(self):
        return self.physical_exams

    def diagnosis_information(self):
        return self.diagnosis


def _scenario_class(dataset: str):
    return NEJMScenario if dataset.startswith("NEJM") else OSCEScenario


def _data_path(dataset: str) -> Path:
    name = DATA_FILES[dataset]
    local_dir = os.environ.get("AGENTCLINIC_DIR")
    if local_dir:
        return Path(local_dir) / name
    rev = os.environ.get("AGENTCLINIC_REV", "main")
    dest = cache_root() / "agentclinic" / rev / name
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _UPSTREAM.format(rev=rev, name=name)
    req = urllib.request.Request(url, headers={"User-Agent": "cura-eval/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def load_scenarios(dataset: str) -> list[Any]:
    if dataset not in DATA_FILES:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(DATA_FILES)}")
    path = _data_path(dataset)
    cls = _scenario_class(dataset)
    with path.open() as f:
        return [cls(json.loads(line)) for line in f if line.strip()]


def mcq_match(diagnosis: str, scenario: Any) -> bool | None:
    """Loose extra signal for NEJM cases: does the utterance contain the gold option text?"""
    answers = getattr(scenario, "scenario_dict", {}).get("answers")
    if not answers:
        return None
    correct = next((a["text"] for a in answers if a.get("correct")), None)
    return bool(correct) and correct.lower() in diagnosis.lower()
