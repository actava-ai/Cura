from conftest import FakeOpenAI, make_message
from cura_eval.benchmarks.agentclinic.doctor import build_doctor_system_prompt
from cura_eval.benchmarks.agentclinic.episode import run_episode
from cura_eval.benchmarks.agentclinic.npc import MeasurementAgent, PatientAgent, compare_results
from cura_eval.benchmarks.agentclinic.scenarios import NEJMScenario, OSCEScenario, mcq_match

OSCE_DICT = {
    "OSCE_Examination": {
        "Test_Results": {"Blood": "normal"},
        "Correct_Diagnosis": "Appendicitis",
        "Patient_Actor": {"Demographics": "31yo", "Symptoms": "RLQ pain"},
        "Objective_for_Doctor": "Assess acute abdominal pain",
        "Physical_Examination_Findings": {"Abdomen": "guarding"},
    }
}

NEJM_DICT = {
    "question": "What is the diagnosis?",
    "image_url": "https://example.org/case.jpg",
    "answers": [
        {"text": "Melanoma", "correct": True},
        {"text": "Nevus", "correct": False},
    ],
    "patient_info": "55yo with a pigmented lesion",
    "physical_exams": "irregular borders",
}


def test_osce_scenario_accessors():
    s = OSCEScenario(OSCE_DICT)
    assert s.diagnosis_information() == "Appendicitis"
    assert s.examiner_information() == "Assess acute abdominal pain"
    exams = s.exam_information()
    assert exams["tests"] == {"Blood": "normal"}
    assert exams["Abdomen"] == "guarding"
    assert s.image_url is None


def test_nejm_scenario_accessors_and_mcq_match():
    s = NEJMScenario(NEJM_DICT)
    assert s.diagnosis_information() == "Melanoma"
    assert s.examiner_information() == "What is the most likely diagnosis?"
    assert mcq_match("DIAGNOSIS READY: melanoma of the skin", s) is True
    assert mcq_match("DIAGNOSIS READY: nevus", s) is False
    assert mcq_match("anything", OSCEScenario(OSCE_DICT)) is None


def test_doctor_system_prompt_mentions_budget_and_objective():
    prompt = build_doctor_system_prompt(OSCEScenario(OSCE_DICT), 20, image_request=False)
    assert "ask 20 questions total" in prompt
    assert "Assess acute abdominal pain" in prompt
    assert "REQUEST IMAGES" not in prompt
    assert "REQUEST IMAGES" in build_doctor_system_prompt(
        OSCEScenario(OSCE_DICT), 20, image_request=True
    )


async def test_compare_results_strict_yes():
    assert await compare_results(FakeOpenAI([make_message("Yes")]), "gpt-5.5", "d", "g") is True
    assert await compare_results(FakeOpenAI([make_message("No")]), "gpt-5.5", "d", "g") is False
    assert (
        await compare_results(FakeOpenAI([make_message("Yes, same disease")]), "gpt-5.5", "d", "g")
        is False
    )


class _ScriptedDoctor:
    def __init__(self, utterances):
        self._utterances = list(utterances)
        self.observations = []

    async def step(self, observation, image_uri):
        self.observations.append(observation)
        from cura_eval.benchmarks.agentclinic.doctor import DoctorTurn

        return DoctorTurn(utterance=self._utterances.pop(0), reasoning=None)


async def test_episode_routes_tests_then_grades_diagnosis():
    scenario = OSCEScenario(OSCE_DICT)
    npc_fake = FakeOpenAI([make_message("RESULTS: guarding present")])
    judge_fake = FakeOpenAI([make_message("Yes")])
    doctor = _ScriptedDoctor(["REQUEST TEST: Abdominal_Exam", "DIAGNOSIS READY: Appendicitis"])
    result = await run_episode(
        scenario,
        "MedQA",
        doctor=doctor,
        patient_agent=PatientAgent(scenario, npc_fake, "gpt-5.5"),
        measurement_agent=MeasurementAgent(scenario, npc_fake, "gpt-5.5"),
        judge_client=judge_fake,
        judge_model="gpt-5.5",
        total_inferences=5,
    )
    assert result.reward == 1.0
    assert result.logs["stopped"] == "diagnosis"
    assert result.logs["n_turns"] == 2
    assert [t["role"] for t in result.turns] == ["doctor", "measurement", "doctor"]
    # measurement observation was routed back to the doctor
    assert "RESULTS" in doctor.observations[1]


async def test_episode_final_turn_nudges_for_diagnosis():
    scenario = OSCEScenario(OSCE_DICT)
    npc_fake = FakeOpenAI([make_message("It hurts on the right side.")])
    judge_fake = FakeOpenAI([make_message("No")])
    doctor = _ScriptedDoctor(["Where does it hurt?", "DIAGNOSIS READY: Gastritis"])
    result = await run_episode(
        scenario,
        "MedQA",
        doctor=doctor,
        patient_agent=PatientAgent(scenario, npc_fake, "gpt-5.5"),
        measurement_agent=MeasurementAgent(scenario, npc_fake, "gpt-5.5"),
        judge_client=judge_fake,
        judge_model="gpt-5.5",
        total_inferences=2,
    )
    assert result.reward == 0.0
    assert "final question" in doctor.observations[-1]


async def test_npc_prompts_carry_history_and_role_shape():
    scenario = OSCEScenario(OSCE_DICT)
    fake = FakeOpenAI([make_message("I feel a sharp pain.")])
    patient = PatientAgent(scenario, fake, "gpt-4.1")
    answer = await patient.inference_patient("Where does it hurt?")
    assert answer == "I feel a sharp pain."
    (call,) = fake.calls
    system = call["messages"][0]["content"]
    assert "must not reveal your disease" in system
    assert "RLQ pain" in system  # patient info formatted into the system prompt
    assert call["temperature"] == 0.0  # non-reasoning NPC model pinned to 0
    assert "Where does it hurt?" in patient.agent_hist
