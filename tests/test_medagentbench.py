import json

from conftest import FakeOpenAI, make_message
from cura_eval.benchmarks.medagentbench import refsol, runner
from cura_eval.benchmarks.medagentbench.loaders import (
    MABExample,
    build_system_prompt,
    build_user_message,
    load_medagentbench,
)
from cura_eval.benchmarks.medagentbench.tools import TOOL_SPECS, EpisodeState, execute_tool_call
from cura_eval.client import ChatClient


def test_vendored_data_loads():
    examples = load_medagentbench("v1")
    assert len(examples) == 100
    assert examples[0].task_type == "task1"


def test_system_prompt_embeds_api_base_and_functions():
    prompt = build_system_prompt("http://localhost:8080/fhir/")
    assert "http://localhost:8080/fhir/" in prompt
    assert "fhir_get" in prompt and "finish" in prompt
    # the funcs JSON survived str.format: its literal {api_base} placeholders are intact
    assert "{api_base}" in prompt


def test_user_message_with_and_without_context():
    ex = MABExample(id="task1_1", instruction="Q", context="C", sol=None, eval_MRN="", raw={})
    assert build_user_message(ex) == "Context: C\nQuestion: Q"
    ex2 = MABExample(id="task1_1", instruction="Q", context="", sol=None, eval_MRN="", raw={})
    assert build_user_message(ex2) == "Question: Q"


async def test_execute_tool_call_records_posts_and_finish():
    state = EpisodeState()
    obs = await execute_tool_call(
        {
            "id": "c1",
            "function": {
                "name": "fhir_post",
                "arguments": json.dumps({"url": "http://x/Observation", "payload": {"a": 1}}),
            },
        },
        state,
        "http://x/",
    )
    assert obs == "POST request accepted and executed successfully."
    assert state.posts == [("http://x/Observation", {"a": 1})]

    obs = await execute_tool_call(
        {"id": "c2", "function": {"name": "finish", "arguments": '{"answer": [6.2]}'}},
        state,
        "http://x/",
    )
    assert state.finished and state.result == "[6.2]"


async def test_execute_tool_call_rejects_bad_arguments():
    state = EpisodeState()
    obs = await execute_tool_call(
        {"id": "c", "function": {"name": "finish", "arguments": "not json"}}, state, "http://x/"
    )
    assert obs.startswith("Error")
    assert not state.finished


def test_refsol_task1_grades_answer_and_forbids_posts():
    case = {"id": "task1_1", "sol": ["S6534835"]}
    ok = EpisodeState(result='["S6534835"]')
    assert refsol.grade(case, ok, "http://x/") == (True, ["S6534835"])

    wrong = EpisodeState(result='["nope"]')
    assert refsol.grade(case, wrong, "http://x/")[0] is False

    posted = EpisodeState(result='["S6534835"]', posts=[("u", {})])
    assert refsol.grade(case, posted, "http://x/")[0] is False


def test_refsol_task3_validates_post_payload():
    case = {"id": "task3_1", "eval_MRN": "S123"}
    payload = {
        "resourceType": "Observation",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://hl7.org/fhir/observation-category",
                        "code": "vital-signs",
                        "display": "Vital Signs",
                    }
                ]
            }
        ],
        "code": {"text": "BP"},
        "effectiveDateTime": "2023-11-13T10:15:00+00:00",
        "status": "final",
        "valueString": "118/77 mmHg",
        "subject": {"reference": "Patient/S123"},
    }
    good = EpisodeState(result="[]", posts=[("http://x/Observation", payload)])
    assert refsol.grade(case, good, "http://x/")[0] is True

    bad = EpisodeState(
        result="[]", posts=[("http://x/Observation", {**payload, "status": "draft"})]
    )
    assert refsol.grade(case, bad, "http://x/")[0] is False


async def test_runner_episode_nudges_then_finishes(monkeypatch):
    monkeypatch.setattr(runner, "verify_fhir_server", lambda base: True)
    fake = FakeOpenAI(
        [
            make_message("let me think", reasoning_content="hmm"),  # no tool call -> nudge
            make_message(
                None,
                tool_calls=[
                    {
                        "id": "c1",
                        "function": {"name": "finish", "arguments": '{"answer": ["S6534835"]}'},
                    }
                ],
            ),
        ]
    )
    candidate = ChatClient(model="fake", client=fake)
    rows, summary = await runner.run_medagentbench(
        candidate, fhir_api_base="http://x/", limit=1, max_round=4
    )
    assert summary["score"] == 1.0 and summary["n"] == 1
    (row,) = rows
    assert row["score"]["correct"] is True
    assert row["response"] == '["S6534835"]'
    assert row["tools"] == TOOL_SPECS
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "tool"]
    # the tool loop passed the tool specs to the model
    assert fake.calls[0]["tools"] == TOOL_SPECS
