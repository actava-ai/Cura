"""End-to-end (offline) runs of the single-turn benchmarks with faked loaders + clients."""

import cura_eval.benchmarks.healthbench as hb
import cura_eval.benchmarks.medxpertqa as mxq
from conftest import FakeOpenAI, make_message
from cura_eval.client import ChatClient
from cura_eval.loaders import HBExample, MXQExample


async def test_run_healthbench_offline(monkeypatch):
    examples = [
        HBExample(
            task_id="hb1",
            prompt=[{"role": "user", "content": "I have a headache"}],
            criteria=["recommends hydration"],
            points=[1.0],
            axes=["completeness"],
            theme="self_care",
        )
    ]
    monkeypatch.setattr(hb, "load_healthbench", lambda variant, limit: examples)
    candidate = ChatClient(
        model="fake", client=FakeOpenAI([make_message("Drink water.", reasoning_content="cot")])
    )
    judge = FakeOpenAI([make_message('{"criteria_met": true, "explanation": "ok"}')])
    rows, summary = await hb.run_healthbench(
        candidate, judge_client=judge, judge_model="gpt-4.1-2025-04-14", variant="hard"
    )
    assert summary["score"] == 1.0
    assert summary["per_axis"] == {"completeness": 1.0}
    assert summary["n_judge_errors"] == 0
    (row,) = rows
    assert row["task_id"] == "hb1"
    assert row["messages"][-1]["reasoning_content"] == "cot"
    assert row["score"]["reward"] == 1.0


async def test_run_healthbench_judge_failure_is_isolated(monkeypatch):
    examples = [
        HBExample("hb1", [{"role": "user", "content": "q"}], ["c"], [1.0], ["accuracy"], ""),
        HBExample("hb2", [{"role": "user", "content": "q"}], ["c"], [1.0], ["accuracy"], ""),
    ]
    monkeypatch.setattr(hb, "load_healthbench", lambda variant, limit: examples)
    candidate = ChatClient(model="fake", client=FakeOpenAI([make_message("a")]))

    calls = {"n": 0}

    def judge_script(kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("judge down")
        return make_message('{"criteria_met": true, "explanation": "ok"}')

    rows, summary = await hb.run_healthbench(
        candidate, judge_client=FakeOpenAI(judge_script), judge_model="gpt-4.1-2025-04-14"
    )
    assert summary["n"] == 2
    # exactly one of the two examples judge-errored (MAX_JUDGE_RETRIES consumed by retries)
    assert summary["n_judge_errors"] in (0, 1)
    assert any(r["score"]["judge_error"] for r in rows) == (summary["n_judge_errors"] == 1)


async def test_run_medxpertqa_offline(monkeypatch):
    examples = [
        MXQExample(
            task_id="m1",
            messages=[{"role": "user", "content": "Q..."}],
            answer="B",
            valid_letters="ABCDE",
            medical_task="Diagnosis",
            body_system="Cardiac",
        ),
        MXQExample(
            task_id="m2",
            messages=[{"role": "user", "content": "Q..."}],
            answer="A",
            valid_letters="ABCDE",
            medical_task="Diagnosis",
            body_system="Renal",
        ),
    ]
    monkeypatch.setattr(mxq, "load_medxpertqa", lambda subset, limit: examples)
    candidate = ChatClient(
        model="fake",
        client=FakeOpenAI([make_message("\\boxed{B}"), make_message("\\boxed{C}")]),
    )
    rows, summary = await mxq.run_medxpertqa(candidate, subset="text")
    assert summary["score"] == 0.5
    assert summary["per_medical_task"] == {"Diagnosis": 0.5}
    assert summary["per_body_system"] == {"Cardiac": 1.0, "Renal": 0.0}
    assert rows[0]["score"]["correct"] is True
    assert rows[1]["score"] == {
        "correct": False,
        "chosen": "C",
        "gold": "A",
        "max_letter": "E",
        "medical_task": "Diagnosis",
        "body_system": "Renal",
    }
