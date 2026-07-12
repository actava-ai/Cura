from conftest import FakeOpenAI, make_message
from cura_eval.judge import (
    _parse_judge_json,
    compute_reward,
    grade_hbp,
    grade_healthbench,
    is_reasoning_model,
    length_adjust,
)


def test_compute_reward_positive_points_only_in_denominator():
    # earned 3 of 5 positive points, plus a met negative criterion (-2)
    assert compute_reward([3.0, 2.0, -2.0], [True, False, True]) == (3.0 - 2.0) / 5.0


def test_compute_reward_clamps_to_unit_interval():
    assert compute_reward([1.0, -5.0], [True, True]) == 0.0
    assert compute_reward([], []) == 0.0


def test_length_adjust_free_below_threshold():
    assert length_adjust(0.8, 2000) == 0.8
    assert length_adjust(0.8, 2001) < 0.8


def test_parse_judge_json_accepts_fenced_and_rejects_malformed():
    assert _parse_judge_json('```json\n{"criteria_met": true}\n```')["criteria_met"] is True
    assert _parse_judge_json('{"criteria_met": "yes"}') is None
    assert _parse_judge_json("not json") is None


def test_is_reasoning_model_strips_provider_prefix():
    assert is_reasoning_model("openai/gpt-5.4")
    assert is_reasoning_model("o3-mini")
    assert not is_reasoning_model("gpt-4.1-2025-04-14")


async def test_grade_healthbench_end_to_end_with_fake_judge():
    fake = FakeOpenAI(
        [
            make_message('{"criteria_met": true, "explanation": "met"}'),
            make_message('{"criteria_met": false, "explanation": "not met"}'),
        ]
    )
    result = await grade_healthbench(
        [{"role": "user", "content": "q"}],
        "completion",
        criteria=["does A", "does B"],
        points=[2.0, 2.0],
        client=fake,
        model="gpt-4.1-2025-04-14",
    )
    assert result["reward"] == 0.5
    assert len(result["judgments"]) == 2
    # non-reasoning judge is pinned to temperature 0
    assert all(c["temperature"] == 0.0 for c in fake.calls)


async def test_grade_hbp_reports_raw_and_adjusted():
    fake = FakeOpenAI([make_message('{"criteria_met": true, "explanation": "met"}')])
    result = await grade_hbp(
        [{"role": "user", "content": "q"}],
        "short answer",
        rubric_items=[{"criterion_text": "does A", "points": 1.0}],
        client=fake,
        model="gpt-5.4",
    )
    assert result["raw"] == 1.0
    assert result["adjusted"] == 1.0  # under the free-length threshold
    assert fake.calls[0]["reasoning_effort"] == "low"


async def test_judge_parse_failure_scores_criterion_unmet():
    fake = FakeOpenAI([make_message("never json")])
    result = await grade_healthbench(
        [{"role": "user", "content": "q"}],
        "completion",
        criteria=["does A"],
        points=[1.0],
        client=fake,
        model="gpt-4.1-2025-04-14",
    )
    assert result["reward"] == 0.0
    assert result["judgments"][0]["explanation"] == "JUDGE_PARSE_FAILED"
