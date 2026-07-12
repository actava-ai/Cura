from cura_eval.loaders import _extract_images, _hb_parse, _hbp_parse, _mxq_prompt


def test_hb_parse_extracts_rubric_axes_and_theme():
    ex = _hb_parse(
        {
            "prompt_id": "p1",
            "prompt": [{"role": "user", "content": "q"}],
            "rubrics": [
                {"criterion": "does A", "points": 5.0, "tags": ["axis:accuracy", "other"]},
                {"criterion": "avoids B", "points": -2.0, "tags": []},
            ],
            "example_tags": ["theme:emergency_referrals"],
        }
    )
    assert ex.task_id == "p1"
    assert ex.criteria == ["does A", "avoids B"]
    assert ex.points == [5.0, -2.0]
    assert ex.axes == ["accuracy", "unknown"]
    assert ex.theme == "emergency_referrals"


def test_hb_parse_wraps_string_prompt():
    ex = _hb_parse({"prompt_id": "p2", "prompt": "hello", "rubrics": []})
    assert ex.prompt == [{"role": "user", "content": "hello"}]


def test_hbp_parse():
    ex = _hbp_parse(
        {
            "id": 7,
            "conversation": {"messages": [{"role": "user", "content": "q"}]},
            "rubric_items": [{"criterion_text": "c", "points": 1.0}],
            "use_case": "triage",
            "type": "clinical",
            "difficulty": "hard",
            "specialty": "cardiology",
        }
    )
    assert ex.task_id == "7"
    assert ex.rubric_items[0]["points"] == 1.0
    assert ex.specialty == "cardiology"


def test_mxq_prompt_lists_options_and_bounds_letters():
    prompt = _mxq_prompt("Q?", {"B": "beta", "A": "alpha"}, "E", mm=False)
    assert "A. alpha\nB. beta" in prompt
    assert "from A through E" in prompt
    assert "\\boxed{X}" in prompt
    assert "image(s)" not in prompt
    assert "image(s)" in _mxq_prompt("Q?", {"A": "x"}, "E", mm=True)


def test_extract_images_handles_strings_and_dicts():
    assert _extract_images(["a.png", {"filename": "b.png"}, {"path": "c.png"}, {}]) == [
        "a.png",
        "b.png",
        "c.png",
    ]
    assert _extract_images(None) == []
