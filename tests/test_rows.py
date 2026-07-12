from cura_eval.rows import assistant_message, breakdown, response_row, run_stem, strip_images


def test_assistant_message_omits_empty_reasoning():
    assert assistant_message("hi") == {"role": "assistant", "content": "hi"}
    assert assistant_message("hi", "") == {"role": "assistant", "content": "hi"}
    assert assistant_message("hi", "cot") == {
        "role": "assistant",
        "reasoning_content": "cot",
        "content": "hi",
    }


def test_response_row_appends_assistant_and_orders_keys():
    row = response_row(
        [{"role": "user", "content": "q"}],
        assistant_message("a", "cot"),
        example_id="t1",
        id_key="task_id",
        response="a",
        score={"correct": True},
    )
    assert list(row) == ["task_id", "messages", "response", "score"]
    assert row["messages"][-1]["reasoning_content"] == "cot"
    assert len(row["messages"]) == 2


def test_breakdown_ignores_empty_labels():
    result = breakdown([("a", 1.0), ("a", 0.0), ("b", 1.0), (None, 1.0), ("", 1.0)])
    assert result == {"a": 0.5, "b": 1.0}


def test_strip_images_replaces_image_parts_only():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "q"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxxx"}},
            ],
        },
        {"role": "assistant", "content": "a"},
    ]
    stripped = strip_images(messages)
    assert stripped[0]["content"][0] == {"type": "text", "text": "q"}
    assert stripped[0]["content"][1] == {"type": "text", "text": "[image omitted]"}
    assert stripped[1] == {"role": "assistant", "content": "a"}
    # original untouched
    assert messages[0]["content"][1]["type"] == "image_url"


def test_run_stem_slugs_model_ids():
    stem = run_stem("medxpertqa_text", "actava/cura-soar")
    assert stem.startswith("medxpertqa_text_actava-cura-soar_")
