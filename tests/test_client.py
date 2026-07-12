from conftest import FakeOpenAI, make_message
from cura_eval.client import ChatClient


async def test_captures_reasoning_content():
    fake = FakeOpenAI([make_message("answer", reasoning_content="chain of thought")])
    client = ChatClient(model="m", client=fake)
    result = await client.complete([{"role": "user", "content": "q"}])
    assert result.content == "answer"
    assert result.reasoning == "chain of thought"
    assert result.assistant_message() == {
        "role": "assistant",
        "reasoning_content": "chain of thought",
        "content": "answer",
    }


async def test_captures_openrouter_reasoning_field():
    fake = FakeOpenAI([make_message("answer", reasoning="trace")])
    client = ChatClient(model="m", client=fake)
    result = await client.complete([{"role": "user", "content": "q"}])
    assert result.reasoning == "trace"


async def test_tool_calls_become_plain_dicts():
    fake = FakeOpenAI(
        [
            make_message(
                None,
                tool_calls=[
                    {"id": "c1", "function": {"name": "finish", "arguments": '{"answer": []}'}}
                ],
            )
        ]
    )
    client = ChatClient(model="m", client=fake)
    result = await client.complete([{"role": "user", "content": "q"}], tools=[{"type": "function"}])
    assert result.content == ""
    assert result.tool_calls == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "finish", "arguments": '{"answer": []}'},
        }
    ]
    assert result.assistant_message()["tool_calls"] == result.tool_calls


async def test_retries_transient_errors():
    fake = FakeOpenAI([RuntimeError("boom"), make_message("recovered")])
    client = ChatClient(model="m", client=fake, max_retries=3)
    result = await client.complete([{"role": "user", "content": "q"}])
    assert result.content == "recovered"
    assert len(fake.calls) == 2


async def test_omits_unset_sampling_params_and_forwards_extra_body():
    fake = FakeOpenAI([make_message("ok")])
    client = ChatClient(model="m", client=fake, extra_body={"thinking": {"type": "enabled"}})
    await client.complete([{"role": "user", "content": "q"}], max_tokens=64)
    (call,) = fake.calls
    assert call["max_tokens"] == 64
    assert "temperature" not in call
    assert call["extra_body"] == {"thinking": {"type": "enabled"}}
