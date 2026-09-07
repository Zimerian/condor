from types import SimpleNamespace

from condor.acp.pydantic_ai_client import PydanticAIClient


def test_capture_run_telemetry_keeps_secret_free_usage_and_response_id() -> None:
    client = PydanticAIClient("openrouter:openai/gpt-5-mini")
    run = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1200,
            output_tokens=180,
            cost=0.0032,
        )
    )
    messages = [
        SimpleNamespace(provider_response_id=None),
        SimpleNamespace(provider_response_id="openrouter-response-7"),
    ]

    client._capture_run_telemetry(run, messages)

    assert client.last_usage == {
        "prompt_tokens": 1200,
        "completion_tokens": 180,
        "total_tokens": 1380,
        "cost_usd": "0.0032",
    }
    assert client.last_response_id == "openrouter-response-7"


def test_capture_run_telemetry_keeps_missing_usage_explicit() -> None:
    client = PydanticAIClient("openrouter:openai/gpt-5-mini")
    run = SimpleNamespace(usage=SimpleNamespace())
    messages = [SimpleNamespace(provider_response_id=None)]

    client._capture_run_telemetry(run, messages)

    assert client.last_usage == {}
    assert client.last_response_id is None
