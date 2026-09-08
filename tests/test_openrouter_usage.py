from __future__ import annotations

import httpx
import pytest

from condor.openrouter_usage import OpenRouterUsageError, fetch_openrouter_key_usage


@pytest.mark.asyncio
async def test_openrouter_key_usage_returns_only_secret_free_accounting_fields() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-secret"
        return httpx.Response(
            200,
            json={
                "data": {
                    "usage": 1.25,
                    "usage_daily": 0.289,
                    "usage_weekly": 0.8,
                    "usage_monthly": 1.25,
                    "limit": 2,
                    "limit_remaining": 1.711,
                    "limit_reset": "daily",
                    "label": "AITrader-Guarded",
                    "hash": "must-not-leak",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await fetch_openrouter_key_usage("test-secret", client=client)

    assert result == {
        "usage": "1.25",
        "usage_daily": "0.289",
        "usage_weekly": "0.8",
        "usage_monthly": "1.25",
        "limit": "2",
        "limit_remaining": "1.711",
        "limit_reset": "daily",
    }
    assert "secret" not in str(result)
    assert "hash" not in result
    assert "label" not in result


@pytest.mark.asyncio
async def test_openrouter_key_usage_fails_closed_when_daily_usage_missing() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"usage_monthly": 1.25}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OpenRouterUsageError):
            await fetch_openrouter_key_usage("test-secret", client=client)
