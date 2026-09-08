"""Secret-free OpenRouter key-usage readback for budget reconciliation."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import httpx


class OpenRouterUsageError(RuntimeError):
    """Raised when authoritative key usage cannot be read safely."""


def _decimal_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if parsed < 0:
        return None
    return str(parsed)


async def fetch_openrouter_key_usage(
    api_key: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, str | None]:
    """Return only secret-free key accounting fields from OpenRouter."""

    if not api_key.strip():
        raise OpenRouterUsageError("OpenRouter API key is unavailable")

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=10.0)
    try:
        response = await http.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise OpenRouterUsageError(
            f"OpenRouter key usage readback failed ({type(exc).__name__})"
        ) from exc
    finally:
        if owns_client:
            await http.aclose()

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise OpenRouterUsageError("OpenRouter key usage response was malformed")

    result: dict[str, str | None] = {
        "usage": _decimal_text(data.get("usage")),
        "usage_daily": _decimal_text(data.get("usage_daily")),
        "usage_weekly": _decimal_text(data.get("usage_weekly")),
        "usage_monthly": _decimal_text(data.get("usage_monthly")),
        "limit": _decimal_text(data.get("limit")),
        "limit_remaining": _decimal_text(data.get("limit_remaining")),
        "limit_reset": (
            str(data.get("limit_reset"))
            if isinstance(data.get("limit_reset"), str)
            else None
        ),
    }
    if result["usage_daily"] is None or result["usage_monthly"] is None:
        raise OpenRouterUsageError(
            "OpenRouter key usage response omitted daily or monthly usage"
        )
    return result
