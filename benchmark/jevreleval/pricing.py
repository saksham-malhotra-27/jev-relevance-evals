"""Model pricing for the benchmark: map model ids to USD per 1M tokens.

OpenRouter exposes a public model catalog (``GET /api/v1/models``) listing each
model's ``pricing.prompt`` and ``pricing.completion`` in USD per token. We
resolve real prices from that catalog so per-call cost in the ledger is accurate,
and fall back to env-overridable rate constants when the catalog is unreachable.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from jevreleval.constants import (
    FALLBACK_INPUT_USD_PER_1M,
    FALLBACK_OUTPUT_USD_PER_1M,
    PRICING_RATE_ENV_INPUT,
    PRICING_RATE_ENV_OUTPUT,
    OPENROUTER_MODELS_URL,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceTable:
    """USD cost per 1M tokens for a set of model ids."""

    rates_usd_per_1m: dict[str, tuple[float, float]]
    source: str

    def price(self, model: str) -> tuple[float, float] | None:
        """Return (input USD/1M, output USD/1M) for ``model`` if known."""
        return self.rates_usd_per_1m.get(model)


def load_price_table() -> PriceTable:
    """Fetch the OpenRouter catalog, falling back to constant rates on failure."""
    env_input = _env_float(PRICING_RATE_ENV_INPUT)
    env_output = _env_float(PRICING_RATE_ENV_OUTPUT)
    if env_input is not None and env_output is not None:
        return PriceTable(
            rates_usd_per_1m={"*": (env_input, env_output)}, source="env"
        )
    try:
        body = urllib.request.urlopen(
            OPENROUTER_MODELS_URL, timeout=_CATALOG_TIMEOUT_S
        ).read()
        payload = json.loads(body)
        rates: dict[str, tuple[float, float]] = {}
        for entry in payload.get("data", []):
            model_id = entry.get("id")
            pricing = entry.get("pricing") or {}
            prompt = pricing.get("prompt")
            completion = pricing.get("completion")
            if model_id is None or prompt is None or completion is None:
                continue
            try:
                prompt_rate = float(prompt) * _USD_PER_TOKEN
                completion_rate = float(completion) * _USD_PER_TOKEN
            except (TypeError, ValueError):
                continue
            # OpenRouter reports "-1" for endpoints whose pricing is not public;
            # treat those as unknown so they fall back to the default rates.
            if prompt_rate < 0.0 or completion_rate < 0.0:
                continue
            rates[model_id] = (prompt_rate, completion_rate)
        return PriceTable(rates_usd_per_1m=rates, source="openrouter-catalog")
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as error:
        logger.warning(
            "OpenRouter catalog unavailable (%s); falling back to rate constants.",
            error,
        )
        return PriceTable(
            rates_usd_per_1m={"*": (FALLBACK_INPUT_USD_PER_1M, FALLBACK_OUTPUT_USD_PER_1M)},
            source="constants-fallback",
        )


def cost_usd(
    price_table: PriceTable, model: str, input_tokens: int, output_tokens: int
) -> float:
    """USD cost of one call, using the model's exact rate or a fallback."""
    rate = price_table.price(model)
    if rate is None:
        rate = price_table.rates_usd_per_1m.get(
            "*", (FALLBACK_INPUT_USD_PER_1M, FALLBACK_OUTPUT_USD_PER_1M)
        )
    input_rate, output_rate = rate
    return (input_tokens / _TOKENS_PER_UNIT) * input_rate + (
        output_tokens / _TOKENS_PER_UNIT
    ) * output_rate


#: OpenRouter reports pricing per 1 token; we keep per 1M in constants.
_USD_PER_TOKEN = 1e-6
_TOKENS_PER_UNIT = 1_000_000


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for env var %s=%r; ignoring.", name, value)
        return None


#: Catalog fetch timeout in seconds.
_CATALOG_TIMEOUT_S = 15.0


__all__ = ["PriceTable", "load_price_table", "cost_usd"]