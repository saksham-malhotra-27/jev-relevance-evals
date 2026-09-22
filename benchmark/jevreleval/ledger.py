"""Per-call costing and instrumentation for the benchmark pipes.

Both the Jev classifier and the LLM judge are wrapped so every underlying API
call is recorded with its real token usage and latency, then priced against the
OpenRouter price table. The ledger lives here because it is benchmark-owned;
the ``jev_relevance`` library is deliberately not instrumented.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from langchain_typesafe import ClassifierRequest, TypeSafeClassifier

from jevreleval.pricing import PriceTable, cost_usd

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """One priced API call made by the benchmark."""

    pipe: str
    model: str
    question_key: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float

    def as_dict(self) -> dict[str, object]:
        return {
            "pipe": self.pipe,
            "model": self.model,
            "question_key": self.question_key,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
        }


class InstrumentedClassifier(TypeSafeClassifier):
    """A TypeSafeClassifier that records every invoke/ainvoke call.

    Subclassing keeps the library's ``isinstance(classifier, TypeSafeClassifier)``
    guard satisfied, so the builder accepts it unchanged. ``price_table`` and
    ``records`` are declared pydantic fields (with the pydantic default) so both
    survive model construction; the record list is only ever mutated in place.
    """

    price_table: PriceTable
    records: list[CallRecord] = field(default_factory=list)

    def invoke(self, request: ClassifierRequest, config=None, **kwargs):
        start = time.perf_counter()
        response = super().invoke(request, config=config, **kwargs)
        self._record("jev", request, response, time.perf_counter() - start)
        return response

    async def ainvoke(self, request: ClassifierRequest, config=None, **kwargs):
        start = time.perf_counter()
        response = await super().ainvoke(request, config=config, **kwargs)
        self._record("jev", request, response, time.perf_counter() - start)
        return response

    def _record(
        self, pipe: str, request: ClassifierRequest, response, latency_s: float
    ) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        questions = request.get("questions", {}) if isinstance(request, dict) else request.questions
        self.records.append(
            CallRecord(
                pipe=pipe,
                model=self.model,
                question_key=f"{len(questions)} question(s)",
                latency_ms=latency_s * 1000.0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd(
                    self.price_table, self.model, input_tokens, output_tokens
                ),
            )
        )


@dataclass
class Ledger:
    """Collects every priced call across the benchmark pipes.

    Attributes:
        price_table: Price source used to cost each call.
        records: All priced calls (Jev + LLM judge) in execution order.
    """

    price_table: PriceTable
    records: list[CallRecord] = field(default_factory=list)

    def add(self, record: CallRecord) -> None:
        """Append one priced call."""
        self.records.append(record)

    def by_model(self, model: str) -> list[CallRecord]:
        """Calls originating from a specific model."""
        return [record for record in self.records if record.model == model]

    def summed(self) -> dict[str, object]:
        """Aggregate totals and latency percentiles across all calls."""
        return {
            "calls": len(self.records),
            "input_tokens": sum(record.input_tokens for record in self.records),
            "output_tokens": sum(record.output_tokens for record in self.records),
            "cost_usd": round(sum(record.cost_usd for record in self.records), 8),
            "latency_ms_p50": _percentile_ms(self.records, 0.50),
            "latency_ms_p95": _percentile_ms(self.records, 0.95),
        }


def _percentile_ms(records: list[CallRecord], fraction: float) -> float:
    if not records:
        return 0.0
    ordered = sorted(record.latency_ms for record in records)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return round(ordered[max(0, index)], 2)


def record_llm_judge_call(
    ledger: Ledger,
    *,
    model: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Add a priced LLM-judge call to the ledger."""
    ledger.add(
        CallRecord(
            pipe="llm-judge",
            model=model,
            question_key="-",
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(ledger.price_table, model, input_tokens, output_tokens),
        )
    )


__all__ = ["CallRecord", "InstrumentedClassifier", "Ledger", "record_llm_judge_call"]