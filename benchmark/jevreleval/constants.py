"""Constants for the jevreleval benchmark.

Kept separate from ``jev_relevance.constants``: the benchmark is a standalone
companion and must be runnable even if the library defaults change.
"""

from __future__ import annotations

#: Number of BM25 candidates handed to each relevance method. Mirrors the
#: library default of 20, but independent so the two sides can differ.
BENCHMARK_CANDIDATE_K: int = 20

#: Relevance threshold applied to the Jev pipeline in benchmark runs.
BENCHMARK_THRESHOLD: float = 0.5

#: Cheap default judge/baseline model exposed over OpenRouter. Users should
#: override with OPENROUTER_MODEL when they have a favourite.
JUDGE_MODEL_DEFAULT: str = "openrouter/auto"

#: Judge prompt family id stored in each result row.
JUDGE_SOURCE_LLM: str = "llm-judge"
JUDGE_SOURCE_JEV: str = "jev"

#: OpenRouter env var used for the baseline LLM judge.
OPENROUTER_API_KEY_ENV: str = "OPENROUTER_API_KEY"
#: TypeSafe env var used for the Jev pipeline.
TYPESAFE_API_KEY_ENV: str = "TYPESAFE_API_KEY"
#: Optional override for the baseline judge model name.
JUDGE_MODEL_ENV: str = "JUDGE_MODEL"

# ---------------------------------------------------------------------------
# Pricing / cost bookkeeping
# ---------------------------------------------------------------------------

#: OpenRouter public model catalog used to look up real per-token prices.
OPENROUTER_MODELS_URL: str = "https://openrouter.ai/api/v1/models"

#: Fallback USD per 1M tokens used when the catalog cannot be fetched. These are
#: rough defaults; set the env overrides below to reflect your actual bill.
FALLBACK_INPUT_USD_PER_1M: float = 0.15
FALLBACK_OUTPUT_USD_PER_1M: float = 0.60

#: Optional env overrides for the fallback rates (input/output USD per 1M tokens).
PRICING_RATE_ENV_INPUT: str = "JEV_BENCH_INPUT_USD_PER_1M"
PRICING_RATE_ENV_OUTPUT: str = "JEV_BENCH_OUTPUT_USD_PER_1M"

#: Where JSON per-run results land, relative to the benchmark root.
RESULTS_DIR_NAME: str = "results"

#: JSON result file naming: {dataset}-{YYYYmmddHHMM}.json
RESULT_JSON_PATTERN: str = "{dataset}-{timestamp}.json"

#: Built-in demo corpus shipped with the benchmark (id -> text). A tiny
#: stand-in for a real dataset used by ``--dataset demo``.
DEMO_CORPUS: tuple[tuple[str, str], ...] = (
    (
        "demo-eiffel",
        "The Eiffel Tower is an iron lattice tower on the Champ de Mars in Paris.",
    ),
    (
        "demo-python",
        "Python is a programming language prized for its readable syntax.",
    ),
    (
        "demo-moon",
        "The Moon is Earth's only natural satellite and is not made of cheese.",
    ),
    (
        "demo-llm",
        "Large language models generate text by predicting the next token.",
    ),
    (
        "demo-rag",
        "Retrieval augmented generation grounds model answers in retrieved passages.",
    ),
)

#: Built-in demo queries: (query, list of ids that genuinely answer it).
DEMO_QA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Where is the Eiffel Tower located?", ("demo-eiffel",)),
    ("What makes Python popular as a language?", ("demo-python",)),
    ("Is the Moon made of cheese?", ("demo-moon",)),
    ("What does RAG stand for and do?", ("demo-rag", "demo-llm")),
)


__all__ = [
    "BENCHMARK_CANDIDATE_K",
    "BENCHMARK_THRESHOLD",
    "JUDGE_MODEL_DEFAULT",
    "JUDGE_SOURCE_LLM",
    "JUDGE_SOURCE_JEV",
    "OPENROUTER_API_KEY_ENV",
    "TYPESAFE_API_KEY_ENV",
    "JUDGE_MODEL_ENV",
    "OPENROUTER_MODELS_URL",
    "FALLBACK_INPUT_USD_PER_1M",
    "FALLBACK_OUTPUT_USD_PER_1M",
    "PRICING_RATE_ENV_INPUT",
    "PRICING_RATE_ENV_OUTPUT",
    "RESULTS_DIR_NAME",
    "RESULT_JSON_PATTERN",
    "DEMO_CORPUS",
    "DEMO_QA",
]