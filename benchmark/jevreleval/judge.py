"""Baseline LLM judge and metrics for the benchmark.

The LLM judge is deliberately *cheap*: it re-ranks the same BM25 candidates with
an LLM instead of Jev, giving a reference pipe to compare Jev's cost/latency
against. It is a benchmark fixture, NOT part of the jev_relevance library.
"""

from __future__ import annotations

import logging
import os
import time

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from jevreleval.constants import (
    JUDGE_MODEL_DEFAULT,
    JUDGE_MODEL_ENV,
    JUDGE_SOURCE_LLM,
    OPENROUTER_API_KEY_ENV,
)

logger = logging.getLogger(__name__)

#: Judge prompt: list candidate ids, ask for the relevance-relevant ones.
JUDGE_PROMPT = (
    "You are an exactness filter for retrieval-augmented generation.\n"
    "Given a query and candidate passages, return the ids of passages that "
    "actually answer the query. Do not include tangential or irrelevant ones.\n\n"
    "QUERY: {query}\n\nCANDIDATES:\n{candidates}\n\n"
    "ANSWER: a comma-separated list of ids only."
)


def build_judge_model() -> ChatOpenAI | None:
    """Create the baseline LLM, or None when no OpenRouter key is configured.

    The benchmark never blocks on a missing key: LLM-judge rows are simply
    skipped and only the Jev pipe runs.
    """
    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        logger.warning("OPENROUTER_API_KEY unset; LLM judge baseline disabled.")
        return None
    model = os.environ.get(JUDGE_MODEL_ENV, JUDGE_MODEL_DEFAULT)
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model=model,
        temperature=0.0,
    )


def judge_llm_baseline(
    judge: ChatOpenAI, query: str, candidates: list[Document]
) -> tuple[list[str], float, object]:
    """Run the LLM judge and return (kept ids, wall seconds, response).

    The judge is asked for candidate ids; the candidate block numbers each
    candidate (``0. [id]``), so any numeric token the model echoes back is
    mapped to the corresponding candidate id. Tokens that match neither an
    index nor a known id are dropped.
    """
    id_by_index: dict[str, str] = {
        str(index): str(document.metadata.get("id", "unknown"))
        for index, document in enumerate(candidates)
    }
    known_ids = set(id_by_index.values())
    start = time.perf_counter()
    candidate_block = "\n".join(
        f"{index}. [{document.metadata.get('id', 'unknown')}]\n{document.page_content}"
        for index, document in enumerate(candidates)
    )
    response = judge.invoke(
        JUDGE_PROMPT.format(query=query, candidates=candidate_block)
    )
    elapsed = time.perf_counter() - start
    text = response.content if isinstance(response.content, str) else str(response.content)
    tokens = [part.strip() for part in text.replace("\n", ",").split(",") if part.strip()]
    kept: list[str] = []
    for token in tokens:
        doc_id = id_by_index.get(token, token)
        if doc_id in known_ids:
            kept.append(doc_id)
    return kept, elapsed, response


__all__ = ["build_judge_model", "judge_llm_baseline", "JUDGE_SOURCE_LLM"]