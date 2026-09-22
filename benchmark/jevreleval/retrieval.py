"""BM25 candidate retrieval for the benchmark.

A plain keyword retriever is an ideal base for measuring the *added value* of
Jev relevance filtering: BM25 finds related documents, then the filter decides
which ones actually answer the query.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi


class BM25Retriever(BaseRetriever):
    """Tiny BM25 retriever over an in-memory corpus."""

    documents: list[Document]
    k: int = 20
    _bm25: Any = None

    def model_post_init(self, __context: Any) -> None:
        """Build the BM25 index once the pydantic model is validated."""
        self._bm25 = BM25Okapi(
            [_tokenize(document.page_content) for document in self.documents]
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        tokens = _tokenize(query)
        if not tokens or self._bm25 is None:
            return []
        ranked = self._bm25.get_top_n(
            tokens, self.documents, n=min(self.k, len(self.documents))
        )
        return ranked


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization, sufficient for BM25 over plain text."""
    return re.findall(r"[a-z0-9']+", text.lower())


__all__ = ["BM25Retriever", "_tokenize"]