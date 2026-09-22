"""Benchmark entry point: BM25 baseline vs Jev relevance filter vs LLM judge.

Usage::

    python -m jevreleval --dataset demo                 # run the tiny demo
    python -m jevreleval --dataset beir:nfcorpus        # full BEIR dataset

Every query runs the identical BM25 candidate pool through three pipes and
their results (ranked ids, metrics, latency, and per-call cost) are written to
a JSON file under ``benchmark/results/``; a summary table is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from jevreleval.constants import (
    BENCHMARK_CANDIDATE_K,
    BENCHMARK_THRESHOLD,
    OPENROUTER_API_KEY_ENV,
    RESULT_JSON_PATTERN,
    RESULTS_DIR_NAME,
)
from jevreleval.data import Dataset, load_dataset
from jevreleval.env import (
    ENV_FILE,
    apply_env_file,
    openrouter_api_key,
    typesafe_api_key,
)
from jevreleval.judge import build_judge_model, judge_llm_baseline
from jevreleval.ledger import (
    InstrumentedClassifier,
    Ledger,
    record_llm_judge_call,
)
from jevreleval.metrics import rank_metrics, summarize
from jevreleval.pricing import load_price_table
from jevreleval.retrieval import BM25Retriever

__version__ = "0.2.0"


def _document(doc_id: str, text: str) -> Document:
    return Document(page_content=text, metadata={"id": doc_id})


def _build_base_retriever(corpus: dict[str, str], candidate_k: int) -> BM25Retriever:
    """BM25 over the corpus; its output is the frozen candidate pool."""
    documents = [_document(doc_id, text) for doc_id, text in corpus.items()]
    return BM25Retriever(documents=documents, k=candidate_k)


def _build_jev_pipeline(
    base: BM25Retriever,
    *,
    provider: str,
    api_key: str,
    threshold: float,
    candidate_k: int,
    scoring_mode: str,
    price_table,
) -> tuple[BaseRetriever, InstrumentedClassifier]:
    """Wrap the BM25 base in the drop-in Jev relevance retriever.

    The classified is instrumented (so the benchmark can price every call) and
    handed to the builder as an already-built classifier — the library only ever
    receives credentials and network details as parameters.
    """
    from jev_relevance import JevRelevanceConfig, JevRelevanceRetrieverBuilder
    from jev_relevance.config import ProviderConfig
    from jev_relevance.constants import (
        DEFAULT_MODEL,
        OPENROUTER_DEFAULT_BASE_URL,
        OPENROUTER_MODEL_PREFIX,
        TYPESAFE_DEFAULT_BASE_URL,
    )

    model = DEFAULT_MODEL
    base_url = TYPESAFE_DEFAULT_BASE_URL
    if provider == "openrouter":
        model = f"{OPENROUTER_MODEL_PREFIX}{DEFAULT_MODEL}"
        base_url = OPENROUTER_DEFAULT_BASE_URL

    classifier = InstrumentedClassifier(
        price_table=price_table,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    retriever = (
        JevRelevanceRetrieverBuilder()
        .with_base_retriever(base)
        .with_classifier(classifier)
        .with_config(
            JevRelevanceConfig(
                scoring_mode=scoring_mode,
                threshold=threshold,
                top_k=candidate_k,
            )
        )
        .build()
    )
    return retriever, classifier


def _ranked_ids(documents: list[Document]) -> list[str]:
    return [str(document.metadata.get("id", "unknown")) for document in documents]


def _collect_jev_calls(
    ledger: Ledger, classifier: InstrumentedClassifier, before: int
) -> list[dict[str, object]]:
    """Fold this query's scoring calls into the ledger and return them as dicts."""
    fresh = classifier.records[before:]
    ledger.records.extend(fresh)
    return [record.as_dict() for record in fresh]


def _run_query(
    *,
    query: str,
    gold_ids: tuple[str, ...],
    base: BM25Retriever,
    jev_retriever: BaseRetriever,
    classifier: InstrumentedClassifier,
    judge,
    candidate_k: int,
    ledger: Ledger,
) -> dict[str, object]:
    gold_set = set(gold_ids)
    row: dict[str, object] = {
        "query": query,
        "gold_ids": gold_ids,
    }

    # --- BM25 baseline (frozen candidates) ---
    start = time.perf_counter()
    candidates = base.invoke(query)
    baseline_ms = (time.perf_counter() - start) * 1000.0
    baseline_ids = _ranked_ids(candidates)
    row["baseline"] = {
        "ranked_ids": baseline_ids,
        "latency_ms": round(baseline_ms, 2),
        "metrics": rank_metrics(baseline_ids, gold_set, k=candidate_k),
    }
    # The re-ranker ceiling: is the gold even in the candidate pool?
    row["recall_ceiling"] = rank_metrics(baseline_ids, gold_set, k=candidate_k)

    # --- Jev relevance-filtered ---
    before = len(classifier.records)
    start = time.perf_counter()
    jev_documents = jev_retriever.invoke(query)
    jev_ms = (time.perf_counter() - start) * 1000.0
    jev_ids = _ranked_ids(jev_documents)
    row["jev"] = {
        "ranked_ids": jev_ids,
        "latency_ms": round(jev_ms, 2),
        "metrics": rank_metrics(jev_ids, gold_set, k=candidate_k),
        "call_ledger": _collect_jev_calls(ledger, classifier, before),
    }

    # --- LLM judge baseline ---
    if judge is not None:
        judge_start = time.perf_counter()
        judge_kept, _, judge_response = judge_llm_baseline(judge, query, candidates)
        judge_ms = (time.perf_counter() - judge_start) * 1000.0
        usage = getattr(judge_response, "usage_metadata", None) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        judge_model = getattr(judge_response, "response_metadata", {}).get(
            "model", getattr(judge, "model_name", "openrouter/auto")
        )
        record_llm_judge_call(
            ledger,
            model=judge_model,
            latency_ms=judge_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        row["llm_judge"] = {
            "kept_ids": judge_kept,
            "latency_ms": round(judge_ms, 2),
            "metrics": rank_metrics(judge_kept, gold_set, k=candidate_k),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": judge_model,
        }
    else:
        row["llm_judge"] = None

    return row


def _write_json(document: dict[str, object], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / RESULT_JSON_PATTERN.format(dataset=_safe_name(document["run"]["dataset"]), timestamp=timestamp)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=True)
    print(f"\nWrote {document['run']['dataset']} results to {path}")
    return path


def _safe_name(dataset: str) -> str:
    return "".join(chunk if chunk.isalnum() or chunk in "-_" else "-" for chunk in dataset)


def _print_metrics(name: str, summary: dict[str, float]) -> None:
    print(f"\n--- {name} ---")
    metrics = sorted({key[:-5] for key in summary if key.endswith("_mean")})
    for metric in metrics:
        mean = summary.get(f"{metric}_mean", 0.0)
        p95 = summary.get(f"{metric}_p95", 0.0)
        print(f"  {metric:<16} mean={mean:.3f}  p95={p95:.3f}")


def _print_summary(
    document: dict[str, object], rows: list[dict[str, object]]
) -> None:
    running = {
        "baseline": [row["baseline"]["metrics"] for row in rows],  # type: ignore[index]
        "jev": [row["jev"]["metrics"] for row in rows],  # type: ignore[index]
        "llm_judge": [
            row["llm_judge"]["metrics"]
            for row in rows
            if row["llm_judge"] is not None
        ],
        "recall_ceiling": [row["recall_ceiling"] for row in rows],  # type: ignore[index]
    }
    for name, metrics_rows in running.items():
        _print_metrics(name, summarize(metrics_rows))

    print("\n--- Cost & latency (per pipe) ---")
    for pipe, key in (("BM25", "baseline"), ("Jev", "jev"), ("LLM judge", "llm_judge")):
        present = [row[key] for row in rows if key in row and row[key] is not None]  # type: ignore[index]
        if not present:
            continue
        latency = sorted(float(entry["latency_ms"]) for entry in present)  # type: ignore[index]
        p50 = latency[len(latency) // 2]
        p95 = latency[min(int(0.95 * len(latency)), len(latency) - 1)]
        print(f"  {pipe:<9} calls={len(present):>4}  p50={p50:7.1f}ms  p95={p95:7.1f}ms")

    total = document["aggregates"]["total_costs"]  # type: ignore[index]
    for model, costs in total.items():
        print(f"  priced {model:<28} cost=${costs['cost_usd']:.6f}  tokens={costs['input_tokens'] + costs['output_tokens']}")


def main(argv: list[str] | None = None) -> int:
    apply_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="demo",
        help="Dataset spec: demo | json:<path> | beir:<name>. Default: demo.")
    parser.add_argument("--provider", choices=["openrouter", "typesafe"],
        default="openrouter", help="Jev provider used for scoring.")
    parser.add_argument("--candidate-k", type=int, default=BENCHMARK_CANDIDATE_K,
        help="Number of BM25 candidates scored per query.")
    parser.add_argument("--threshold", type=float, default=BENCHMARK_THRESHOLD,
        help="Noul-mode relevance cutoff for the Jev filter.")
    parser.add_argument("--scoring-mode", choices=["per_chunk", "batched"],
        default="per_chunk", help="Jev scoring dispatch mode.")
    parser.add_argument("--max-queries", type=int, default=None,
        help="Cap on how many queries to evaluate (None = all).")
    parser.add_argument("--no-judge", action="store_true",
        help="Disable the LLM-judge baseline even if a key is present.")
    parser.add_argument("--out", type=Path, default=None,
        help="Output directory (default: benchmark/results).")
    args = parser.parse_args(argv)

    api_key = (
        openrouter_api_key() if args.provider == "openrouter" else typesafe_api_key()
    )
    if not api_key:
        required = OPENROUTER_API_KEY_ENV if args.provider == "openrouter" else "TYPESAFE_API_KEY"
        print(
            f"error: {required} is not set. Add it to {ENV_FILE} or export it in the shell.",
            file=sys.stderr,
        )
        return 2

    try:
        dataset = load_dataset(args.dataset)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    queries = dataset.queries
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    price_table = load_price_table()
    base = _build_base_retriever(dataset.corpus, args.candidate_k)
    jev_retriever, classifier = _build_jev_pipeline(
        base,
        provider=args.provider,
        api_key=api_key,
        threshold=args.threshold,
        candidate_k=args.candidate_k,
        scoring_mode=args.scoring_mode,
        price_table=price_table,
    )
    judge = None if args.no_judge else build_judge_model()
    ledger = Ledger(price_table=price_table)

    rows: list[dict[str, object]] = []
    for index, (query, gold_ids) in enumerate(queries, start=1):
        row = _run_query(
            query=query,
            gold_ids=gold_ids,
            base=base,
            jev_retriever=jev_retriever,
            classifier=classifier,
            judge=judge,
            candidate_k=args.candidate_k,
            ledger=ledger,
        )
        rows.append(row)
        print(f"  [{index}/{len(queries)}] done ({_safe_query(query)})", end="\r")
    print()

    price_snapshot = {
        model_id: {"input_usd_per_1m": rate[0], "output_usd_per_1m": rate[1]}
        for model_id, rate in price_table.rates_usd_per_1m.items()
    }
    total_by_model: dict[str, dict[str, object]] = {}
    for record in ledger.records:
        bucket = total_by_model.setdefault(
            record.model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        )
        bucket["calls"] += 1
        bucket["input_tokens"] += record.input_tokens
        bucket["output_tokens"] += record.output_tokens
        bucket["cost_usd"] = round(bucket["cost_usd"] + record.cost_usd, 8)

    document: dict[str, object] = {
        "run": {
            "tool": "jevreleval",
            "version": __version__,
            "dataset": args.dataset,
            "provider": args.provider,
            "threshold": args.threshold,
            "candidate_k": args.candidate_k,
            "scoring_mode": args.scoring_mode,
            "query_count": len(rows),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "price_source": price_table.source,
        },
        "pricing": price_snapshot,
        "aggregates": {
            "baseline": summarize([row["baseline"]["metrics"] for row in rows]),  # type: ignore[index]
            "jev": summarize([row["jev"]["metrics"] for row in rows]),  # type: ignore[index]
            "llm_judge": summarize(
                [row["llm_judge"]["metrics"] for row in rows if row["llm_judge"] is not None]
            ),
            "recall_ceiling": summarize(
                [row["recall_ceiling"] for row in rows]  # type: ignore[index]
            ),
            "total_costs": total_by_model,
            "ledger_summary": ledger.summed(),
        },
        "per_query": rows,
    }

    out_dir = args.out if args.out is not None else Path(__file__).resolve().parents[1] / RESULTS_DIR_NAME
    _write_json(document, out_dir)
    _print_summary(document, rows)
    return 0


def _safe_query(query: str) -> str:
    return query[:48].replace("\n", " ")


if __name__ == "__main__":
    sys.exit(main())