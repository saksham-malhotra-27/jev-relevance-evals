"""Streamlit dashboard for jevreleval run results.

Reads the machine-readable JSON emitted by ``benchmark/jevreleval`` and shows
summary metrics, cost/latency, per-query drill-downs, the price snapshot, and
side-by-side run comparisons. Standalone: only depends on ``streamlit``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st


def default_results_dir() -> Path:
    """Where dashboard data lives.

    Resolution order:
      1. ``JEV_RESULTS_DIR`` env var (Docker and power users).
      2. ``<repo>/benchmark/results`` when present (fresh local benchmark runs).
      3. ``ui/sample_data`` (committed sample runs, so a fresh clone just works).
    """
    override = os.environ.get("JEV_RESULTS_DIR")
    if override:
        return Path(override)
    repo_results = Path(__file__).resolve().parents[1] / "benchmark" / "results"
    if repo_results.is_dir():
        return repo_results
    return Path(__file__).resolve().parent / "sample_data"

#: Metric columns rendered as a matrix of pipes x metrics.
METRICS = ["recall_at_k", "precision_at_k", "hit_rate_at_k", "mrr", "ndcg_at_k"]

#: Pipes that appear in per-query rows and aggregates.
PIPES = ["baseline", "jev", "llm_judge", "recall_ceiling"]

#: Human-readable pipe labels.
PIPE_LABELS = {
    "baseline": "BM25",
    "jev": "Jev filter",
    "llm_judge": "LLM judge",
    "recall_ceiling": "Recall ceiling",
}


def list_run_files(results_dir: Path) -> list[Path]:
    """All ``*.json`` result files under ``results_dir``, newest first."""
    if not results_dir.is_dir():
        return []
    return sorted(results_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_run(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def metric_matrix(document: dict, pipes: list[str]) -> list[dict]:
    """Rows of (pipe, metric -> mean/p50/p95) from a run's aggregates."""
    aggregates = document.get("aggregates", {})
    rows: list[dict] = []
    for pipe in pipes:
        summary = aggregates.get(pipe, {})
        if not summary or pipe not in PIPES:
            continue
        row: dict = {"pipe": PIPE_LABELS.get(pipe, pipe)}
        for metric in METRICS:
            row[f"{metric} mean"] = summary.get(f"{metric}_mean", None)
            row[f"{metric} p50"] = summary.get(f"{metric}_p50", None)
            row[f"{metric} p95"] = summary.get(f"{metric}_p95", None)
        rows.append(row)
    return rows


def cost_table(document: dict) -> list[dict]:
    """Per-model rows of calls/tokens/cost from a run's ledger."""
    totals = document.get("aggregates", {}).get("total_costs", {})
    return [
        {
            "model": model,
            "calls": bucket.get("calls", 0),
            "input_tokens": bucket.get("input_tokens", 0),
            "output_tokens": bucket.get("output_tokens", 0),
            "cost_usd": bucket.get("cost_usd", 0.0),
        }
        for model, bucket in sorted(totals.items(), key=lambda item: item[0])
    ]


def latency_by_pipe(document: dict) -> list[dict]:
    """p50/p95 wall latency (ms) per pipe, from the per-query rows."""
    rows = document.get("per_query", [])
    buckets: dict[str, list[float]] = {}
    for row in rows:
        for pipe in ("baseline", "jev", "llm_judge"):
            entry = row.get(pipe)
            if entry and entry.get("latency_ms") is not None:
                buckets.setdefault(pipe, []).append(float(entry["latency_ms"]))
    result = []
    for pipe in ("baseline", "jev", "llm_judge"):
        values = sorted(buckets.get(pipe, []))
        if not values:
            continue
        result.append(
            {
                "pipe": PIPE_LABELS.get(pipe, pipe),
                "calls": len(values),
                "latency_ms_p50": values[len(values) // 2],
                "latency_ms_p95": _p95(values),
            }
        )
    return result


def per_query_table(document: dict) -> list[dict]:
    """One row per query with per-pipe metrics and latency."""
    out: list[dict] = []
    for index, row in enumerate(document.get("per_query", [])):
        entry: dict = {"#": index + 1, "query": row.get("query", ""), "gold_count": len(row.get("gold_ids", []) or [])}
        for pipe in ("baseline", "jev", "llm_judge"):
            data = row.get(pipe)
            if not data:
                continue
            metrics = data.get("metrics", {}) or {}
            entry[f"{pipe} r"] = metrics.get("recall_at_k")
            entry[f"{pipe} p"] = metrics.get("precision_at_k")
            entry[f"{pipe} mrr"] = metrics.get("mrr")
            entry[f"{pipe} ndcg"] = metrics.get("ndcg_at_k")
            entry[f"{pipe} ms"] = data.get("latency_ms")
        out.append(entry)
    return out


#: Effective USD-per-1M rates the benchmark falls back to for models the
#: OpenRouter catalog does not publish (e.g. routing aliases like
#: ``openrouter/auto``), matching jevreleval's rate constants.
_FALLBACK_INPUT_USD_PER_1M = 0.15
_FALLBACK_OUTPUT_USD_PER_1M = 0.60


def pricing_table(document: dict) -> list[dict]:
    """Rows of (model, input/output USD per 1M tokens) from a run."""
    pricing = document.get("pricing", {}) or {}
    rows = [
        {
            "model": model,
            "input_usd_per_1m": _price(rate.get("input_usd_per_1m")),
            "output_usd_per_1m": _price(rate.get("output_usd_per_1m")),
            "source": rate.get("source", "catalog"),
        }
        for model, rate in sorted(pricing.items())
    ]
    used = document.get("aggregates", {}).get("total_costs", {}) or {}
    for model in sorted(set(used) - set(pricing)):
        rows.append(
            {
                "model": model,
                "input_usd_per_1m": _price(_FALLBACK_INPUT_USD_PER_1M),
                "output_usd_per_1m": _price(_FALLBACK_OUTPUT_USD_PER_1M),
                "source": "fallback (unlisted)",
            }
        )
    return rows


def _price(value: float | None) -> str:
    """Format a USD-per-1M rate for display (``$10.0000`` or ``$0.000250``)."""
    if value is None:
        return ""
    return f"${value:,.4f}" if value >= 0.01 else f"${value:.6f}"


def _p95(sorted_values: list[float]) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(0.95 * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[max(0, index)]


def _selected_query_expansion(document: dict, key: str) -> None:
    rows = document.get("per_query", [])
    if not rows:
        st.info("No per-query rows in this run.")
        return
    labels = [f"{index + 1}. {row.get('query', '')[:70]}" for index, row in enumerate(rows)]
    choice = st.selectbox("Query", options=range(len(rows)), format_func=lambda i: labels[i], key=key)
    row = rows[choice]
    st.write(f"**Gold ids ({len(row.get('gold_ids', []) or [])}):** {', '.join(row.get('gold_ids', []) or [])}")
    for pipe, title in (("baseline", "BM25 ranked ids"), ("jev", "Jev filtered ids"), ("llm_judge", "LLM judge kept ids")):
        data = row.get(pipe)
        if not data:
            continue
        ids = data.get("ranked_ids", data.get("kept_ids", []))
        st.write(f"**{title}:** {', '.join(ids[:15])}{' …' if len(ids) > 15 else ''}")
    ledger = (row.get("jev") or {}).get("call_ledger", [])
    if ledger:
        st.write("**Jev call ledger**")
        st.dataframe(
            [
                {
                    "model": call.get("model"),
                    "question_key": call.get("question_key"),
                    "latency_ms": call.get("latency_ms"),
                    "input_tokens": call.get("input_tokens"),
                    "output_tokens": call.get("output_tokens"),
                    "cost_usd": call.get("cost_usd"),
                }
                for call in ledger
            ],
            hide_index=True,
        )


def compare_matrix(runs: list[Path]) -> list[dict]:
    """Side-by-side mean metric per pipe across the selected runs."""
    out: list[dict] = []
    for document, path in _load_runs(runs):
        for row in metric_matrix(document, PIPES):
            out.append(
                {
                    "file": path.name,
                    **row,
                }
            )
    return out


def _load_runs(paths: list[Path]) -> list[tuple[dict, Path]]:
    documents = []
    for path in paths:
        try:
            documents.append((load_run(path), path))
        except (OSError, json.JSONDecodeError) as error:
            st.warning(f"Skipping {path.name}: {error}")
    return documents


def render(document: dict, path: Path) -> None:
    run = document.get("run", {})
    st.header("jev-relevance benchmark results")
    meta = [
        ("dataset", run.get("dataset")),
        ("provider", run.get("provider")),
        ("threshold", run.get("threshold")),
        ("candidate_k", run.get("candidate_k")),
        ("scoring_mode", run.get("scoring_mode")),
        ("query_count", run.get("query_count")),
        ("generated_at", run.get("generated_at")),
        ("price_source", run.get("price_source")),
    ]
    st.caption(" · ".join(f"{key}={value}" for key, value in meta if value is not None))

    tab_summary, tab_costs, tab_queries, tab_pricing, tab_compare = st.tabs(
        ["Summary", "Costs", "Per-query", "Pricing", "Compare runs"]
    )

    with tab_summary:
        st.subheader("Metric cross-tab (mean / p50 / p95 per pipe)")
        st.dataframe(metric_matrix(document, PIPES), hide_index=True)

    with tab_costs:
        st.subheader("Cost by model")
        st.dataframe(cost_table(document), hide_index=True)
        summary = document.get("aggregates", {}).get("ledger_summary", {})
        if summary:
            st.subheader("Aggregate ledger")
            st.dataframe([{"calls": summary.get("calls"), "input_tokens": summary.get("input_tokens"),
                           "output_tokens": summary.get("output_tokens"), "cost_usd": summary.get("cost_usd"),
                           "latency_ms_p50": summary.get("latency_ms_p50"), "latency_ms_p95": summary.get("latency_ms_p95")}],
                          hide_index=True)
        st.subheader("Wall latency per pipe (p50/p95, ms)")
        st.dataframe(latency_by_pipe(document), hide_index=True)

    with tab_queries:
        st.subheader("Per-query metrics")
        st.dataframe(per_query_table(document), hide_index=True)
        st.divider()
        _selected_query_expansion(document, key=self_key(path))

    with tab_pricing:
        st.subheader("Price snapshot (USD per 1M tokens)")
        rows = pricing_table(document)
        if not rows:
            st.info("No pricing snapshot in this run.")
        st.caption(
            "Rates are USD per 1M input/output tokens. 'catalog' rows come from "
            "the run-time OpenRouter snapshot; 'fallback' rows are the effective "
            "rates used for aliases the catalog does not publish "
            "(e.g. openrouter/auto, ~typesafe/jev-latest)."
        )
        st.dataframe(rows, hide_index=True)

    with tab_compare:
        st.subheader("Compare runs (per-pipe means)")
        runs = list_run_files(default_results_dir())
        if not runs:
            st.info(f"No result files found in {default_results_dir()}.")
            return
        labels = [f"{r.name}" for r in runs]
        selected = st.multiselect(
            "Runs to compare", options=list(range(len(runs))), format_func=lambda i: labels[i],
            default=[next((i for i, r in enumerate(runs) if r == path), 0)],
            key=f"compare-{path.stem}",
        )
        if not selected:
            return
        st.dataframe(compare_matrix([runs[i] for i in selected]), hide_index=True)


def self_key(path: Path) -> str:
    """A stable widget key derived from the file name to avoid clashes."""
    return f"run-{path.stem}"


def main() -> None:
    st.set_page_config(page_title="jev-relevance results", layout="wide")
    st.sidebar.title("jev-relevance")
    results_dir = default_results_dir()
    runs = list_run_files(results_dir)
    if not runs:
        st.error(f"No result JSON files found in {results_dir}. Run the benchmark first.")
        st.stop()
    labels = [f"{r.name} ({_run_label(r)})" for r in runs]
    selection = st.sidebar.selectbox("Result file", options=range(len(runs)), format_func=lambda i: labels[i])
    path = runs[selection]
    try:
        document = load_run(path)
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read {path.name}: {error}")
        st.stop()
    st.sidebar.caption("Refresh the file list to pick up new runs.")
    if st.sidebar.button("Refresh files"):
        st.rerun()
    render(document, path)


def _run_label(path: Path) -> str:
    try:
        document = load_run(path)
        return str(document.get("run", {}).get("dataset", "?"))
    except Exception:
        return "?"


if __name__ == "__main__":
    main()