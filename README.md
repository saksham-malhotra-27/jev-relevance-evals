# jev-relevance-evals

Benchmark harness and results dashboard for
[jev-relevance](https://github.com/saksham-malhotra-27/jev-relevance), the
drop-in LangChain relevance filter powered by
[TypeSafe Jev](https://typesafe.ai).

Two pieces live here:

- `benchmark/` — CLI that scores BM25 candidates three ways on the same frozen
  retrieval pool: plain keyword retrieval, the Jev relevance filter (the
  drop-in retriever), and a cheap OpenRouter LLM judge. Emits full IR metrics +
  a per-API-call cost ledger to JSON.
- `ui/` — a Streamlit dashboard that renders those JSON runs (summary metrics,
  cost/latency, per-query drill-downs, price snapshot, run comparisons). Ship
  it standalone, locally or in Docker.

Nothing in this repo is published to PyPI; the library ships on its own.

## Repo layout

```
benchmark/jevreleval/     benchmark CLI (python -m jevreleval)
ui/                       Streamlit dashboard
  sample_data/            committed example runs (renders out of the box)
.github/workflows/        docker-staging.yml -> ghcr.io staging image
```

## Benchmark

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r benchmark/requirements.txt
```

The benchmark is the *credential owner*: copy `benchmark/jevreleval/.env.example`
to `benchmark/jevreleval/.env` and fill in your keys. It loads them itself and
passes them to the library **as parameters** (the library never reads env vars).

```bash
cd benchmark/jevreleval
copy .env.example .env            # fill in OPENROUTER_API_KEY (Jev tube + LLM judge)
cd ..
python -m jevreleval --dataset demo              # validate the setup
python -m jevreleval --dataset beir:nfcorpus     # full BEIR dataset
```

Datasets: `demo` | `json:<path>` | `beir:<name>`. The `beir:` source downloads
the canonical UKP archive to `benchmark/jevreleval/cache/` (stdlib-only, no
pandas). BEIR archives are ~50–400 MB; the nfcorpus run evaluates 323 queries
and costs roughly $1 across ~6.5k Jev scoring calls and 323 judge calls, with
every call priced against the live OpenRouter catalog.

Each run writes `benchmark/results/<dataset>-<timestamp>.json` (override with
`--out`) containing:

- per-pipe metrics (Recall@k, P@k, Hit-rate@k, Mean Reciprocal Rank, nDCG@k)
  for `baseline`, `jev`, and `llm_judge`, plus the `recall_ceiling` (whether the
  gold is even in the BM25 candidate pool);
- a per-call ledger: model, latency, input/output tokens, and cost in USD for
  every Jev and judge API call;
- aggregate means/p50/p95 and totals, so results can be re-processed in any form.

### Example nfcorpus run (threshold 0.5, k=20)

| pipe        | Recall@20 | P@20 | MRR   | nDCG@20 | Hit@20 | cost                |
|-------------|-----------|------|-------|---------|--------|---------------------|
| BM25        | 0.171     | 0.152| 0.510 | 0.268   | 0.718  | $0 (0 calls)        |
| Jev filter  | 0.118     | 0.397| 0.492 | 0.217   | 0.545  | $0.73 (6,460 calls) |
| LLM judge   | 0.129     | 0.464| 0.534 | 0.235   | 0.601  | $0.71 (323 calls)   |

^Costs are near-parity by coincidence: Jev does ~20× the judge's call volume (one per candidate vs one per query) but is ~19× cheaper per call.

Jev trades recall for a doubling of precision at the default 0.5 threshold;
because the candidate pool is the recall ceiling, the gap mostly falls out of
BM25 retrieval, which itself misses 28% of queries entirely (hit-rate 0.718).

## Dashboard

```bash
pip install -r ui/requirements.txt   # just streamlit
streamlit run ui/app.py              # opens http://localhost:8501
```

On a fresh clone it renders the committed `ui/sample_data/` runs. If you've run
the benchmark locally, it automatically prefers `benchmark/results/`. Override
explicitly with the `JEV_RESULTS_DIR` env var.

### Docker (staging flow)

The dashboard is containerized for reproducible staging:

```bash
docker compose -f ui/docker-compose.yml up   # build + run on :8501
```

On every merge to `main`, the GitHub Actions workflow pushes the same image to
`ghcr.io/saksham-malhotra-27/jev-relevance-evals:staging`, so the staging
artifact is a stable URL you can pin in a compose file or deploy to any host:

```bash
docker run -p 8501:8501 ghcr.io/saksham-malhotra-27/jev-relevance-evals:staging
```

## License

MIT — see `LICENSE` in the main
[jev-relevance](https://github.com/saksham-malhotra-27/jev-relevance) repo.