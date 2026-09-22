"""Data loading for the benchmark: demo corpus, JSON files, or BEIR datasets."""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from jevreleval.constants import DEMO_CORPUS, DEMO_QA

#: Sentinel used when no dataset override is passed on the CLI.
DEMO = "demo"

#: Canonical BEIR dataset archives (UKP). Larger than HF but plain zip/jsonl,
#: so loading needs only the standard library — no pandas/DLL risk.
BEIR_ARCHIVE_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)

#: BEIR qrels split evaluated by default (the official BEIR test split).
BEIR_QRELS_SPLIT = "test"

#: Local cache directory for downloaded BEIR archives (gitignored).
CACHE_DIR = Path(__file__).resolve().parent / "cache"


@dataclass(frozen=True)
class Dataset:
    """A benchmark dataset: passages plus (query, gold passage ids) pairs."""

    corpus: dict[str, str]
    queries: tuple[tuple[str, tuple[str, ...]], ...] = field(default_factory=tuple)


def load_dataset(spec: str) -> Dataset:
    """Load a dataset from ``spec``.

    Supported specs:
      * ``demo``              — built-in tiny corpus (default).
      * ``json:<path>``       — {"corpus": {id: text}, "queries": [[query, [id,...]]]}.
      * ``beir:<name>``       — BEIR dataset via HuggingFace ``datasets``.

    Raises:
        ValueError: When ``spec`` is malformed or the source cannot be loaded.
    """
    if spec == DEMO:
        return Dataset(corpus=dict(DEMO_CORPUS), queries=DEMO_QA)
    if spec.startswith("json:"):
        return _load_json_dataset(spec[len("json:") :])
    if spec.startswith("beir:"):
        return _load_beir_dataset(spec[len("beir:") :])
    raise ValueError(
        f"Unknown dataset spec {spec!r}. Expected 'demo', 'json:<path>', or 'beir:<name>'."
    )


def _load_json_dataset(path: str) -> Dataset:
    if not os.path.isfile(path):
        raise ValueError(f"json: dataset file does not exist: {path!r}.")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    corpus = payload.get("corpus")
    queries = payload.get("queries")
    if not isinstance(corpus, dict) or not isinstance(queries, list):
        raise ValueError(
            "json: dataset must contain an object 'corpus' (id -> text) and a "
            "list 'queries' of [query, [gold ids]]."
        )
    normalized_queries = tuple(
        (entry[0], tuple(str(gold_id) for gold_id in entry[1]))
        for entry in queries
    )
    return Dataset(corpus=corpus, queries=normalized_queries)


def _load_beir_dataset(name: str) -> Dataset:
    """Download the canonical BEIR archive for ``name`` and read its jsonl/tsv.

    The archives are plain zip files (``corpus.jsonl``, ``queries.jsonl``,
    ``qrels/{split}.tsv``), so this path deliberately avoids the ``datasets``
    and ``pandas`` stack.
    """
    archive_path = _beir_archive(name)
    with zipfile.ZipFile(archive_path) as archive:
        prefix = f"{name}/"
        corpus = {
            record["_id"]: record["text"]
            for record in _read_jsonl(archive, f"{prefix}corpus.jsonl")
        }
        queries_by_id = {
            record["_id"]: record["text"]
            for record in _read_jsonl(archive, f"{prefix}queries.jsonl")
        }
        qrels = _read_qrels(
            archive, f"{prefix}qrels/{BEIR_QRELS_SPLIT}.tsv", queries_by_id
        )
    return Dataset(corpus=corpus, queries=qrels)


def _beir_archive(name: str) -> Path:
    """Return a cached archive path, downloading the UKP zip when missing."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE_DIR / f"{name}.zip"
    if archive_path.is_file():
        return archive_path
    url = BEIR_ARCHIVE_URL.format(name=name)
    try:
        import requests

        with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_S) as response:
            response.raise_for_status()
            with open(archive_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    if chunk:
                        handle.write(chunk)
    except Exception as error:
        raise ValueError(f"Failed to download BEIR archive for {name!r} from {url!r}: {error}") from error
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise ValueError(f"BEIR archive download produced no data for {name!r}.")
    return archive_path


#: Seconds to wait for the BEIR archive download to respond/complete.
_DOWNLOAD_TIMEOUT_S = 300.0


def _read_jsonl(archive: zipfile.ZipFile, member: str) -> list[dict]:
    """Read a jsonl member of the archive as a list of records."""
    try:
        content = archive.read(member)
    except KeyError as error:
        raise ValueError(f"BEIR archive is missing {member!r}.") from error
    return [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]


def _read_qrels(
    archive: zipfile.ZipFile,
    member: str,
    queries_by_id: dict[str, str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Read a qrels TSV (query-id, corpus-id, score) for the given split.

    Relevance scores above zero count as relevant; queries with no labeled
    passages are skipped.
    """
    try:
        content = archive.read(member)
    except KeyError as error:
        raise ValueError(f"BEIR archive is missing {member!r}.") from error
    pairs: dict[str, list[str]] = {}
    for line in content.decode("utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        query_id, corpus_id, score = parts[0], parts[1], parts[2]
        if query_id == "query-id":  # header row
            continue
        try:
            if float(score) <= 0.0:
                continue
        except ValueError:
            continue
        pairs.setdefault(query_id, []).append(corpus_id)
    queries = tuple(
        (queries_by_id[query_id], tuple(gold_ids))
        for query_id, gold_ids in pairs.items()
        if query_id in queries_by_id
    )
    return queries


__all__ = ["Dataset", "load_dataset", "DEMO"]