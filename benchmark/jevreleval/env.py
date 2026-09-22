"""Env loading for the benchmark.

The benchmark is the *caller* in the credential contract: it owns secret
management and passes api_key/base_url into the library as parameters. It reads
a ``.env`` file living inside this package (``benchmark/jevreleval/.env``) via
``python-dotenv``, then exposes the credential values the CLI builds provider
configs from.
"""

from __future__ import annotations

import os
from pathlib import Path

from jevreleval.constants import OPENROUTER_API_KEY_ENV, TYPESAFE_API_KEY_ENV

#: The .env file expected next to this module.
ENV_FILE = Path(__file__).with_name(".env")


def apply_env_file() -> None:
    """Merge the package-local ``.env`` into ``os.environ``.

    Existing environment variables win; the file only fills gaps. Fails safely
    if ``python-dotenv`` is not installed — callers fall back to empty keys and
    the CLI reports a clear error instead.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ImportError:
        return
    if ENV_FILE.is_file():
        load_dotenv(ENV_FILE, override=False)


def typesafe_api_key() -> str:
    """TypeSafe credential from the environment (fed by the package ``.env``)."""
    return os.environ.get(TYPESAFE_API_KEY_ENV, "")


def openrouter_api_key() -> str:
    """OpenRouter credential from the environment (fed by the package ``.env``)."""
    return os.environ.get(OPENROUTER_API_KEY_ENV, "")


__all__ = ["ENV_FILE", "apply_env_file", "typesafe_api_key", "openrouter_api_key"]