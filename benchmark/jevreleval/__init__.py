"""jevreleval: a standalone benchmark for jev-relevance.

This package is intentionally separate from the ``jev_relevance`` library. It
owns its own data loading, baseline judge, and metrics so that none of that
couples to the library's public API. Run it with::

    python -m jevreleval --dataset json:path/to/queries.json --out results/
"""