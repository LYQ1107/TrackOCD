"""Phase76AR: corrected, selective raw-anchored relation route.

The namespace is intentionally independent from Phase76A.  Frozen Phase30
manifests and Phase75D feature tables are read-only inputs; no controller or
semantic memory is imported here.
"""

__all__ = ["data", "pair_cache", "relation_model", "runtime", "losses", "evaluator"]
