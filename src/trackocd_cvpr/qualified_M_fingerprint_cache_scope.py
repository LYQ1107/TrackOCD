"""Exact Raw support SHA overlay for frozen M as well as Mean/state-EMA.

NEW own-process composition; original qualified scope/source unchanged.
Shared Raw32MiB + complete byte-segment SHA32MiB, never a model-output memo.
Only omitted-argument sha256() support builders in weighted memory are deferred.
ALL initialized numeric hashes (even explicit b'') use original hashlib.
No global hashing, state refresh, revision, history or numerical kernel changes.
Actual ALL11 whole-policy shadow required before any full-development use.
"""
from contextlib import contextmanager
import hashlib
from .qualified_fingerprint_cache_scope import qualified_fingerprint_cache_scope, _DeferredSHA256

_OMITTED = object()


class _SupportOnlySHA256Proxy:
    def __init__(self, cache): self._cache = cache

    def sha256(self, data=_OMITTED, **kwargs):
        if data is _OMITTED and not kwargs:
            return _DeferredSHA256(self._cache)
        return hashlib.sha256(b'' if data is _OMITTED else data, **kwargs)


@contextmanager
def qualified_M_fingerprint_cache_scope():
    from . import versioned_reliability_memory as reliability
    if reliability.hashlib is not hashlib:
        raise ValueError('Original weighted-memory hashing alias, no nested/unknown overlay')
    with qualified_fingerprint_cache_scope() as (raw, fingerprints):
        reliability.hashlib = _SupportOnlySHA256Proxy(fingerprints)
        try:
            yield raw, fingerprints
        finally:
            reliability.hashlib = hashlib
