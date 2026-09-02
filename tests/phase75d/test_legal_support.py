from pathlib import Path

from src.iclr27_phase75d.legal_support import load_legal_episodes


def test_legal_support_uses_manifest_candidates_only():
    root = Path("outputs/iclr27_phase30/manifests")
    episodes, unevaluable, summary = load_legal_episodes(root, 3, None)
    assert episodes
    assert summary["candidate_construction"].startswith("manifest-explicit")
    assert all(ep.positive_support_keys and ep.negative_support_keys for ep in episodes)
    assert all(set(ep.positive_support_keys).isdisjoint(ep.negative_support_keys) for ep in episodes)
    assert isinstance(unevaluable, list)
