import json

from scripts.trackocd_v2.run_formal_baselines_after_cache import _reusable_audit


def _write_bound_cache(tmp_path, *, receipt_source="source-sha"):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    manifest = cache_root / "cache_manifest.json"
    manifest.write_text(
        json.dumps({
            "status": "COMPLETE",
            "generated_utc": "2026-10-24T20:24:22.830278+00:00",
            "cache_root": str(cache_root),
            "total_tracks": 2,
            "shard_tracks": 1,
            "completed_shards": [{"track_start": 0}, {"track_start": 1}],
            "input": {"sha256": "source-sha"},
        }),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({
            "status": "PASS_COMPLETE",
            "formal_cache_complete": True,
            "generated_utc": "2026-10-24T20:40:06.595230+00:00",
            "manifest": str(manifest.resolve()),
            "cache_root": str(cache_root.resolve()),
            "frontend": "COVTrack-native",
            "split": "val",
            "input_stream_sha256": receipt_source,
            "completed_shards": 2,
            "required_shards": 2,
            "error_count": 0,
            "invalid_box_count": 0,
            "test_content_opened": False,
        }),
        encoding="utf-8",
    )
    return manifest, audit


def test_reusable_audit_requires_exact_current_manifest_binding(tmp_path, monkeypatch):
    import scripts.trackocd_v2.run_formal_baselines_after_cache as gate

    monkeypatch.setattr(gate, "SOURCE_SHA256", "source-sha")
    manifest, audit = _write_bound_cache(tmp_path)
    assert _reusable_audit(manifest, audit)["status"] == "PASS_COMPLETE"

    audit_bad = tmp_path / "audit_bad.json"
    audit_bad.write_text(audit.read_text(encoding="utf-8").replace("source-sha", "other-sha"), encoding="utf-8")
    assert _reusable_audit(manifest, audit_bad) is None
