"""A100 migration harness; leave all 81 recovered NAS files byte-identical.

Legacy tests assume the old project root and output mount. Redirect their
synthetic output layout to pytest's temporary directory, not /data2. The
process-ownership test simulates the old root named in its command fixture;
this does not probe, stop, or claim ownership of a real process.
"""
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_legacy_output_layout(monkeypatch, tmp_path, request):
    from src.trackocd_v2 import io
    from scripts.trackocd_v2 import build_sharded_feature_cache as builder

    output = tmp_path / "legacy_outputs"
    monkeypatch.setattr(io, "OUTPUT_TARGET", output)
    monkeypatch.setattr(io, "OUTPUT_LINK", tmp_path / "project" / "outputs" / "trackocd_v2")
    monkeypatch.setattr(builder, "OUTPUT_TARGET", output)
    monkeypatch.setattr(builder, "FORMAL_ROOT", output / "features" / "formal")
    monkeypatch.setattr(builder, "FINAL_FREEZE", output / "audit" / "FINAL_FREEZE.json")
    if request.node.name == "test_task_owned_only_matches_trackocd_paths":
        from scripts.trackocd_v2 import frontend_execution_preflight as preflight

        monkeypatch.setattr(
            preflight, "ROOT", Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
        )
