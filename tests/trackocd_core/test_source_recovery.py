import json
from pathlib import Path

import pytest

from scripts.trackocd_core.verify_recovered_source import blob_sha, verify

ROOT = Path(__file__).resolve().parents[2]


def _delivery():
    return json.loads((ROOT / "outputs/trackocd_core/audit/nas_source_delivery.json").read_text())


def test_exact_nas_git_and_a100_source_identity():
    result = verify(ROOT, _delivery())
    assert result["file_count"] == 81
    assert result["total_bytes"] == 642256
    assert not result["runtime_assets_recovered"]


def test_corrupt_source_manifest_is_not_identity_proof():
    delivery = _delivery()
    delivery["files"][0]["git_blob_sha"] = blob_sha(b"different source")
    with pytest.raises(ValueError, match="NAS/Git tree mismatch"):
        verify(ROOT, delivery)


def test_unregistered_asset_path_is_rejected_before_read():
    delivery = _delivery()
    delivery["files"][0]["path"] = "data/annotations/test.json"
    with pytest.raises(ValueError, match="Unexpected source path"):
        verify(ROOT, delivery)
