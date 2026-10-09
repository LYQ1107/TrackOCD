"""Installer guards, with no network, environment installation or inference."""
from pathlib import Path
import os

import pytest

from scripts.trackocd_core.install_masa_runtime import allocated_bytes, locked_requirement, matches_install_provenance


def test_budget_deduplicates_same_filesystem_hardlinks(tmp_path):
    first, second = tmp_path / "download", tmp_path / "environment"
    first.mkdir(); second.mkdir()
    payload = first / "lib.so"
    payload.write_bytes(b"x" * 8192)
    os.link(payload, second / "lib.so")
    expected = first.stat().st_blocks * 512 + second.stat().st_blocks * 512 + payload.stat().st_blocks * 512
    assert allocated_bytes([first, second]) == expected


def test_budget_never_follows_an_external_directory_symlink(tmp_path):
    owned, foreign = tmp_path / "owned", tmp_path / "foreign"
    owned.mkdir(); foreign.mkdir()
    (foreign / "large").write_bytes(b"x" * 8192)
    link = owned / "external"
    link.symlink_to(foreign, target_is_directory=True)
    assert allocated_bytes([owned]) == (owned.stat().st_blocks + link.lstat().st_blocks) * 512


def test_requirement_uses_explicit_archive_hash_without_new_resolution():
    record = {"name": "demo", "url": "https://files.pythonhosted.org/demo-1.0-py3-none-any.whl", "expected_sha256_from_lock": "a" * 64}
    assert locked_requirement(record).endswith(" --hash=sha256:" + "a" * 64 + "\n")


@pytest.mark.parametrize("url,digest", [
    ("https://example.org/demo.whl", "a" * 64),
    ("http://files.pythonhosted.org/demo.whl", "a" * 64),
    ("https://files.pythonhosted.org/demo.tar.gz", "a" * 64),
    ("https://files.pythonhosted.org/demo.whl", "not-a-hash"),
])
def test_unknown_binary_source_or_hash_is_rejected(url, digest):
    with pytest.raises(ValueError):
        locked_requirement({"name": "demo", "url": url, "expected_sha256_from_lock": digest})


def test_empty_uv_archive_metadata_requires_saved_successful_hash_required_operation():
    record = {"version": "1.0", "url": "https://files.pythonhosted.org/demo.whl", "expected_sha256_from_lock": "a" * 64}
    actual = {"version": "1.0", "direct_url": {"url": record["url"], "archive_info": {}}}
    operation = {"returncode": 0, "url": record["url"], "sha256": "a" * 64, "require_hashes": True}
    assert not matches_install_provenance(actual, record, None)
    assert matches_install_provenance(actual, record, operation)
    for key, bad in [("returncode", 1), ("url", "elsewhere"), ("sha256", "b" * 64), ("require_hashes", False)]:
        assert not matches_install_provenance(actual, record, {**operation, key: bad})
    assert not matches_install_provenance({**actual, "version": "2.0"}, record, operation)


def test_conflicting_saved_archive_hash_is_never_overridden_by_operation():
    record = {"version": "1.0", "url": "https://files.pythonhosted.org/demo.whl", "expected_sha256_from_lock": "a" * 64}
    actual = {"version": "1.0", "direct_url": {"url": record["url"], "archive_info": {"hashes": {"sha256": "b" * 64}}}}
    operation = {"returncode": 0, "url": record["url"], "sha256": "a" * 64, "require_hashes": True}
    assert not matches_install_provenance(actual, record, operation)
