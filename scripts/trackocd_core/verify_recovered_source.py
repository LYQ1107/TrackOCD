#!/usr/bin/env python3
"""Compare source-only NAS evidence, the fetched Git tree and A100 bytes.

Reads only the manifest's named source/test/config files and Git objects. No
legacy runner, data annotation, model or cache payload is opened.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json

PREFIXES = ("src/trackocd_v2/", "scripts/trackocd_v2/", "tests/trackocd_v2/")


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verify(root: Path, delivery: dict) -> dict:
    commit = delivery["recovery_commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Invalid recovery commit")
    rows = subprocess.check_output(
        ["git", "ls-tree", "-r", commit], cwd=root, text=True
    )
    tree = {}
    for line in rows.splitlines():
        meta, name = line.split("\t", 1)
        mode, kind, sha = meta.split()
        tree[name] = (mode, kind, sha)
    files = []
    seen = set()
    for entry in delivery["files"]:
        name = entry["path"]
        if name in seen or ".." in Path(name).parts or not (
            name.startswith(PREFIXES) and name.endswith(".py")
            or name == "configs/trackocd_v2/protocol.json"
        ):
            raise ValueError("Unexpected source path: " + name)
        seen.add(name)
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Source missing or symlink: " + name)
        data = path.read_bytes()
        git_entry = tree.get(name)
        if git_entry != ("100644", "blob", entry["git_blob_sha"]):
            raise ValueError("NAS/Git tree mismatch: " + name)
        if len(data) != entry["bytes"] or blob_sha(data) != entry["git_blob_sha"]:
            raise ValueError("A100/NAS source mismatch: " + name)
        files.append(dict(entry, sha256=hashlib.sha256(data).hexdigest()))
    if len(files) != delivery["file_count"] or sum(f["bytes"] for f in files) != delivery["total_bytes"]:
        raise ValueError("Source inventory totals mismatch")
    changed = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit],
        cwd=root, text=True,
    ).splitlines()
    if set(changed) != seen:
        raise ValueError("Recovery commit contains unregistered changes")
    return {
        "schema_version": "trackocd.core.recovered-source-verification.v1",
        "status": "PASS",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recovery_branch": delivery["recovery_branch"],
        "recovery_commit": commit,
        "nas_worktree_head": delivery["source_worktree_head"],
        "nas_evidence_turn": delivery["evidence_turn_id"],
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
        "nas_git_a100_blob_identity_verified": True,
        "full_historical_git_history_recovered": False,
        "runtime_assets_recovered": False,
        "legacy_source_modified_for_migration": False,
        "training_started": False,
        "test_data_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery", type=Path, default=ROOT / "outputs/trackocd_core/audit/nas_source_delivery.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/trackocd_core/audit/source_recovery.json")
    args = parser.parse_args()
    receipt = verify(ROOT, json.loads(args.delivery.read_text()))
    atomic_json(args.output, receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "recovery_commit", "file_count", "total_bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
