#!/usr/bin/env python3
"""Recover tiny historical split assets only when their exact SHA256 matches.

The complete role IDs were recorded in the original machine audit. The
historical writer's indent=1/no-final-newline recipe is still in
src/data/build_protocol.py. This is verified asset reconstruction, not a
new data split, a recovered old worktree, or reconstructed experiment results.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json, sha256_file

ROLE_FILES = {
    'known_ids': ('known_base', 'known_ids.json'),
    'novel_ids': ('genuine_novel', 'unknown_ids_val.json'),
    'distractor_ids': ('distractor', 'distractor_ids.json'),
}


def recover(roles: dict, destination: Path) -> dict:
    pending = []
    # Validate all three payloads and any existing targets before any write.
    for field, (original, filename) in ROLE_FILES.items():
        payload = json.dumps(roles[field], indent=1).encode('utf-8')
        expected = roles['original_files'][original]['sha256']
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'ORIGINAL_ROLE_HASH_MISMATCH: {filename}')
        target = destination / filename
        if target.exists() and (not target.is_file() or sha256_file(target) != expected):
            raise ValueError(f'REFUSING_TO_OVERWRITE_DIFFERENT_ASSET: {target}')
        pending.append((target, payload, expected, len(roles[field])))
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for target, payload, expected, count in pending:
        if not target.exists():
            temporary = target.with_name(f'.{target.name}.tmp.{os.getpid()}')
            with temporary.open('xb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        if sha256_file(target) != expected:
            raise RuntimeError(f'POST_WRITE_ROLE_HASH_MISMATCH: {target}')
        records[target.name] = {'path': str(target), 'sha256': expected,
                                'size_bytes': len(payload), 'count': count,
                                'matches_original_byte_hash': True}
    return {'schema_version': 'trackocd.core.role_recovery.v1', 'status': 'PASS',
            'serialization': {'indent': 1, 'final_newline': False},
            'role_ids_changed': False, 'original_worktree_recovered': False,
            'source_audit': roles['source'],
            'original_writer': 'src/data/build_protocol.py:225-228', 'files': records}


def main() -> int:
    roles = json.loads((ROOT / 'configs/trackocd_core/roles.json').read_text())
    result = recover(roles, ROOT / 'outputs/trackocd_core/recovered_splits')
    result['writer_sha256'] = sha256_file(ROOT / 'src/data/build_protocol.py')
    atomic_json(ROOT / 'outputs/trackocd_core/audit/role_recovery.json', result)
    print(json.dumps({'status': result['status'], 'files': result['files']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
