#!/usr/bin/env python3
"""Prepare an explicit, no-copy local mapping for verified Train/Val assets.

The new server already contains the TAO-Amodal Train/Val tree and a byte
recovered role split.  This tool can expose those assets at the exact v2 paths
through symlinks, avoiding a second 358GB data copy.  It is a dry-run by
default, refuses to replace any existing path, writes no Test mapping, and
never touches predicted streams, checkpoints, feature caches, or processes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "train_annotation": "7eb551fdeeeebc76b876ae255f91dc5662c7270a125955c5f1be2d9bd30921d0",
    "val_annotation": "0414885ee2702c2d3176cf6184e7811a7bd1c1347a157fef57a91020976776ee",
    "known_ids": "0a8684ee4df3c7315153bf7559066c7a9707fa46539aa54110e003bf7657cedc",
    "unknown_ids_val": "a05c7563ba06734b4ce755d9d574a1284e936babb7e592f381aee3862aca016f",
    "distractor_ids": "9dc538ee9ba3b1c9f2604315113baa204d566296507658d2312a497988a709fc",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mapping_specs(project_root: Path, tao_root: Path, role_root: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "train_annotation",
            "source": tao_root / "annotations/train.json",
            "target": project_root / "data/raw/tao/annotations/train.json",
            "kind": "file",
            "expected_sha256": EXPECTED["train_annotation"],
        },
        {
            "role": "val_annotation",
            "source": tao_root / "annotations/validation.json",
            "target": project_root / "data/raw/tao/annotations/validation.json",
            "kind": "file",
            "expected_sha256": EXPECTED["val_annotation"],
        },
        {
            "role": "frames_train_val_only",
            "source": tao_root / "frames",
            "target": project_root / "data/raw/tao/frames",
            "kind": "directory",
            "expected_sha256": None,
        },
        {
            "role": "known_ids",
            "source": role_root / "known_ids.json",
            "target": project_root / "data/tao_ow_ocd_v1/splits/known_ids.json",
            "kind": "file",
            "expected_sha256": EXPECTED["known_ids"],
        },
        {
            "role": "unknown_ids_val",
            "source": role_root / "unknown_ids_val.json",
            "target": project_root / "data/tao_ow_ocd_v1/splits/unknown_ids_val.json",
            "kind": "file",
            "expected_sha256": EXPECTED["unknown_ids_val"],
        },
        {
            "role": "distractor_ids",
            "source": role_root / "distractor_ids.json",
            "target": project_root / "data/tao_ow_ocd_v1/splits/distractor_ids.json",
            "kind": "file",
            "expected_sha256": EXPECTED["distractor_ids"],
        },
    ]


def _inspect(spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    source = Path(spec["source"])
    target = Path(spec["target"])
    item = {
        "role": spec["role"],
        "source": str(source),
        "source_resolved": str(source.resolve(strict=False)),
        "target": str(target),
        "target_resolved": str(target.resolve(strict=False)),
        "kind": spec["kind"],
        "expected_sha256": spec.get("expected_sha256"),
        "source_exists": source.exists(),
        "target_exists": target.exists() or target.is_symlink(),
        "target_is_symlink": target.is_symlink(),
        "action": "PENDING",
    }
    problems: list[str] = []
    if spec["kind"] == "file":
        if not source.is_file():
            problems.append(f"{spec['role']}:source_missing_or_not_file")
        else:
            item["source_size_bytes"] = source.stat().st_size
            item["source_sha256"] = _sha256(source)
            if item["source_sha256"] != spec["expected_sha256"]:
                problems.append(f"{spec['role']}:source_sha256_mismatch")
    elif spec["kind"] == "directory":
        if not source.is_dir():
            problems.append(f"{spec['role']}:source_missing_or_not_directory")
        else:
            item["source_entry_count"] = sum(1 for _ in source.iterdir())
    else:
        problems.append(f"{spec['role']}:unknown_kind")

    if item["target_exists"]:
        if not target.is_symlink():
            problems.append(f"{spec['role']}:refuse_existing_non_symlink_target")
        elif target.resolve(strict=False) != source.resolve(strict=False):
            problems.append(f"{spec['role']}:existing_symlink_points_elsewhere")
        else:
            item["action"] = "EXISTING_CORRECT_SYMLINK"
    else:
        item["action"] = "CREATE_SYMLINK"
    return item, problems


def build_report(project_root: Path, tao_root: Path, role_root: Path) -> dict[str, Any]:
    specs = mapping_specs(project_root, tao_root, role_root)
    assets: list[dict[str, Any]] = []
    problems: list[str] = []
    for spec in specs:
        item, item_problems = _inspect(spec)
        assets.append(item)
        problems.extend(item_problems)
    return {
        "schema_version": "trackocd.v2.reversible_local_mapping.v1",
        "status": "READY_TO_APPLY" if not problems else "REFUSED_SOURCE_OR_TARGET_CHECK",
        "generated_utc": _now(),
        "project_root": str(project_root.resolve(strict=False)),
        "tao_root": str(tao_root.resolve(strict=False)),
        "role_root": str(role_root.resolve(strict=False)),
        "read_only": True,
        "copy_performed": False,
        "test_semantic_accessed": False,
        "test_paths_mapped": [],
        "predicted_stream_touched": False,
        "checkpoint_touched": False,
        "formal_cache_touched": False,
        "assets": assets,
        "problems": problems,
    }


def apply_mapping(report: dict[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    if report["problems"]:
        raise RuntimeError("refusing mapping with source/target problems")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise RuntimeError(f"refusing to overwrite existing manifest: {manifest_path}")
    for item in report["assets"]:
        if item["action"] == "CREATE_SYMLINK":
            target = Path(item["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(Path(item["source"]).resolve(), target_is_directory=item["kind"] == "directory")
            item["action"] = "CREATED_SYMLINK"
    report["status"] = "APPLIED_REVERSIBLE_SYMLINK_MAPPING"
    report["read_only"] = False
    report["copy_performed"] = False
    report["write_performed"] = True
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--tao-root", type=Path, default=Path("/data3/liuyeqiang/TAO-Amodal"))
    parser.add_argument("--role-root", type=Path, default=ROOT / "outputs/trackocd_core/recovered_splits")
    parser.add_argument("--apply", action="store_true", help="create only the verified symlinks")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/trackocd_cvpr/v2_local_reversible_mapping_20261016.json")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    report = build_report(args.project_root, args.tao_root, args.role_root)
    if args.apply:
        report = apply_mapping(report, manifest_path=args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"READY_TO_APPLY", "APPLIED_REVERSIBLE_SYMLINK_MAPPING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
