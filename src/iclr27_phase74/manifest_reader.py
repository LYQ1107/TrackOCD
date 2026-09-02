"""Manifest reader that preserves the source file order exactly."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .io import canonical_hash, sha256


@dataclass(frozen=True)
class ManifestEvent:
    manifest_path: str
    manifest_realpath: str
    manifest_sha256: str
    manifest_line_number: int
    polarity: str
    polarity_file_ordinal: int
    original_event_index_within_file: int
    event_key: str
    fold: int
    kind: str
    raw: dict[str, Any]
    canonical_serialization_hash: str

    def as_record(self) -> dict[str, Any]:
        return {"manifest_path": self.manifest_path, "manifest_realpath": self.manifest_realpath,
                "manifest_sha256": self.manifest_sha256, "manifest_line_number": self.manifest_line_number,
                "polarity": self.polarity, "polarity_file_ordinal": self.polarity_file_ordinal,
                "original_event_index_within_file": self.original_event_index_within_file,
                "event_key": self.event_key, "fold": self.fold, "kind": self.kind,
                "raw": self.raw, "canonical_serialization_hash": self.canonical_serialization_hash}


def read_manifest_preserving_order(path: Path, polarity: str, file_ordinal: int) -> Iterator[ManifestEvent]:
    path = Path(path); digest = sha256(path); ordinal = 0
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            raw = __import__("json").loads(line)
            if not isinstance(raw, dict): raise ValueError(f"{path}:{line_no} not object")
            yield ManifestEvent(str(path), str(path.resolve()), digest, line_no, str(polarity), int(file_ordinal), ordinal,
                                str(raw.get("event_key", "")), int(raw.get("fold", -1)), str(raw.get("kind", "")), raw,
                                canonical_hash(raw))
            ordinal += 1


def read_both_preserving_order(positive: Path, negative: Path) -> list[ManifestEvent]:
    # This is the actual Phase19R fallback order (positive file then negative
    # file); no fold/event sorting is applied here.
    return list(read_manifest_preserving_order(positive, "positive", 0)) + list(read_manifest_preserving_order(negative, "negative", 1))


def manifest_contract(events: list[ManifestEvent], positive: Path, negative: Path) -> dict[str, Any]:
    pos = [e for e in events if e.polarity == "positive"]; neg = [e for e in events if e.polarity == "negative"]
    keys = [e.event_key for e in events]
    return {"schema_version": "phase74.manifest_order.v1", "reader": "read_manifest_preserving_order",
            "global_order_source": "Phase19R freeze_predictions.public_events fallback: positive_events.jsonl then negative_events.jsonl; existing public_model_events was not used",
            "positive_path": str(positive.resolve()), "negative_path": str(negative.resolve()),
            "positive_sha256": pos[0].manifest_sha256 if pos else None, "negative_sha256": neg[0].manifest_sha256 if neg else None,
            "positive_count": len(pos), "negative_count": len(neg), "total": len(events),
            "positive_line_order_preserved": [e.raw for e in pos] == [e.raw for e in read_manifest_preserving_order(positive, "positive", 0)],
            "negative_line_order_preserved": [e.raw for e in neg] == [e.raw for e in read_manifest_preserving_order(negative, "negative", 1)],
            "event_key_unique": len(keys) == len(set(keys)), "event_key_set_count": len(set(keys)),
            "fold_counts": {str(f): {p: sum(1 for e in events if e.fold == f and e.polarity == p) for p in ("positive", "negative")} for f in range(4)},
            "implicit_sort_used": False, "key_order_sha256": canonical_hash(keys)}
