"""Canonical-content lookup for Q0 physical rows."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .asset_identity import AssetRecord
from .io import iter_json_array


class PhysicalIndex:
    """Index Q0 rows by content asset while retaining candidate order."""

    def __init__(self, rows_by_content: Mapping[str, list[dict[str, Any]]] | None = None) -> None:
        self.rows_by_content: dict[str, list[dict[str, Any]]] = {str(k): list(v) for k, v in (rows_by_content or {}).items()}

    @classmethod
    def from_stream(cls, stream_path: Path, q0_assets: Iterable[AssetRecord], allowed_content: set[str] | None = None) -> "PhysicalIndex":
        image_content = {int(record.image_id): record.content_asset_key for record in q0_assets if record.image_id is not None}
        index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in iter_json_array(stream_path):
            try:
                image_id = int(row.get("image_id"))
            except (TypeError, ValueError):
                continue
            content = image_content.get(image_id)
            if content is None or (allowed_content is not None and content not in allowed_content):
                continue
            index[content].append({**dict(row), "content_asset_key": content, "candidate_order": len(index[content])})
        return cls(index)

    def lookup_image(self, content_asset_key: str) -> list[dict[str, Any]]:
        return list(self.rows_by_content.get(str(content_asset_key), []))

    def track_rows(self, content_asset_key: str) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.lookup_image(content_asset_key):
            key = f"v{row.get('video_id')}:p{row.get('track_id')}"
            grouped[key].append(row)
        return dict(grouped)
