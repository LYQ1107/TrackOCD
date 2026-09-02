"""Evidence-aware mapping gate; numeric overlap alone is never accepted."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MappingEvidence:
    mapping_type: str
    provenance_verified: bool
    one_to_one: bool
    category_independent: bool
    track_id_value_independent: bool
    frame_identity_verified: bool
    bbox_space_verified: bool

    def legal(self) -> bool:
        return all((self.provenance_verified, self.one_to_one, self.category_independent,
                    self.track_id_value_independent, self.frame_identity_verified, self.bbox_space_verified))


def assess_mapping(*, mapping_type: str, provenance_verified: bool, one_to_one: bool, category_used: bool = False,
                   track_id_used: bool = False, frame_identity_verified: bool = False, bbox_space_verified: bool = False,
                   numeric_only: bool = False) -> dict[str, Any]:
    ev = MappingEvidence(mapping_type, provenance_verified and not numeric_only, one_to_one, not category_used,
                         not track_id_used, frame_identity_verified, bbox_space_verified)
    return {"mapping_type": ev.mapping_type, "provenance_verified": ev.provenance_verified, "one_to_one": ev.one_to_one,
            "category_independent": ev.category_independent, "track_id_value_independent": ev.track_id_value_independent,
            "frame_identity_verified": ev.frame_identity_verified, "bbox_space_verified": ev.bbox_space_verified,
            "legal": ev.legal(), "numeric_only_rejected": bool(numeric_only)}
