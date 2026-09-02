"""Small Phase74S contract tests; no project event data is loaded."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.iclr27_phase74s.io import atomic_jsonl, canonical_hash, iter_jsonl
from src.iclr27_phase74s.protocol import build_model_contract


def _track(video: int, physical: int) -> str:
    return f"v{video}:p{physical}"


class Phase74SContractTest(unittest.TestCase):
    def test_model_contract_hides_evaluator_fields_and_preserves_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "outputs/iclr27_phase19r/manifests"
            manifest.mkdir(parents=True)
            positives = []
            negatives = []
            model = []
            for index in range(2):
                row = {
                    "event_key": f"event_{index}",
                    "fold": index,
                    "category_gt_denominator_only": 100 + index,
                    "source_tracklet_keys": [_track(index, 10 + index)],
                    "target_tracklet_key": _track(index + 1, 20 + index),
                    "source_video": index,
                    "target_video": index + 1,
                    "target_first_reliable_prefix_index_gt_only": 4,
                    "kind": "positive",
                }
                positives.append(row)
                model.append({
                    "event_key": row["event_key"],
                    "source_tracklet_keys": row["source_tracklet_keys"],
                    "target_tracklet_key": row["target_tracklet_key"],
                    "target_video": row["target_video"],
                })
            for index in range(2, 4):
                row = {
                    "event_key": f"event_{index}",
                    "fold": index - 2,
                    "target_category_gt_denominator_only": 200 + index,
                    "source_tracklet_keys": [_track(index, 10 + index)],
                    "target_tracklet_key": _track(index + 1, 20 + index),
                    "source_video": index,
                    "target_video": index + 1,
                    "target_first_reliable_prefix_index_gt_only": 8,
                    "kind": "negative",
                }
                negatives.append(row)
                model.append({
                    "event_key": row["event_key"],
                    "source_tracklet_keys": row["source_tracklet_keys"],
                    "target_tracklet_key": row["target_tracklet_key"],
                    "target_video": row["target_video"],
                })
            # The production contract requires exactly 76 rows per polarity;
            # this fixture exercises the same checks at the small-test scale
            # through the helper below rather than weakening production code.
            self.assertEqual(len(positives) + len(negatives), 4)
            path_map = {
                "held_known_positive_events.jsonl": positives,
                "held_known_negative_events.jsonl": negatives,
                "held_known_model_events.jsonl": model,
            }
            for name, rows in path_map.items():
                atomic_jsonl(manifest / name, rows)
            # Expand the fixture to the production cardinality while retaining
            # deterministic source/target consistency.
            for name, rows in (("held_known_positive_events.jsonl", positives), ("held_known_negative_events.jsonl", negatives)):
                base = list(rows)
                while len(rows) < 76:
                    i = len(rows)
                    src = i % 8
                    row = dict(base[i % len(base)])
                    polarity_prefix = "pos" if "positive" in name else "neg"
                    row["event_key"] = f"{polarity_prefix}_{i}"
                    row["fold"] = i % 4
                    row["source_tracklet_keys"] = [_track(src, 100 + i)]
                    row["target_tracklet_key"] = _track(src + 1, 200 + i)
                    row["source_video"] = src
                    row["target_video"] = src + 1
                    rows.append(row)
                atomic_jsonl(manifest / name, rows)
            model_rows = []
            for polarity_name in ("held_known_positive_events.jsonl", "held_known_negative_events.jsonl"):
                for row in iter_jsonl(manifest / polarity_name):
                    model_rows.append({
                        "event_key": row["event_key"],
                        "source_tracklet_keys": row["source_tracklet_keys"],
                        "target_tracklet_key": row["target_tracklet_key"],
                        "target_video": row["target_video"],
                    })
            atomic_jsonl(manifest / "held_known_model_events.jsonl", model_rows)
            result = build_model_contract(root)
            self.assertEqual(result["contract"]["model_event_count"], 152)
            self.assertEqual(result["contract"]["join_count"], 152)
            self.assertFalse(result["contract"]["forbidden_model_fields_seen"])
            self.assertEqual(result["model_records"][0]["model_event_uid"], "evt_000000")
            self.assertNotIn("event_key", result["model_records"][0])
            self.assertEqual(result["contract"]["model_order_sha256"], canonical_hash(result["model_records"]))

    def test_jsonl_round_trip_is_atomic_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            atomic_jsonl(path, ({"i": 1}, {"i": 2}))
            self.assertEqual(list(iter_jsonl(path)), [{"i": 1}, {"i": 2}])
            self.assertTrue(json.loads(path.read_text().splitlines()[0])["i"] == 1)


if __name__ == "__main__":
    unittest.main()
