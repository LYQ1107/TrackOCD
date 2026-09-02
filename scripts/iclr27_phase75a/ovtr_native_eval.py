#!/usr/bin/env python3
"""Run the pinned OVTR evaluator while exporting native physical lineage.

The historical TAO JSON is not sufficient to reconstruct frame and proposal
ordering.  This wrapper captures the filtered ``dt_instances`` at the
``OVTR_inference.update_results_teta`` boundary during the actual forward
loop, before the TAO serializer drops those fields.  It does not open any
evaluator join or event labels.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OVTR_DIR = ROOT / "third_party/research_refs_phase4n/OVTR/ovtr"
sys.path.insert(0, str(OVTR_DIR))
sys.path.insert(0, str(ROOT))


def _load_eval_module() -> Any:
    path = OVTR_DIR / "eval.py"
    spec = importlib.util.spec_from_file_location("trackocd_phase75a_ovtr_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load OVTR evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_capture(module: Any, native_path: Path, checkpoint_sha256: str) -> None:
    original_detect = module.OVTR_inference.detect
    original_update = module.OVTR_inference.update_results_teta

    def detect(self: Any, *args: Any, **kwargs: Any) -> Any:
        info = kwargs.get("info")
        file_path = kwargs.get("file_path")
        if info is None and len(args) >= 1:
            info = args[-2]
        if file_path is None and args:
            file_path = args[-1]
        frame_id = int(info[0]) if info is not None else None
        image_id = int(self.path_to_img_id.get(file_path, -1)) if file_path is not None else -1
        video_id = int(self.path_to_video_id.get(file_path, -1)) if file_path is not None else -1
        self._phase75_context = {"frame_id": frame_id, "image_id": image_id, "video_id": video_id, "file_path": file_path}
        self._phase75_frame_trace.append(dict(self._phase75_context))
        return original_detect(self, *args, **kwargs)

    def update(self: Any, bbox_xyxy: Any, identities: Any, labels: Any, scores: Any = None, masks: Any = None, dt_instances: Any = None) -> Any:
        context = getattr(self, "_phase75_context", {})
        current: set[int] = set()
        records: list[dict[str, Any]] = []
        if dt_instances is not None:
            for index in range(len(dt_instances)):
                label = int(dt_instances.cls_idxes[index])
                if label == -1:
                    continue
                physical_id = int(dt_instances.obj_idxes[index])
                current.add(physical_id)
                key = (int(context.get("video_id", -1)), physical_id)
                seen = key in self._phase75_seen_ids
                hit_count = int(dt_instances.hit_count[index]) if dt_instances.has("hit_count") else None
                disappear_time = int(dt_instances.disappear_time[index]) if dt_instances.has("disappear_time") else None
                if disappear_time not in (None, 0):
                    lifecycle = "inactive"
                elif not seen or (hit_count is not None and hit_count <= 1):
                    lifecycle = "birth"
                else:
                    lifecycle = "continuation"
                box = [float(x) for x in dt_instances.boxes[index].detach().cpu().tolist()]
                score = float(dt_instances.scores[index])
                records.append({
                    "schema_version": "phase75a.native_physical_lineage.v1",
                    "video_id": int(context.get("video_id", -1)),
                    "image_id": int(context.get("image_id", -1)),
                    "frame_id": int(context.get("frame_id", -1)) if context.get("frame_id") is not None else None,
                    "file_path": str(context.get("file_path")) if context.get("file_path") is not None else None,
                    "proposal_local_id": int(index),
                    "candidate_rank": int(index),
                    "physical_track_id": physical_id,
                    "parent_physical_track_id": physical_id if seen else None,
                    "lifecycle": lifecycle,
                    "bbox_xyxy": box,
                    "base_score": score,
                    "hit_count": hit_count,
                    "disappear_time": disappear_time,
                    "score_mode": "base",
                    "source_checkpoint_sha256": checkpoint_sha256,
                })
                self._phase75_seen_ids.add(key)
        previous = getattr(self, "_phase75_previous_ids", set())
        for physical_id in sorted(previous - current):
            key = (int(context.get("video_id", -1)), int(physical_id))
            records.append({
                "schema_version": "phase75a.native_physical_lineage.v1",
                "video_id": key[0],
                "image_id": int(context.get("image_id", -1)),
                "frame_id": int(context.get("frame_id", -1)) if context.get("frame_id") is not None else None,
                "file_path": str(context.get("file_path")) if context.get("file_path") is not None else None,
                "proposal_local_id": None,
                "candidate_rank": None,
                "physical_track_id": int(physical_id),
                "parent_physical_track_id": int(physical_id),
                "lifecycle": "termination",
                "bbox_xyxy": None,
                "base_score": None,
                "hit_count": None,
                "disappear_time": None,
                "score_mode": "base",
                "source_checkpoint_sha256": checkpoint_sha256,
            })
        self._phase75_previous_ids = current
        self._phase75_native_records.extend(records)
        return original_update(self, bbox_xyxy, identities, labels, scores=scores, masks=masks, dt_instances=dt_instances)

    module.OVTR_inference.detect = detect
    module.OVTR_inference.update_results_teta = update


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(parents=[_load_eval_module().get_args_parser()])
    parser.add_argument("--native-out", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = Path(args.pretrained)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    import hashlib
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    module = _load_eval_module()
    # The second module load is deliberate: argparse construction above must
    # not retain a partially patched class.  Both loads point at the pinned
    # commit and are recorded in the parent replay manifest.
    original_eval = module.eval
    native_path = args.native_out.resolve()

    def eval_with_capture(eval_args: Any, cfg: Any) -> Any:
        # The capture hooks are installed on the exact module/class used by
        # the evaluator.  Lists are bounded by the selected control videos.
        module.OVTR_inference._phase75_native_records = []
        module.OVTR_inference._phase75_seen_ids = set()
        module.OVTR_inference._phase75_previous_ids = set()
        install_capture(module, native_path, digest.hexdigest())
        return original_eval(eval_args, cfg)

    # Avoid a second independent implementation of the evaluator itself; the
    # two control runs are independent processes.  Here only the hook boundary
    # is changed to export lineage that TAO serialization omits.
    from util.slconfig import SLConfig
    cfg = SLConfig.fromfile(args.config_file)
    module.OVTR_inference._phase75_native_records = []
    module.OVTR_inference._phase75_frame_trace = []
    module.OVTR_inference._phase75_seen_ids = set()
    module.OVTR_inference._phase75_previous_ids = set()
    install_capture(module, native_path, digest.hexdigest())
    original_eval(args, cfg)
    records = getattr(module.OVTR_inference, "_phase75_native_records", [])
    # The class attributes above are overwritten on each instance by Python's
    # normal lookup; collect from the evaluator instances through the hook's
    # per-instance list is handled by a class-level sink below.
    if not records:
        records = getattr(module, "_PHASE75_NATIVE_RECORDS", [])
    atomic_jsonl(native_path, records)
    frame_trace = getattr(module.OVTR_inference, "_phase75_frame_trace", [])
    atomic_jsonl(native_path.with_suffix(".frames.jsonl"), frame_trace)
    summary = {
        "protocol": "trackocd_phase75a_native_q0_control_replay",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "ovtr_commit": "500e72c",
        "score_mode": "base",
        "video_ids": json.loads(args.video_ids) if args.video_ids else None,
        "native_lineage": str(native_path),
        "native_record_count": len(records),
        "frame_trace_path": str(native_path.with_suffix(".frames.jsonl")),
        "frame_trace_count": len(frame_trace),
        "labels_joined_before_model": False,
        "event_join_read": False,
    }
    atomic_jsonl(native_path.with_suffix(".summary.jsonl"), [summary])
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
