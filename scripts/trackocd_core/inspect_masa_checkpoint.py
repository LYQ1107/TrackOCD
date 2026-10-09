#!/usr/bin/env python3
"""Inspect a pinned frozen checkpoint safely; never construct/run a model."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import pickletools
import re
import resource
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json


def prefix_summary(state) -> dict:
    """Tensor-key coverage is structural evidence, not exact model compatibility."""
    groups = collections.defaultdict(lambda: {"tensors": 0, "elements": 0, "tensor_bytes": 0})
    for key, tensor in state.items():
        if not isinstance(key, str) or not hasattr(tensor, "numel"):
            raise ValueError("Expected a tensor-only state_dict")
        parts = key.split(".")
        prefix = ".".join(parts[:2]) if parts[0] == "detector" else parts[0]
        group = groups[prefix]
        group["tensors"] += 1
        group["elements"] += tensor.numel()
        group["tensor_bytes"] += tensor.numel() * tensor.element_size()
    return dict(sorted(groups.items()))


def pickle_global_names(path: Path) -> list[str]:
    """Read opcode names only; no unpickle, class import, REDUCE or code execution."""
    with zipfile.ZipFile(path) as archive:
        infos = [i for i in archive.infolist() if i.filename.endswith("/data.pkl")]
        if len(infos) != 1 or infos[0].file_size > 8 * 1024**2:
            raise ValueError("Unexpected pickle metadata layout or size")
        payload = archive.read(infos[0])
    return sorted({arg.replace(" ", ".") for op, arg, _ in pickletools.genops(payload) if op.name == "GLOBAL"})


def main() -> None:
    started = time.perf_counter()
    plan_path = ROOT / "configs/trackocd_core/masa_sam_candidate_smoke.json"
    plan = json.loads(plan_path.read_text())
    spec = plan["weights"][0]
    path = ROOT / "checkpoints/masa_sam_candidate" / spec["filename"]
    if path.is_symlink() or path.resolve().parent != (ROOT / "checkpoints/masa_sam_candidate").resolve():
        raise ValueError("Unexpected checkpoint destination")
    if path.stat().st_size != spec["bytes"]:
        raise ValueError("Checkpoint size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != spec["sha256_from_repository_lfs_metadata"]:
        raise ValueError("Checkpoint SHA256 mismatch")
    globals_found = pickle_global_names(path)
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    ram_fraction = int(mem["MemAvailable"].split()[0]) / int(mem["MemTotal"].split()[0])
    if ram_fraction < plan["execution_limits"]["minimum_system_ram_headroom_fraction"]:
        raise RuntimeError("Insufficient system RAM headroom")
    import torch
    torch.set_num_threads(1)
    result = {
        "schema_version": "trackocd.core.masa_checkpoint_recovery.v1",
        "inspected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_url": plan["weight_repository"] + "/resolve/" + plan["weight_repository_revision"] + "/" + spec["filename"],
        "weight_repository_revision": plan["weight_repository_revision"],
        "checkpoint_path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
        "size_and_sha256_match": True,
        "safe_load": {"weights_only": True, "mmap": True, "map_location": "cpu",
                      "extra_globals_allowlisted": [], "unsafe_fallback": False,
                      "pickle_GLOBAL_names_inspected_without_execution": globals_found},
        "tensor_structure": None,
        "exact_runtime_model_key_and_shape_match_verified": False,
        "second_pretrain_weight_downloaded": False,
        "second_pretrain_weight_need": "Deferred until exact frozen model load; prefix presence alone is insufficient",
        "resources": {"workers": 1, "system_ram_headroom_fraction_before": ram_fraction},
        "boundary": {"model_constructed": False, "model_training_or_inference": False,
                     "checkpoint_metadata_as_model_input": False, "annotations_or_gt_opened": False,
                     "test_access": False, "base_environment_changed": False,
                     "foreign_process_interference": False, "primary_frontend_selected": False}}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        meta = payload.get("meta", {})
        result["released_training_metadata"] = {
            "meta_field_present": "meta" in payload,
            "metadata_key_count": len(meta) if isinstance(meta, dict) else None,
            "saved_training_config_present": isinstance(meta, dict) and isinstance(meta.get("config"), str),
            "complete_release_stage_training_binding_verified": False}
        state = payload["state_dict"]
        if not isinstance(state, dict) or not all(isinstance(t, torch.Tensor) for t in state.values()):
            raise ValueError("Non-tensor state_dict")
        groups = prefix_summary(state)
        required = ["detector.backbone", "masa_adapter", "rpn_head", "roi_head", "track_head"]
        selected = ["detector.backbone.pos_embed", "detector.backbone.patch_embed.proj.weight",
                    "rpn_head.rpn_cls.weight", "rpn_head.rpn_reg.weight",
                    "roi_head.bbox_head.fc_cls.weight", "roi_head.bbox_head.fc_reg.weight"]
        result["tensor_structure"] = {
            "groups": groups, "total_tensors": len(state),
            "all_required_component_prefixes_present": all(p in groups for p in required),
            "required_prefixes": required,
            "selected_tensor_shapes": {key: list(state[key].shape) if key in state else None for key in selected},
            "backbone_block_indices": sorted({int(m.group(1)) for k in state
                                              if (m := re.match(r"detector\.backbone\.blocks\.(\d+)\.", k))}),
            "full_numeric_finiteness_scan_performed": False,
            "metadata_text_or_category_names_emitted": False}
        result["status"] = "PASS_SAFE_TENSOR_STRUCTURE_NOT_RUNTIME_OR_FRONTEND_QUALIFICATION"
    except Exception as exc:
        # No torch.load(weights_only=False), metadata evaluation or automatic allowlist.
        result["status"] = "BLOCKED_SAFE_CHECKPOINT_STRUCTURE_INSPECTION"
        result["safe_load"]["error_type"] = type(exc).__name__
        result["safe_load"]["unsupported_global_names"] = re.findall(r"Unsupported global: GLOBAL ([\w.]+)", str(exc))
    result["resources"].update(elapsed_seconds=time.perf_counter() - started,
                                 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    destination = ROOT / "outputs/trackocd_core/audit/masa_checkpoint_recovery.json"
    atomic_json(destination, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
