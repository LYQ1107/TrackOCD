#!/usr/bin/env python3
"""Six bounded feature-only fits; no DINO/detector/tracker/Val/Test training."""
from __future__ import annotations

import datetime as dt
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_core.smoke_gt_features import choose_idle_gpu
from src.trackocd_core.features import CompactGTFeasibilityCache, PREFIXES
from src.trackocd_core.experiment_config import representation_config
from src.trackocd_v2.io import atomic_json, sha256_file


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/trackocd_core/gt_representation_pilot.json')
    config_path = parser.parse_args().config.resolve()
    config = representation_config(ROOT, config_path)
    data_config_path = ROOT / config["data_plan"]
    data_config = json.loads(data_config_path.read_text())
    output = ROOT / config["output_directory"]
    if output.exists():
        raise RuntimeError("Preserve previous bounded fits; no silent overwrite/retry")
    mem = {k: int(v.split()[0]) for k, v in (s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())}
    if mem["MemAvailable"] * 1024 - config["ram_plan_bytes"] < mem["MemTotal"] * 1024 * .25:
        raise RuntimeError("Resource wait: RAM plan crosses 25% headroom")
    cache_root = ROOT / data_config["cache_directory"]
    cache = CompactGTFeasibilityCache(cache_root)
    if cache.manifest["pilot_config"]["sha256"] != sha256_file(data_config_path):
        raise ValueError("Data differs from original pre-feature selection")
    uuid = choose_idle_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    import torch
    import pyarrow.parquet as pq
    from torch.nn import functional as F
    from src.trackocd_core.representation import CategoryEvidence, cross_video_category_loss

    torch.set_num_threads(config["torch_cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    labels = pq.read_table(cache_root / "train_labels.parquet").to_pylist()
    routes = {r["key"]: r for r in pq.read_table(cache_root / "index.parquet").to_pylist()}
    fit = [r for r in labels if r["purpose"] == "representation_fit" and r["partition"] == "representation"]
    if len(fit) != config["batch_tracks"] or {r["category_id"] for r in fit} != set(cache.manifest["selection"]["selected_known_categories"]):
        raise ValueError("Not the registered Train representation-fit subset")
    views = [cache.get_prefix(r["key"], 16) for r in fit]
    visual = np.stack([v.visual for v in views])
    quality = np.stack([v.quality for v in views])
    elapsed = np.stack([v.elapsed_frames for v in views])
    categories = np.array([r["category_id"] for r in fit])
    videos = torch.tensor([routes[r["key"]]["video_id"] for r in fit], device="cuda:0")
    identities = torch.tensor([routes[r["key"]]["physical_track_id"] for r in fit], device="cuda:0")
    category_tensor = torch.tensor(categories, device="cuda:0")
    output.mkdir(parents=True)
    fits, checkpoint_bytes = [], 0
    for seed in config["training_seeds"]:
        for learned, name in ((False, "A1_semantic_adapter_mean"), (True, "A2_semantic_adapter_evidence")):
            torch.manual_seed(seed)
            model = CategoryEvidence(learned).to("cuda:0").train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            rng = np.random.default_rng(seed + 11027)  # Independent of model RNG; identical A1/A2 inputs.
            trace, corruption_count, fit_started = [], 0, time.monotonic()
            for step in range(config["steps_per_model_seed"]):
                if time.monotonic() - started > config["maximum_training_wall_seconds"]:
                    raise RuntimeError("Registered short-training wall budget exceeded")
                if step % 24 == 0:
                    available = {k: int(v.split()[0]) for k, v in (s.split(":", 1) for s in Path("/proc/meminfo").read_text().splitlines())}
                    if available["MemAvailable"] * 1024 - config["ram_plan_bytes"] < available["MemTotal"] * 1024 * .25:
                        raise RuntimeError("Stop own bounded fit: system RAM headroom crossed")
                p = PREFIXES[step % len(PREFIXES)]
                augmented = visual[:, :p].copy()
                augmented += rng.normal(0, config["corruption"]["feature_noise_std"], augmented.shape).astype(np.float32)
                clean = np.ones((len(fit), p), dtype=np.float32)
                for index in range(len(fit)):
                    if rng.random() < config["corruption"]["track_corruption_probability"]:
                        other = int(rng.choice(np.flatnonzero(categories != categories[index])))
                        frame = int(rng.integers(p))
                        augmented[index, frame] = visual[other, frame]
                        clean[index, frame] = 0
                        corruption_count += 1
                augmented /= np.maximum(np.linalg.norm(augmented, axis=-1, keepdims=True), 1e-12)
                result = model(torch.tensor(augmented, device="cuda:0"), torch.tensor(quality[:, :p], device="cuda:0"),
                               torch.tensor(elapsed[:, :p], device="cuda:0", dtype=torch.float32))
                category_loss = cross_video_category_loss(result["embedding"], category_tensor, videos, identities,
                                                         config["contrastive_temperature"])
                reliability_loss = F.binary_cross_entropy_with_logits(result["reliability_logits"], torch.tensor(clean, device="cuda:0")) if learned else torch.zeros((), device="cuda:0")
                geometry_loss = torch.zeros((), device="cuda:0")
                if config.get("teacher_geometry_loss_weight", 0):
                    teacher = F.normalize(torch.tensor(augmented, device="cuda:0").mean(dim=1), dim=-1).detach()
                    geometry_loss = F.mse_loss(result["embedding"] @ result["embedding"].T, teacher @ teacher.T)
                loss = (category_loss + config["reliability_loss_weight"] * reliability_loss
                        + config.get("teacher_geometry_loss_weight", 0) * geometry_loss)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite bounded fit loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                optimizer.step()
                trace.append({"step": step + 1, "prefix": p, "category_loss": float(category_loss.detach().cpu()),
                              "synthetic_reliability_loss": float(reliability_loss.detach().cpu()),
                              "fit_only_teacher_geometry_loss": float(geometry_loss.detach().cpu())})
            model.eval()
            checkpoint = output / f"{name}_seed{seed}.pt"
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "seed": seed, "model": name, "config_sha256": sha256_file(config_path),
                        "feature_manifest_sha256": sha256_file(cache_root / "manifest.json")}, temporary)
            temporary.rename(checkpoint)
            checkpoint_bytes += checkpoint.stat().st_size
            if checkpoint_bytes > config["maximum_checkpoint_payload_bytes"]:
                raise ValueError("Bounded checkpoint payload budget exceeded")
            fits.append({"model": name, "seed": seed, "fit_tracks": len(fit), "fit_observations": int(visual.shape[0] * visual.shape[1]),
                         "fit_categories": sorted(set(map(int, categories))), "fit_videos": len(set(videos.cpu().tolist())),
                         "parameters": sum(p.numel() for p in model.parameters()), "steps": len(trace), "trace": trace,
                         "synthetic_corruptions": corruption_count, "wall_seconds": time.monotonic() - fit_started,
                         "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint)}})
            print(json.dumps({k: fits[-1][k] for k in ("model", "seed", "parameters", "steps", "wall_seconds")}), flush=True)
            del optimizer, model
    receipt = {"schema_version": "trackocd.core.gt-representation-fit.v1", "status": "COMPLETE_BOUNDED_FIT_NOT_SCIENTIFIC_PASS",
               "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "config_sha256": sha256_file(config_path),
               "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"), "fits": fits, "new_checkpoint_bytes": checkpoint_bytes,
               "A0_no_optimizer_or_training": True, "DINO_detector_tracker_trained": False,
               "only_registered_representation_fit_rows_used": True, "policy_or_heldout_rows_used_for_fit": False,
               "novel_or_val_labels_used_for_fit": False, "test_accessed": False, "external_process_interference": False,
               "loss_drop_is_not_unseen_category_generalization_proof": True, "formal_M5_M8_complete": False,
               "root_cause_correction_rounds_used": config["root_cause_correction_rounds_used"],
               "parent_config_sha256": config.get("parent_config_sha256"),
               "source_sha256": {name: sha256_file(ROOT / name) for name in
                                 ("scripts/trackocd_core/train_gt_representation_pilot.py", "src/trackocd_core/representation.py")},
               "resources": {"wall_seconds": time.monotonic() - started, "worker_count": 1, "gpu_uuid": uuid,
                             "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "peak_cpu_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "initial_mem_available_kib": mem["MemAvailable"], "ram_plan_bytes": config["ram_plan_bytes"]}}
    atomic_json(output / "training_receipt.json", receipt)
    prefix = "gt_representation_r1" if config["root_cause_correction_rounds_used"] else "gt_representation"
    atomic_json(ROOT / f"outputs/trackocd_core/audit/{prefix}_training.json", receipt)
    print(json.dumps({"status": receipt["status"], "fits": len(fits), "new_checkpoint_bytes": checkpoint_bytes, "resources": receipt["resources"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
