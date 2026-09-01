"""Register deterministic, category- and video-disjoint Phase-15 roles."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> None:
    split14 = json.loads((ROOT / "outputs/iclr27_phase14b/manifests/devplus_split.json").read_text())
    class_split = json.loads((ROOT / "outputs/iclr27_phase7c/assets/class_split_hard.json").read_text())
    z = np.load(ROOT / "outputs/iclr27_phase6d/assets/full_tao_tracks.npz", allow_pickle=False)
    labels = z["labels"].astype(np.int64)
    videos = z["video_ids"].astype(np.int64)
    tracks = z["track_ids"].astype(np.int64)
    dev_videos = set(int(x) for x in split14["devplus_videos"])
    train_categories = set(int(x) for x in class_split["train_visible"])
    cal_categories = set(int(x) for x in class_split["hidden_train"])
    meta_categories = set(int(x) for x in class_split["hidden_val"])
    candidate = set(int(x) for x in np.unique(videos)) - dev_videos

    # Whole videos are assigned by priority so no public row can appear in two
    # roles.  Category sets are already disjoint in class_split_hard.json.
    meta_videos = set(int(v) for v in candidate if np.any((videos == v) & np.isin(labels, list(meta_categories))))
    cal_videos = set(int(v) for v in candidate - meta_videos if np.any((videos == v) & np.isin(labels, list(cal_categories))))
    train_videos = candidate - meta_videos - cal_videos

    roles = {
        "representation_train": (train_videos, train_categories),
        "calibration": (cal_videos, cal_categories),
        "meta_validation": (meta_videos, meta_categories),
    }
    role_payload = {}
    role_of_track = np.full(len(labels), "excluded", dtype=object)
    for role, (vset, cset) in roles.items():
        mask = np.isin(videos, list(vset)) & np.isin(labels, list(cset))
        idx = np.where(mask)[0]
        role_of_track[idx] = role
        role_payload[role] = {
            "videos": sorted(vset),
            "categories": sorted(cset),
            "track_indices": [int(x) for x in idx],
            "track_ids": [f"{int(videos[i])}_{int(tracks[i])}" for i in idx],
            "n_tracks": int(len(idx)),
            "n_categories_present": int(len(set(labels[idx].tolist()))),
            "n_videos_present": int(len(set(videos[idx].tolist()))),
        }

    video_sets = [set(v) for v, _ in roles.values()]
    cat_sets = [set(c) for _, c in roles.values()]
    audit = {
        "protocol": "phase15",
        "source": "outputs/iclr27_phase6d/assets/full_tao_tracks.npz",
        "source_keys": list(z.files),
        "source_track_rows": int(len(labels)),
        "devplus_videos_excluded": sorted(dev_videos),
        "roles": role_payload,
        "video_overlap_pairs": {
            "train_calibration": len(video_sets[0] & video_sets[1]),
            "train_meta": len(video_sets[0] & video_sets[2]),
            "calibration_meta": len(video_sets[1] & video_sets[2]),
        },
        "category_overlap_pairs": {
            "train_calibration": len(cat_sets[0] & cat_sets[1]),
            "train_meta": len(cat_sets[0] & cat_sets[2]),
            "calibration_meta": len(cat_sets[1] & cat_sets[2]),
        },
        "devplus_used_for_fit": False,
        "devplus_used_for_calibration": False,
        "q1_label_used": False,
        "private_gt_used": False,
        "physical_id_used_as_feature": False,
        "future_frames_used": False,
        "pair_labels_use_public_categories": True,
        "pass": bool(
            all(len(video_sets[i] & video_sets[j]) == 0 for i in range(3) for j in range(i + 1, 3))
            and all(len(cat_sets[i] & cat_sets[j]) == 0 for i in range(3) for j in range(i + 1, 3))
            and all(not (set(role_payload[r]["videos"]) & dev_videos) for r in role_payload)
        ),
    }
    # Keep the exact scientific configuration beside the split.  This file is
    # written before any Phase-15 metric is observed.
    prereg = {
        "protocol": "phase15",
        "registration_date": "2026-08-24",
        "source": audit["source"],
        "split": role_payload,
        "devplus_videos": sorted(dev_videos),
        "devplus_novel_categories": sorted(int(x) for x in split14["selected_devplus_categories"]),
        "prefixes": [1, 2, 4, 8, 16],
        "public_prefix_max": 8,
        "raw_feature_dim": 768,
        "verifier_training_prefix": 8,
        "pair_model": "MLP([z_i,z_j,abs(z_i-z_j),z_i*z_j]) -> relation probability",
        "pair_controls": ["raw_cosine", "temporal_only"],
        "pair_seeds": [20260824, 20260825],
        "pair_sampling": {
            "positive": "distinct physical tracks, prefer different videos",
            "negative": "balanced random and hard raw-cosine negatives",
            "grouped_by_category": True,
        },
        "causal_state": {
            "max_exemplars_per_state": 4,
            "exemplar_rule": "first occurrence plus highest-confidence later assignments, causal only",
            "physical_track_carry_forward": "first birth match, later occurrences carry immutable action",
            "known_threshold_source": "calibration pairs",
            "existing_threshold_source": "calibration pairs",
            "new_threshold_source": "calibration pairs",
        },
        "final_gate": "Known occurrence accuracy >= 0.60 AND CT-Reuse > 0",
        "q1_quarantined": True,
        "branch_logic": "docs/iclr27_phase15/PROTOCOL.md",
    }
    atomic_json(ROOT / "outputs/iclr27_phase15/manifests/data_and_leakage_audit.json", audit)
    atomic_json(ROOT / "outputs/iclr27_phase15/manifests/phase15_preregistration.json", prereg)
    print(json.dumps({"audit_pass": audit["pass"], "roles": {k: {q: v[q] for q in ("n_tracks", "n_categories_present", "n_videos_present")} for k, v in role_payload.items()}}, indent=2))


if __name__ == "__main__":
    main()
