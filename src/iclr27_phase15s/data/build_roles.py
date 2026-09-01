"""Deterministic public-TRAIN role construction for Phase 15S.

Known recognition and unseen-category correspondence are separate problems.
Known roles use every legal public TRAIN video that is not DEV+/Q1; the
category-disjoint Phase-15 meta role is retained as a *novel* audit and is not
used to fit the known bank.  Role membership is video-disjoint and all
fallbacks are explicit in the JSON output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
Q1_VIDEOS = {88, 90, 122, 291, 334, 888, 931, 1159, 1232, 1276, 1572,
             1865, 2254, 2347, 2564, 2675, 2690, 2759, 2802, 2888}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.resolve().open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def construct(annotation: Path, prereg: Path, known_path: Path, out: Path,
              bank_max: int = 320, role_max: int = 40) -> dict:
    src = json.loads(annotation.read_text())
    manifest = json.loads(prereg.read_text())
    # Frozen DEV+ identities are inherited from the historical registration;
    # the Phase15S preregistration intentionally records the source rather
    # than duplicating a 130-video list.
    frozen_manifest = json.loads((ROOT / "outputs/iclr27_phase15/manifests/phase15_preregistration.json").read_text())
    known = {int(x) for x in json.loads(known_path.read_text())}
    dev_videos = {int(x) for x in frozen_manifest["devplus_videos"]}
    all_videos = {int(v["id"]) for v in src.get("videos", [])}
    legal = sorted(all_videos - dev_videos - Q1_VIDEOS)
    image_video = {int(i["id"]): int(i["video_id"]) for i in src.get("images", [])}
    video_frames = Counter(int(i["video_id"]) for i in src.get("images", []))
    video_ann = Counter()
    video_tracks: dict[int, set[tuple[int, int]]] = defaultdict(set)
    cat_videos: dict[int, set[int]] = defaultdict(set)
    cat_annotations: Counter[int] = Counter()
    for a in src.get("annotations", []):
        v = image_video.get(int(a.get("image_id", -1)))
        if v is None:
            continue
        c = int(a.get("category_id", -1))
        if c not in known:
            continue
        video_ann[v] += 1
        cat_videos[c].add(v)
        cat_annotations[c] += 1
        video_tracks[v].add((v, int(a.get("track_id", -1))))

    # Greedy coverage: rare/unmet categories dominate; ties prefer more
    # supported-known annotations, more tracks, then smaller video ID.  The
    # fill phase keeps multiple tracks and proposal evidence after coverage.
    target = {c: min(2, len(cat_videos[c] & set(legal))) for c in sorted(known)}
    seen: Counter[int] = Counter()
    bank: list[int] = []
    remaining = set(legal)
    while remaining and len(bank) < bank_max:
        def key(v: int):
            gain = sum(1 for c in known if v in cat_videos[c] and seen[c] < target[c])
            rare = sum(1.0 / max(len(cat_videos[c] & set(legal)), 1)
                       for c in known if v in cat_videos[c] and seen[c] < target[c])
            return (gain, rare, video_ann[v], len(video_tracks[v]), -v)
        best = max(remaining, key=key)
        gain = key(best)[0]
        if gain == 0 and len(bank) >= min(64, bank_max):
            # Continue with dense supported-known videos to make DSCT coverage
            # measurable, but no arbitrary random sampling is introduced.
            best = max(remaining, key=lambda v: (video_ann[v], len(video_tracks[v]), -v))
        bank.append(best)
        remaining.remove(best)
        for c in known:
            if best in cat_videos[c] and seen[c] < target[c]:
                seen[c] += 1

    # Held-out roles are selected from videos not in the bank.  First reserve
    # one per category where possible, then fill by deterministic density.
    def choose_role(available: set[int], n: int, avoid: set[int]) -> list[int]:
        chosen: list[int] = []
        used = set(avoid)
        for c in sorted(known):
            cand = sorted(v for v in available - used if v in cat_videos[c])
            if cand:
                # Prefer a video not already represented by a selected
                # category, then public annotation density, then ID.
                v = max(cand, key=lambda x: (video_ann[x], len(video_tracks[x]), -x))
                chosen.append(v); used.add(v)
                if len(chosen) >= n:
                    return chosen
        rest = sorted(available - used, key=lambda v: (video_ann[v], len(video_tracks[v]), -v), reverse=True)
        chosen.extend(rest[:max(0, n - len(chosen))])
        return chosen

    bank_set = set(bank)
    cal = choose_role(set(legal) - bank_set, role_max, set())
    audit = choose_role(set(legal) - bank_set - set(cal), role_max, set(cal))
    roles = {"known_bank_train": sorted(bank), "known_calibration": sorted(cal),
             "known_audit": sorted(audit),
             "novel_meta": sorted(map(int, frozen_manifest["split"]["meta_validation"]["videos"])),
             "devplus_evaluation": sorted(dev_videos), "q1_quarantined": sorted(Q1_VIDEOS)}
    role_sets = {k: set(v) for k, v in roles.items() if k.startswith("known_")}
    overlap = {f"{a}__{b}": sorted(role_sets[a] & role_sets[b])
               for a in role_sets for b in role_sets if a < b}
    cat_role: dict[str, dict[str, list[int]]] = {}
    for c in sorted(known):
        cat_role[str(c)] = {r: sorted(v for v in vals if v in cat_videos[c])
                            for r, vals in roles.items() if r.startswith("known_")}
    fallbacks = {str(c): {"bank_videos": len(cat_role[str(c)]["known_bank_train"]),
                           "calibration_videos": len(cat_role[str(c)]["known_calibration"]),
                           "audit_videos": len(cat_role[str(c)]["known_audit"]),
                           "fallback": ("unavailable" if not cat_role[str(c)]["known_calibration"]
                                        else "held_out_video")}
                 for c in sorted(known)}
    result = {
        "protocol": "trackocd_iclr27_phase15s16",
        "annotation": str(annotation.resolve()), "annotation_sha256": sha256(annotation),
        "preregistration": str(prereg.resolve()), "known_ids": sorted(known),
        "devplus_excluded": sorted(dev_videos), "q1_excluded": sorted(Q1_VIDEOS),
        "legal_public_train_videos": legal, "roles": roles, "role_overlap": overlap,
        "category_role_videos": cat_role, "fallbacks": fallbacks,
        "video_frames": {str(v): int(video_frames[v]) for v in legal},
        "video_known_annotations": {str(v): int(video_ann[v]) for v in legal},
        "category_annotation_counts": {str(c): int(cat_annotations[c]) for c in sorted(known)},
        "limits": {"bank_max_videos": bank_max, "calibration_max_videos": role_max,
                    "audit_max_videos": role_max, "selection": "greedy_rare_category_then_density"},
        "novel_meta_category_disjoint_role_retained": True,
        "known_bank_does_not_use_gt_boxes_as_final_input": True,
        "label_access": {"public_labels_for_alignment_and_calibration": True,
                          "devplus_labels_for_fit": False, "q1_labels": False}
    }
    atomic_json(out, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", default="data/iclr27_phase15s/sources/tao_train_annotations.json")
    ap.add_argument("--prereg", default="outputs/iclr27_phase15s/manifests/preregistration.json")
    ap.add_argument("--known", default="data/iclr27_phase15s/sources/supported_known_ids.json")
    ap.add_argument("--out", default="outputs/iclr27_phase15s/manifests/data_split_and_leakage_audit.json")
    ap.add_argument("--bank-max", type=int, default=320)
    ap.add_argument("--role-max", type=int, default=40)
    args = ap.parse_args()
    print(json.dumps(construct(ROOT / args.annotation, ROOT / args.prereg,
                               ROOT / args.known, ROOT / args.out,
                               args.bank_max, args.role_max), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
