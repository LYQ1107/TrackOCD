"""Build the preregistered mixed Known+Novel TAO TRAIN evaluation view.

The model-facing annotation intentionally has no annotations/tracks.  GT
labels live only in the evaluator sidecar.  This makes the no-GT-to-model
contract auditable rather than relying on a claim about an evaluator flag.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", default="data/iclr27_phase14c/sources/tao_train_annotations.json")
    ap.add_argument("--split", default="outputs/iclr27_phase14b/manifests/devplus_split.json")
    ap.add_argument("--known-ids", default="data/trackocd_v1/pure/splits/supported_known_ids.json")
    ap.add_argument("--out", default="outputs/iclr27_phase14c")
    ap.add_argument("--model-ann", default="data/iclr27_phase14c/manifests/phase14c_validation_train.json")
    args = ap.parse_args()
    out = ROOT / args.out
    ann = json.loads((ROOT / args.annotation).read_text())
    prior = json.loads((ROOT / args.split).read_text())
    known_ids = {int(x) for x in json.loads((ROOT / args.known_ids).read_text())}
    novel_ids = {int(x) for x in prior["selected_devplus_categories"]}
    dev_videos = sorted(int(x) for x in prior["devplus_videos"])
    dev_set = set(dev_videos)

    images_by_video: dict[int, list[dict]] = defaultdict(list)
    image_by_id: dict[int, dict] = {}
    for im in ann["images"]:
        image_by_id[int(im["id"])] = im
        if int(im["video_id"]) in dev_set:
            images_by_video[int(im["video_id"])].append(im)
    for vid in images_by_video:
        images_by_video[vid].sort(key=lambda x: (int(x.get("frame_index", 0)), int(x["id"])))

    tracks_by_video: dict[int, list[dict]] = defaultdict(list)
    for tr in ann["tracks"]:
        vid, cat = int(tr["video_id"]), int(tr["category_id"])
        if vid in dev_set and (cat in known_ids or cat in novel_ids):
            tracks_by_video[vid].append(tr)

    anns_by_track: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for a in ann["annotations"]:
        key = (int(a["video_id"]), int(a["track_id"]))
        if key in {(int(t["video_id"]), int(t["id"])) for ts in tracks_by_video.values() for t in ts}:
            anns_by_track[key].append(a)

    # Keep the evaluator sidecar labels complete and chronological.
    gt_path = out / "manifests" / "mixed_gt_tracks.jsonl"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid in dev_videos:
        for tr in sorted(tracks_by_video.get(vid, []), key=lambda x: int(x["id"])):
            cat = int(tr["category_id"])
            boxes, image_ids, frame_indices, paths = [], [], [], []
            for a in sorted(anns_by_track[(vid, int(tr["id"]))],
                            key=lambda x: int(image_by_id[int(x["image_id"])].get("frame_index", 0))):
                im = image_by_id[int(a["image_id"])]
                b = a["bbox"]
                boxes.append([float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3])])
                image_ids.append(int(im["id"]))
                frame_indices.append(int(im.get("frame_index", 0)))
                paths.append(str(im["file_name"]))
            if not boxes:
                continue
            rows.append({
                "sample_id": f"{vid}_{int(tr['id'])}",
                "video_id": vid,
                "track_id": int(tr["id"]),
                "category_id": cat,
                "role": "novel" if cat in novel_ids else "supported_known",
                "image_ids": image_ids,
                "frame_indices": frame_indices,
                "boxes_xyxy": boxes,
                "image_paths": paths,
                "category_label_access": "evaluator_only",
                "private_gt_used": False,
                "q1_label_used": False,
                "future_frames_used": False,
                "physical_id_used_as_feature": False,
            })
    with gt_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    # Model-facing annotation: selected images/videos only, no GT instances.
    videos_by_id = {int(v["id"]): v for v in ann["videos"]}
    model_images = []
    for vid in dev_videos:
        for frame_id, im in enumerate(images_by_video[vid]):
            x = {k: v for k, v in im.items() if k not in {"frame_id"}}
            x["frame_id"] = frame_id
            x["frame_index"] = int(im.get("frame_index", frame_id))
            model_images.append(x)
    model_videos = [videos_by_id[v] for v in dev_videos]
    model_ann = {
        "info": ann.get("info", {}),
        "licenses": ann.get("licenses", []),
        "categories": ann.get("categories", []),
        "videos": model_videos,
        "images": model_images,
        "tracks": [],
        "annotations": [],
    }
    atomic_json(ROOT / args.model_ann, model_ann)

    role_tracks = Counter(r["role"] for r in rows)
    role_cats = {role: sorted({r["category_id"] for r in rows if r["role"] == role})
                 for role in ("supported_known", "novel")}
    image_paths = [im["file_name"] for im in model_images]
    assert len(image_paths) == len(set(image_paths))
    summary = {
        "protocol": "phase14c",
        "source_annotation": str((ROOT / args.annotation).resolve()),
        "source_frames": str((ROOT / "data/iclr27_phase14c/sources/tao_train_frames").resolve()),
        "selected_novel_categories": sorted(novel_ids),
        "selected_videos": dev_videos,
        "known_ids_source": str((ROOT / args.known_ids).resolve()),
        "known_population_extension": None,
        "tracks_by_role": dict(role_tracks),
        "categories_by_role": {k: len(v) for k, v in role_cats.items()},
        "category_ids_by_role": role_cats,
        "videos": len(dev_videos),
        "images": len(model_images),
        "model_annotation": str((ROOT / args.model_ann).resolve()),
        "evaluator_gt_tracks": str(gt_path.resolve()),
        "model_input_has_annotations": False,
        "model_input_has_tracks": False,
        "frame_id_reset_per_video": True,
        "future_frames_used": False,
        "q1_label_used": False,
        "private_gt_used": False,
    }
    atomic_json(out / "manifests" / "mixed_eval_split.json", summary)
    atomic_json(out / "eval" / "mixed_population_audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
