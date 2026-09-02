"""Canonical asset manifests for Q0 validation and Phase19R TRAIN events."""
from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


def norm_rel(value: str) -> str:
    value = str(value or "").replace("\\", "/")
    return re.sub(r"/+/", "/", value).lstrip("./")


def canonical_video_key(dataset: str, split: str, video_name: str) -> str:
    return f"{dataset.lower()}|{split.lower()}|{norm_rel(video_name)}"


def canonical_image_key(video_key: str, frame_index: Any, image_name: str = "") -> str:
    try: frame = int(frame_index)
    except (TypeError, ValueError): frame = None
    suffix = f"frame={frame}" if frame is not None else f"file={norm_rel(image_name)}"
    return f"{video_key}|{suffix}"


def _video_index(annotation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(v["id"]): v for v in annotation.get("videos", [])}


def _image_index(annotation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(i["id"]): i for i in annotation.get("images", [])}


def _resolve_image(file_name: str, roots: Iterable[Path]) -> tuple[str | None, bool]:
    name = norm_rel(file_name)
    for root in roots:
        p = Path(root) / name
        if p.exists(): return str(p.resolve()), True
    return str((Path(next(iter(roots), ".") ) / name).resolve()), False


def build_annotation_assets(annotation_path: Path, split: str, roots: Iterable[Path], schema: str, *, image_ids: set[int] | None = None) -> list[dict[str, Any]]:
    import json
    from .io import sha256
    ann = json.loads(Path(annotation_path).read_text())
    videos = _video_index(ann); images = _image_index(ann); roots = list(roots); ah = sha256(annotation_path)
    out = []
    for image in ann.get("images", []):
        iid = int(image.get("id", -1))
        if image_ids is not None and iid not in image_ids: continue
        vid = int(image.get("video_id", image.get("video", -1))) if str(image.get("video_id", image.get("video", ""))).lstrip("-").isdigit() else None
        if vid is None:
            # Annotation image.video is a stable dataset-native video path.
            vp = str(image.get("video", "")); vid = next((k for k, v in videos.items() if str(v.get("name", "")) == vp), -1)
        video = videos.get(vid, {})
        vname = str(video.get("name", image.get("video", ""))); dataset = str(video.get("metadata", {}).get("dataset", vname.split("/")[1] if "/" in vname else "unknown"))
        file_name = str(image.get("file_name", "")); vk = canonical_video_key(dataset, split, vname); ik = canonical_image_key(vk, image.get("frame_index"), file_name)
        resolved, exists = _resolve_image(file_name, roots)
        out.append({"asset_schema": schema, "dataset_name": dataset, "dataset_split": split, "annotation_path": str(Path(annotation_path).resolve()), "annotation_sha256": ah,
                    "video_id": vid, "image_id": iid, "video_file_name": vname, "image_file_name": file_name, "canonical_video_key": vk, "canonical_image_key": ik,
                    "frame_index": image.get("frame_index"), "width": image.get("width", video.get("width")), "height": image.get("height", video.get("height")),
                    "rgb_root": str(roots[0].resolve()) if roots else None, "resolved_path": resolved, "path_exists": exists})
    return out


def event_assets_from_rows(annotation_path: Path, rows: Iterable[dict[str, Any]], roots: Iterable[Path]) -> list[dict[str, Any]]:
    import json
    from .io import sha256
    ann = json.loads(Path(annotation_path).read_text()); videos = _video_index(ann); images = _image_index(ann); roots = list(roots); ah = sha256(annotation_path); out = OrderedDict()
    for row in rows:
        try: iid = int(row["image_id"])
        except (KeyError, TypeError, ValueError): continue
        if iid in out: continue
        image = images.get(iid, {}); vid = int(row.get("video_id", image.get("video_id", -1))); video = videos.get(vid, {}); vname = str(video.get("name", image.get("video", f"video/{vid}"))); dataset = str(video.get("metadata", {}).get("dataset", vname.split("/")[1] if "/" in vname else "unknown")); file_name = str(row.get("image_path", image.get("file_name", ""))); vk = canonical_video_key(dataset, "train", vname); ik = canonical_image_key(vk, row.get("source_frame_index", image.get("frame_index")), file_name); resolved, exists = _resolve_image(file_name, roots)
        out[iid] = {"asset_schema": "phase74.phase19r_asset.v1", "event_video_id": vid, "event_image_id": iid, "source_annotation_path": str(Path(annotation_path).resolve()), "source_annotation_sha256": ah,
                    "dataset_name": dataset, "dataset_split": "train", "video_file_name": vname, "image_file_name": file_name, "canonical_video_key": vk, "canonical_image_key": ik,
                    "frame_index": row.get("source_frame_index", image.get("frame_index")), "width": row.get("image_width", image.get("width", video.get("width"))), "height": row.get("image_height", image.get("height", video.get("height"))), "rgb_root": str(roots[0].resolve()) if roots else None, "resolved_path": resolved, "path_exists": exists}
    return list(out.values())


def mapping_record(event_asset: dict[str, Any], q0_by_key: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    q = (q0_by_key or {}).get(event_asset.get("canonical_image_key"));
    if q is None: return None
    return {"phase19r_canonical_image_key": event_asset["canonical_image_key"], "q0_canonical_image_key": q["canonical_image_key"], "phase19r_image_id": event_asset.get("event_image_id"), "q0_image_id": q.get("image_id"), "mapping_method": "EXACT_CANONICAL_PATH", "evidence": ["canonical_video_key+frame_index"], "one_to_one": True, "category_used": False, "bbox_used": False, "track_id_used": False}
