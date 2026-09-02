#!/usr/bin/env python3
"""Audit two official modern trajectory-representation references.

This deliberately does not download weights or run inference.  It records
whether each public implementation can satisfy the registered TrackOCD
category-free, causal, cross-video contract with the available resources.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase80d/audit"


def run(*args: str) -> str:
    return subprocess.run(args, text=True, capture_output=True, check=False).stdout.strip()


def main() -> None:
    methods = [
        {
            "name": "TrajViT",
            "repo": "https://github.com/RAIVNLab/trajvit",
            "paper": "https://arxiv.org/abs/2505.23617",
            "commit": run("git", "ls-remote", "https://github.com/RAIVNLab/trajvit.git", "refs/heads/main").split()[0],
            "release": "ICCV 2025",
            "license": "No LICENSE file exposed in the official main-tree listing or raw main path at audit time; reuse would require license clarification.",
            "inputs": "video frames plus panoptic sub-object trajectories; released pretraining metadata includes image/video captions",
            "outputs": "one trajectory token per panoptic sub-object and a video transformer representation",
            "online_causal": "trajectory generation consumes ordered frames, but the released representation/pretraining contract is not TrackOCD's prior-only support stream",
            "persistent_query": False,
            "cross_video_correspondence": False,
            "unknown_novel": "not an open-world semantic correspondence evaluator",
            "text_category_id_dependency": "captions are part of the released image/video-text pretraining metadata; panoptic trajectory generation also relies on an external SAM2 checkpoint",
            "resource": "official pretraining command assumes 8 GPUs; released model checkpoint is external Drive",
            "decision": "AUDIT_ONLY_NOT_EXECUTED",
            "reason": "Promising trajectory-token representation, but no direct category-free cross-video correspondence head, unclear code license, 8-GPU pretraining assumption and external SAM2/metadata dependencies make a 4-GPU TrackOCD run non-reproducible within this window.",
        },
        {
            "name": "Trace Anything",
            "repo": "https://github.com/ByteDance-Seed/TraceAnything",
            "paper": "https://arxiv.org/abs/2510.13802",
            "commit": run("git", "ls-remote", "https://github.com/ByteDance-Seed/TraceAnything.git", "refs/heads/main").split()[0],
            "release": "ICLR 2026",
            "license": "Apache-2.0 code; CC BY-NC 4.0 model weights",
            "inputs": "ordered RGB image/video frames",
            "outputs": "4D trajectory-field control points, confidence maps, foreground mask and estimated time",
            "online_causal": "supports ordered frames but processes a scene jointly and is not a causal physical-track query/association module",
            "persistent_query": False,
            "cross_video_correspondence": "pixel trajectory field, not semantic category correspondence",
            "unknown_novel": "not an open-world semantic Commit/Defer module",
            "text_category_id_dependency": "no text/category input documented; foreground masks are geometric diagnostics",
            "resource": "official README reports examples on one GPU with >=48 GB VRAM; model weights are non-commercial",
            "decision": "AUDIT_ONLY_NOT_EXECUTED",
            "reason": "A useful temporal geometry reference, but its output is a 4D field rather than a track embedding and its scene-level inference is not the registered causal support contract; downloading a large non-commercial checkpoint would not answer the correspondence/controller question.",
        },
    ]
    obj = {
        "phase": "Phase80D",
        "route": "modern_visual_trajectory_reference_search",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "methods": methods,
        "selection": "No primary model was selected for execution: neither audited implementation provides a direct, licensed, 4-GPU, category-free causal cross-video TrackOCD correspondence interface.",
        "protocol": {"downloaded": False, "executed": False, "future_rows_or_tracks": False, "category_text_or_id_model_input": False, "sealed_or_public_new_accessed": False},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "modern_trajectory_audit.json"
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"); tmp.replace(path)
    done = OUT / "phase80d_modern.done"
    tmp = done.with_suffix(".tmp"); tmp.write_text(json.dumps({"phase": "Phase80D", "audit": str(path), "executed": False}, sort_keys=True), encoding="utf-8"); tmp.replace(done)
    print(json.dumps({"phase": "Phase80D", "methods": [m["name"] for m in methods], "output": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
