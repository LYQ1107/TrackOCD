#!/usr/bin/env python3
"""Record the eligible/non-eligible boundary of the local TRACT reference.

No model is loaded and no external data is downloaded.  This audit is kept as
an explicit artifact because TRACT's trajectory components are useful
references, while its released TraCLIP path uses category/text cues and its
MASA path expects external detector proposals.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "third_party/TRACT"
OUT = ROOT / "outputs/iclr27_phase80d/audit"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], text=True, capture_output=True, check=False).stdout.strip()


def main() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    clip_readme = (REPO / "TraCLIP/readme.md").read_text(encoding="utf-8")
    license_path = REPO / "masa/LICENSE"
    obj = {
        "phase": "Phase80D",
        "route": "TRACT_trajectory_reference_audit_only",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo": "https://github.com/Nathan-Li123/TRACT",
        "paper": "https://openaccess.thecvf.com/content/ICCV2025/html/Li_TRACT_ICCV2025_paper.html",
        "commit": git("rev-parse", "HEAD"),
        "commit_subject": git("log", "-1", "--format=%s"),
        "license": "Apache-2.0 (masa/LICENSE; TraCLIP license file not present in local checkout)",
        "license_sha256": sha(license_path),
        "release_year": 2025,
        "components": {
            "TCR": "trajectory consistency reinforcement",
            "TFA": "trajectory-aware feature aggregation",
            "TSE": "trajectory semantic enhancement",
            "MASA": "external proposal/instance association component",
            "TraCLIP": "CLIP-based trajectory classification path",
        },
        "causal_online": "trajectory context is causal in the reference design, but released scripts require detector tracks and do not implement TrackOCD's persistent semantic Commit/Defer evaluator",
        "persistent_query": False,
        "cross_video_correspondence": False,
        "unknown_novel": "open-vocabulary classification/association reference, not TrackOCD's category-free cross-video semantic state contract",
        "forbidden_dependency_evidence": {
            "clip_mentions": readme.lower().count("clip") + clip_readme.lower().count("clip"),
            "category_label_mentions": clip_readme.lower().count("category") + clip_readme.lower().count("cate"),
            "text_path": "TraCLIP/demo.py and video_classifier_02.py construct CLIP text prompts/class-name lists",
            "external_proposals": "README describes a replaceable open-vocabulary detector and MASA consumes detector boxes",
        },
        "trackocd_reusable": ["trajectory consistency/aggregation ideas as literature context"],
        "trackocd_not_reusable": ["TraCLIP text/category inference", "GT/ID-labelled tracklet extraction as inference", "external proposal dependency as a drop-in replacement", "no persistent semantic StateMemory/Commit-Defer implementation"],
        "executed": False,
        "downloaded": False,
        "decision": "REJECT_AS_PRIMARY_PHASE80_ROUTE",
        "reason": "No eligible pure-visual, category-free, persistent cross-video correspondence implementation is present in this checkout; running it would import text/category shortcuts or external proposals and would not answer the registered causal question.",
        "sealed_boundary": {"dev_q1_public_new_sealed_accessed": False, "future_rows_or_tracks": False, "physical_id_model_input": False},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tract_route_audit.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
    done = OUT / "phase80d.done"
    tmp = done.with_suffix(".tmp")
    tmp.write_text(json.dumps({"phase": "Phase80D", "decision": obj["decision"], "commit": obj["commit"]}, sort_keys=True), encoding="utf-8")
    tmp.replace(done)
    print(json.dumps({"phase": "Phase80D", "decision": obj["decision"], "commit": obj["commit"], "output": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
