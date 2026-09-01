#!/usr/bin/env python3
"""Read-only Phase57 audit of official method repositories."""
from __future__ import annotations
import hashlib, json, os, subprocess, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase57/audit/github_methods.json"
REPORT = ROOT / "docs/iclr27_phase57/PHASE57_OFFICIAL_METHOD_AUDIT.md"

REPOS = [
  dict(name="OVTR", repo_url="https://github.com/jinyanglii/OVTR", git_url="https://github.com/jinyanglii/OVTR.git", paper_url="https://arxiv.org/abs/2503.10616", release="ICLR 2025", license="MIT", task="end-to-end open-vocabulary MOT on TAO", inputs="frames, Deformable-DETR features, CLIP image/text embeddings", outputs="detections, tracked queries, base/novel category predictions", causal_online=True, persistent_query=True, proposal="dual-branch Deformable-DETR decoder", lifecycle="MOTR-style query tracking", dependency="CLIP text/category vocabulary", supervision="LVIS/TAO detection/tracking", reuse="persistent query decoder reference", incompatible="text/category head; no prior-video semantic state", command="cd ovtr && sh tools/ovtr_multi_frame_lite_train.sh", gpu="README speed test: one RTX3090; CUDA deformable ops", selected=False),
  dict(name="MOTIP-2", repo_url="https://github.com/GISer-WB/MOTIP-2", git_url="https://github.com/GISer-WB/MOTIP-2.git", paper_url="https://arxiv.org/abs/2403.16848", release="CVPR 2025", license="Apache-2.0", task="online MOT as in-context ID prediction", inputs="frames and historical trajectory/query embeddings", outputs="boxes and physical IDs", causal_online=True, persistent_query="historical trajectory prompts", proposal="Deformable/DAB-DETR", lifecycle="physical ID prediction", dependency="physical ID labels are prompts", supervision="DanceTrack/SportsMOT/MOT17 boxes and IDs", reuse="query-memory association pattern", incompatible="ID shortcut; no cross-video semantic state", command="see docs/GET_STARTED.md and configs", gpu="multi-GPU PyTorch configs", selected=False),
  dict(name="ObjectRelator", repo_url="https://github.com/insait-institute/ObjectRelator", git_url="https://github.com/insait-institute/ObjectRelator.git", paper_url="https://arxiv.org/abs/2411.19083", release="ICCV 2025 Highlight", license="Apache-2.0", task="ego/exo cross-view object relation and segmentation", inputs="paired ego/exo images/video, masks, optional language cues", outputs="cross-view object masks/relations", causal_online=False, persistent_query=False, proposal="paired-view segmentation proposals", lifecycle="none", dependency="README explicitly describes text modality in MCFuse", supervision="Ego-Exo4D/HANDAL-X paired masks", reuse="object consistency loss concept", incompatible="static paired-view; text; no MOT lifecycle", command="see docs/Train_Evaluation.md", gpu="Ego-Exo4D assets required", selected=False),
  dict(name="C3Po", repo_url="https://github.com/c3po-correspondence/C3Po", git_url="https://github.com/c3po-correspondence/C3Po.git", paper_url="https://arxiv.org/abs/2511.18559", release="NeurIPS 2025", license="CC BY-NC-SA 4.0 for bundled DUSt3R code", task="cross-view/cross-modality pointmap correspondence", inputs="paired images and DUSt3R visual/geometric data", outputs="dense correspondences and optional pose", causal_online=False, persistent_query=False, proposal="DUSt3R pointmap", lifecycle="none", dependency="no category text in stated task", supervision="C3 paired image/geometric/visual data", reuse="correspondence loss/evaluation idea", incompatible="static pairs; no causal tracks/Commit", command="torchrun --nproc_per_node 8 train.py ...", gpu="official command requests 8 processes and ViT-Large", selected=False),
  dict(name="MASA", repo_url="https://github.com/siyuanliii/masa", git_url="https://github.com/siyuanliii/masa.git", paper_url="https://arxiv.org/abs/2406.04221", release="CVPR 2024 Highlight", license="Apache-2.0", task="universal instance appearance matching", inputs="detector/segmentation proposals and crops", outputs="instance embeddings/association scores", causal_online="adapter can run online", persistent_query=False, proposal="external detector/SAM/Detic/GDINO", lifecycle="tracker integration, not end-to-end", dependency="some variants use open-vocabulary detectors", supervision="unlabeled images with SAM regions/transforms", reuse="appearance association baseline", incompatible="does not generate complete source or semantic controller", command="see docs/train.md", gpu="R50 lighter; SAM/GDINO heavier", selected=False),
  dict(name="MeMOTR", repo_url="https://github.com/MCG-NJU/MeMOTR", git_url="https://github.com/MCG-NJU/MeMOTR.git", paper_url="https://arxiv.org/abs/2307.00656", release="2023", license="MIT", task="memory-based end-to-end MOT", inputs="frames and track queries", outputs="physical tracks", causal_online=True, persistent_query=True, proposal="DETR-style detector", lifecycle="explicit MOT memory/lifecycle", dependency="physical IDs in supervision", supervision="MOT-style boxes/IDs", reuse="memory/lifecycle pattern", incompatible="no unknown semantic correspondence", command="see repository README", gpu="multi-GPU DETR; local reproduction required", selected=False),
  dict(name="MOTR", repo_url="https://github.com/megvii-research/MOTR", git_url="https://github.com/megvii-research/MOTR.git", paper_url="https://arxiv.org/abs/2105.03247", release="2021", license="MIT", task="end-to-end transformer MOT", inputs="frames and persistent track queries", outputs="physical tracks", causal_online=True, persistent_query=True, proposal="Deformable-DETR", lifecycle="track query birth/continue/terminate", dependency="physical IDs in supervision", supervision="MOTChallenge boxes/IDs", reuse="causal query abstraction", incompatible="no semantic support/Commit", command="see repository README", gpu="older DETR dependencies", selected=False),
]

def fetch(url, limit=7000):
  try:
    with urllib.request.urlopen(url, timeout=20) as f: b = f.read(limit)
    return {"ok": True, "sha256": hashlib.sha256(b).hexdigest(), "text": b.decode("utf8", "replace")}
  except Exception as e: return {"ok": False, "error": repr(e), "sha256": None, "text": ""}

def head(url):
  try:
    p = subprocess.run(["git", "ls-remote", url, "HEAD"], capture_output=True, text=True, timeout=30)
    ls = p.stdout.strip().split()
    return {"returncode": p.returncode, "head": ls[0] if ls else None, "stderr": p.stderr.strip()}
  except Exception as e: return {"returncode": -1, "head": None, "stderr": repr(e)}

def main():
  recs=[]
  for item in REPOS:
    r=dict(item); r["remote"]=head(item["git_url"])
    base=item["repo_url"].rstrip("/")
    r["license_evidence"]=fetch(base+"/raw/HEAD/LICENSE",3000)
    r["readme_evidence"]=fetch(base+"/raw/HEAD/README.md",7000)
    recs.append(r)
  payload={"phase":57,"network_audit":True,"weights_downloaded":False,"sealed_or_public_test_access":False,"methods":recs}
  OUT.parent.mkdir(parents=True,exist_ok=True); tmp=OUT.with_suffix(".tmp"); tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n"); os.replace(tmp,OUT)
  lines=["# Phase57 Official Method Audit","","Read-only audit of the official repositories listed in the Phase57 authorization. No external code/checkpoint or sealed label was downloaded. Exact URLs, bounded README/LICENSE evidence and remote HEAD hashes are in [`github_methods.json`](../../outputs/iclr27_phase57/audit/github_methods.json).","","## Method boundaries","","| Method | Online/causal | Persistent query | Proposal/lifecycle | Forbidden dependency or gap | Decision |","|---|---|---|---|---|---|"]
  for r in recs: lines.append(f"| {r['name']} | {r['causal_online']} | {r['persistent_query']} | {r['proposal']} / {r['lifecycle']} | {r['dependency']}; {r['incompatible']} | not selected |")
  lines += ["","## Audit conclusions","","OVTR (ICLR 2025) is the closest persistent-query reference, but its released open-vocabulary path requires CLIP text/category embeddings and has no prior-video semantic state; it is not imported as a TrackOCD solution. MOTIP-2, MeMOTR and MOTR provide physical query/lifecycle patterns but use physical-ID supervision. ObjectRelator (official organisation URL) and C3Po address paired/static cross-view correspondence rather than causal MOT; ObjectRelator includes a text modality and C3Po's official command asks for eight processes/DUSt3R ViT-Large. MASA is an appearance adapter whose proposal source remains external. The Phase59 prototype therefore uses a local class-agnostic pixel model with no text/category/ID input, while these repositories remain references only.","","## Reproduction","","`python scripts/iclr27_phase57/audit_official_methods.py` performs only `git ls-remote` and bounded raw requests and writes JSON atomically. The project root is not a Git worktree; remote HEAD hashes and artifact hashes are the revision record.","","Decision: `P57_AUDIT_COMPLETE_NO_EXTERNAL_METHOD_SELECTED`."]
  REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text("\n".join(lines)+"\n")

if __name__ == "__main__": main()
