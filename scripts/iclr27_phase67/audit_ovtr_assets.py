#!/usr/bin/env python3
"""Phase 67: read-only OVTR asset and lineage audit.

The project root is intentionally not a git worktree.  The OVTR mirror has a
recorded upstream commit but also contains historical TrackOCD edits; this
script records both facts instead of silently treating a dirty mirror as an
upstream checkout.  No model is modified and no sealed data is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
OVTR = ROOT / "third_party/research_refs_phase4n/OVTR"
OUT = ROOT / "outputs/iclr27_phase67"
DOC = ROOT / "docs/iclr27_phase67/PHASE67_OVTR_ASSET_LINEAGE_AUDIT.md"


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run(*args: str, cwd: Path = ROOT) -> str:
    try:
        return subprocess.check_output(args, cwd=str(cwd), text=True,
                                       stderr=subprocess.STDOUT).strip()
    except Exception as e:  # audit must retain missing metadata, not fail open
        return f"ERROR: {e}"


def ckpt_meta(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path),
    }
    if not path.exists():
        return out
    # Keep torch optional so the audit remains useful outside the OVTR env.
    try:
        import torch  # type: ignore
        ck = torch.load(str(path), map_location="cpu")
        out["top_level_keys"] = list(ck.keys()) if isinstance(ck, dict) else []
        model = ck.get("model", {}) if isinstance(ck, dict) else {}
        out["model_tensor_count"] = len(model) if isinstance(model, dict) else None
        if isinstance(ck, dict):
            out["epoch"] = ck.get("epoch")
            args = ck.get("args")
            if args is not None:
                d = vars(args) if hasattr(args, "__dict__") else (args if isinstance(args, dict) else {})
                keep = ["epochs", "max_train_iters", "config_file", "dataset_file",
                        "track_query_iteration", "tco_loss_coef", "dscq_loss_coef",
                        "dsct_coef", "start_epoch", "pretrain", "resume",
                        "score_mode"]
                out["args"] = {k: d.get(k) for k in keep if k in d}
    except Exception as e:
        out["torch_inspection_error"] = repr(e)
    return out


def method(name: str, repo: str, paper: str, commit: str, license_: str,
           release: str, task: str, inputs: str, outputs: str, causal: bool,
           persistent: bool, proposal: str, lifecycle: str, dependency: str,
           supervision: str, reuse: str, incompatible: str, selected: bool) -> Dict[str, Any]:
    return {
        "name": name, "repo_url": repo, "paper_url": paper, "commit_or_tag": commit,
        "license": license_, "release": release, "task_definition": task,
        "inputs": inputs, "outputs": outputs, "online_causal": causal,
        "persistent_query": persistent, "proposal_source": proposal,
        "physical_lifecycle": lifecycle, "forbidden_or_external_dependency": dependency,
        "training_supervision": supervision, "reusable_boundary": reuse,
        "incompatible_boundary": incompatible, "selected_for_phase68": selected,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=str(OUT / "audit/ovtr_assets.json"))
    ap.add_argument("--out-report", default=str(DOC))
    args = ap.parse_args()
    OUT.joinpath("audit").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("completion").mkdir(parents=True, exist_ok=True)

    commit = run("git", "rev-parse", "HEAD", cwd=OVTR)
    status = run("git", "status", "--short", cwd=OVTR)
    remote = run("git", "remote", "get-url", "origin", cwd=OVTR)
    lic = OVTR / "LICENSE"
    readme = OVTR / "README.md"
    assets = [
        {
            "name": "ovtr_det_pretrain_official",
            "role": "official OVTR detection pretraining checkpoint",
            "official_download_url": "https://drive.google.com/file/d/1x5RQ5m6XlLYB_iOPDnbeEKSYQeT4HVwo/view?usp=sharing",
            "local_path": str(ROOT / "checkpoints/ovtr/ovtr_det_pretrain.pth"),
            "source_claim": "OVTR README model-zoo link; not a TrackOCD-held/test artifact",
        },
        {
            "name": "ovtr_5_frame_official",
            "role": "official OVTR 5-frame TAO checkpoint",
            "official_download_url": "https://drive.google.com/file/d/10GKAIBxAseTiXnJXV1MnxnJBTmOHVFh5/view?usp=sharing",
            "local_path": str(ROOT / "checkpoints/ovtr/ovtr_5_frame.pth"),
            "source_claim": "OVTR README evaluation model-zoo link; used as frozen baseline candidate",
        },
        {
            "name": "ovtr_p0_smoke_local",
            "role": "TrackOCD local 20-iteration smoke initialized from OVTR detection checkpoint",
            "official_download_url": None,
            "local_path": str(ROOT / "outputs/iclr27_phase4p/ovtr_training_recovery/smoke/checkpoint.pth"),
            "source_claim": "local Phase4P smoke; not official",
        },
        {
            "name": "q0_long_local",
            "role": "TrackOCD Q0 OVTR baseline, 7 completed epochs (15k iter/epoch schedule)",
            "official_download_url": None,
            "local_path": str(ROOT / "outputs/iclr27_phase4q/q0_long/checkpoint.pth"),
            "source_claim": "local continuation from Phase4P OVTR P0; no TCO/DSCQ losses",
        },
        {
            "name": "q1_long_local",
            "role": "TrackOCD Q1 OVTR plus TCO, 7 completed epochs",
            "official_download_url": None,
            "local_path": str(ROOT / "outputs/iclr27_phase4q/q1_long/checkpoint.pth"),
            "source_claim": "local continuation; TCO branch enabled",
        },
        {
            "name": "q2_long_local",
            "role": "TrackOCD Q2 OVTR plus DSCQ, 7 completed epochs",
            "official_download_url": None,
            "local_path": str(ROOT / "outputs/iclr27_phase4q/q2_long/checkpoint.pth"),
            "source_claim": "local continuation; dual-state branch enabled; known proposal suppression",
        },
    ]
    for a in assets:
        a.update(ckpt_meta(Path(a["local_path"])))

    methods = [
        method("OVTR", "https://github.com/jinyanglii/OVTR", "https://arxiv.org/abs/2503.10616",
               commit, "MIT", "ICLR 2025", "end-to-end open-vocabulary MOT on TAO",
               "raw frames plus Deformable-DETR dual branch and CLIP embeddings",
               "tracked queries, boxes, scores and base/novel category logits", True, True,
               "dual-branch Deformable-DETR decoder", "MOTR-style persistent query tracking",
               "CLIP text/image/category embeddings and LVIS vocabulary",
               "LVIS/TAO detection and trajectory labels", "persistent-query decoder and causal lifecycle implementation",
               "text/category head and no prior-video semantic support; public code mirror is locally dirty",
               True),
        method("MOTIP-2", "https://github.com/GISer-WB/MOTIP-2", "https://arxiv.org/abs/2403.16848",
               "012856c1dc13b324064e79339ae71054518d1b5e", "Apache-2.0", "CVPR 2025",
               "online MOT formulated as in-context physical ID prediction",
               "frames and historical trajectory/query embeddings", "boxes and physical ID predictions", True, True,
               "Deformable/DAB-DETR", "trajectory prompt and physical-ID lifecycle",
               "physical ID labels are in-context prompts", "MOT17/DanceTrack/SportsMOT boxes and IDs",
               "query-memory association ideas", "physical-ID shortcut and no cross-video semantic state", False),
        method("ObjectRelator", "https://github.com/insait-institute/ObjectRelator", "https://arxiv.org/abs/2411.19083",
               "59f79d5d0fa5cfc7169b6737fd414c25d1ed83a6", "Apache-2.0", "ICCV 2025 Highlight",
               "ego/exo cross-view object relation and segmentation", "paired views, masks and optional language",
               "cross-view masks/relations", False, False, "paired-view segmentation proposals", "none",
               "MCFuse text modality and paired-view data", "Ego-Exo4D/HANDAL-X masks",
               "object-level consistency loss concept", "static paired views, language and no MOT lifecycle", False),
        method("C3Po", "https://github.com/c3po-correspondence/C3Po", "https://arxiv.org/abs/2409.13684",
               "21254a078435451e99d2feabd5db9334c02d8483", "MIT (repository evidence)", "NeurIPS 2025",
               "cross-view cross-modality correspondence by pointmap prediction",
               "paired images/pointmaps", "correspondence pointmaps", False, False,
               "none", "none", "DUSt3R/pointmap and paired-view assumptions", "paired-view correspondence supervision",
               "pointmap correspondence objective", "not causal MOT and no lifecycle/Commit interface", False),
        method("MASA", "https://github.com/siyuanliii/masa", "https://arxiv.org/abs/2401.13613",
               "c5472b9c7615f35abdf1188cb1a0c5408fe50d66", "Apache-2.0", "CVPR 2024 Highlight",
               "matching anything by segmenting anything", "external detector/segmenter proposals and image crops",
               "association adapter scores/masks", False, False, "external detector/SAM/Detic/GDINO",
               "adapter-based association", "open-vocabulary detector dependencies in variants",
               "association/video pairs", "appearance adapter only", "does not provide complete source or causal semantic controller", False),
        method("MeMOTR", "https://github.com/MCG-NJU/MeMOTR", "https://arxiv.org/abs/2203.16760",
               "eb7a177b9cbcb89742ec69b2545ab3af2ea31a80", "Apache-2.0 (repository evidence)", "ICCV 2023",
               "end-to-end online MOT with long-term memory", "frames and persistent queries", "boxes and physical IDs",
               True, True, "DETR-style detector", "explicit query memory/lifecycle",
               "physical ID tracking supervision", "MOT trajectory boxes and IDs", "memory/lifecycle pattern",
               "no open-world cross-video semantic state", False),
        method("MOTR", "https://github.com/megvii-research/MOTR", "https://arxiv.org/abs/2105.03247",
               "8690da3392159635ca37c31975126acf40220724", "Apache-2.0 (repository evidence)", "ECCV 2022",
               "end-to-end multi-object tracking", "frames and track queries", "boxes and physical IDs", True, True,
               "Deformable-DETR", "track query birth/continue/terminate", "physical IDs in supervision",
               "MOT trajectory labels", "query lifecycle reference", "closed-set physical tracking only", False),
    ]

    repo_info = {
        "phase": 67,
        "root": str(ROOT),
        "project_git_worktree": False,
        "ovtr_repo": {
            "path": str(OVTR), "origin": remote, "head": commit,
            "worktree_status": status.splitlines() if status else [],
            "license_sha256": sha256(lic), "readme_sha256": sha256(readme),
            "note": "HEAD is upstream commit recorded in prior audit; local Phase4P/4Q edits are retained and are not claimed upstream.",
        },
        "assets": assets,
        "methods": methods,
        "historical_evidence": {
            "phase4p": {"q0_tao_val_teta_combined": 24.14, "q1_tco": 24.67,
                         "p2_tco_novel_recall_dev_at_1fp": 0.233,
                         "score_bug_fixed": True},
            "phase4q": {"q0_teta_combined": 25.843, "q1_teta_combined": 27.58,
                         "q2_teta_combined": 18.83, "epochs_completed": 7,
                         "iters_per_epoch": 15000},
        },
        "phase68_selection": {
            "baseline_checkpoint": str(ROOT / "outputs/iclr27_phase4q/q0_long/checkpoint.pth"),
            "fallback_official_checkpoint": str(ROOT / "checkpoints/ovtr/ovtr_5_frame.pth"),
            "reason": "Q0 is the historical full-stream base OVTR run with score-corrected TAO output; the official 5-frame checkpoint remains a frozen fallback. Q1/Q2 semantic branches are not used as the class-agnostic physical baseline.",
            "text_category_inputs": "disabled for TrackOCD baseline; any OVTR CLIP/category output is isolated from physical track state",
        },
        "sealed_or_public_test_access": False,
        "public_q1_devplus_access": False,
    }
    tmp = Path(args.out_json).with_suffix(".tmp")
    tmp.write_text(json.dumps(repo_info, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, args.out_json)

    lines: List[str] = []
    lines += ["# Phase67 OVTR Asset Lineage Audit", "", "状态：COMPLETE（只读；未修改既有 OVTR 文件，未访问 sealed/public labels）。", ""]
    lines += ["## 结论", "", "已有 OVTR 资产优先于 Phase57–61 的从零 RGB detector。Phase68 选用经过 Phase4Q 七个完整 epoch 的 Q0 基础 OVTR checkpoint 作为历史 full-stream baseline；官方 `ovtr_5_frame.pth` 保留为冻结 fallback。Q1/Q2 的 TCO/DSCQ 分支只作为历史对照，不作为 class-agnostic MOT 主输入。", ""]
    lines += ["## 官方来源与代码边界", "", f"- OVTR repo: https://github.com/jinyanglii/OVTR；本地 origin `{remote}`，HEAD `{commit}`，MIT (`LICENSE` SHA256 `{sha256(lic)}`)。", "- 本地 mirror 有 Phase4P/4Q 修改（`git status --short` 非空），因此报告区分 upstream commit 与 TrackOCD lineage。", "- OVTR 官方任务是 TAO open-vocabulary MOT，依赖 CLIP image/text/category embeddings；Phase68 的 class-agnostic physical path 禁止向 semantic/physical state 传递这些类别文本/logits。", "- 历史 score 修复：`ovtr/eval.py:update_results_teta` 已将真实 detection score 写入第 6 列；Phase4P/Q 输出沿用修复后路径。", ""]
    lines += ["## Checkpoint lineage", "", "| asset | role | bytes | SHA256 | lineage |", "|---|---|---:|---|---|"]
    for a in assets:
        lineage = "official" if "official" in a["name"] else "TrackOCD local"
        lines.append(f"| `{a['name']}` | {a['role']} | {a.get('bytes')} | `{a.get('sha256')}` | {lineage} |")
    lines += ["", "Q0/Q1/Q2 checkpoint metadata (epoch=7; configured 8 epochs, 15,000 updates/epoch) and args are recorded in `outputs/iclr27_phase67/audit/ovtr_assets.json`. Q0 has `tco_loss_coef=0`, `dscq_loss_coef=0`; Q1 enables TCO; Q2 enables DSCQ and is known to suppress proposals.", ""]
    lines += ["## Method audit summary", "", "| method | causal/persistent | useful boundary | incompatibility for TrackOCD | selected |", "|---|---|---|---|---|"]
    for m in methods:
        lines.append(f"| {m['name']} | {m['online_causal']}/{m['persistent_query']} | {m['reusable_boundary']} | {m['incompatible_boundary']} | {m['selected_for_phase68']} |")
    lines += ["", "详细字段（paper/repo/commit/license/输入输出/监督）在机器可读 JSON 中。ObjectRelator/C3Po 是 paired/static correspondence，不是 causal MOT；MOTIP-2/MeMOTR/MOTR 用 physical-ID 监督，不能把 ID 作为 TrackOCD semantic 输入；MASA 仍依赖外部 detector。", ""]
    lines += ["## Phase68 reproduction command", "", "```bash", "cd third_party/research_refs_phase4n/OVTR/ovtr", "CUDA_VISIBLE_DEVICES=4 /home/lwr/anaconda3/envs/ovtr/bin/python eval.py \\", "  --config_file ./config/ovtr_lite_train_val.py \\", "  --dataset_file lvis_generated_img_seqs --batch_size 1 \\", "  --with_box_refine --two_stage \\", f"  --pretrain {ROOT}/outputs/iclr27_phase4q/q0_long/checkpoint.pth \\", "  --score_mode base --num_workers 4 --sampler_lengths 2 \\", "  --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \\", "  --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \\", "  --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \\", f"  --result_path_track {ROOT}/outputs/iclr27_phase68/metrics/ovtr_baseline/teta_results", "```", "", "该命令仅使用 TAO validation annotations作为历史 full-stream comparator；Phase68 会额外建立 TrackEval-compatible exporter 与原始 76-event alignment。不要将 TAO test 或 Q1 labels 写入 outputs。", ""]
    lines += ["## Resource/sealed boundary", "", "审计阶段无训练进程。开始 Phase68 前再次记录 `free -h`、`nvidia-smi`、进程数和磁盘；只使用空闲 GPU，最多四个有界 worker。Phase68/69 的每单元 `.launched/.done` 与 checkpoint 使用原子写入；不使用 broad kill。sealed/public/Q1 labels 未访问。", ""]
    Path(args.out_report).write_text("\n".join(lines) + "\n")
    marker = OUT / "completion/phase67_audit.done"
    marker.write_text("phase=67\nstatus=complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
