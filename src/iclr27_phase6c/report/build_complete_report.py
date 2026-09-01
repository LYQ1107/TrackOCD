"""Build docs/iclr27_phase6c/PHASE6C_COMPLETE_COPYABLE_REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "docs" / "iclr27_phase6c" / "PHASE6C_COMPLETE_COPYABLE_REPORT.md"
EVAL = ROOT / "outputs" / "iclr27_phase6c" / "eval"


def fmt(v, nd=3):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def load_strict(name):
    p = EVAL / name / "strict" / "summary.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return d.get("strict", {}), d.get("legacy_first_frame", {})


def load_physical(name):
    p = EVAL / name / "physical.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_cal(name):
    p = EVAL / name / "calibration.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def run_table(names):
    rows = []
    for n in names:
        s, lf = load_strict(n)
        p = load_physical(n)
        c = load_cal(n)
        rows.append({
            "run": n,
            "known": s.get("known_occurrence_acc"),
            "birth": s.get("first_novel_birth_acc"),
            "reuse": s.get("novel_reuse_acc"),
            "cross": s.get("cross_physical_reuse_acc"),
            "born": s.get("n_born_novel_states"),
            "cnt_err": s.get("novel_count_abs_error"),
            "nmi": s.get("novel_nmi"),
            "ari": s.get("novel_ari"),
            "frag": s.get("mean_fragmentation"),
            "dup": s.get("duplicate_creation_rate"),
            "legacy_known": lf.get("overall_known_acc"),
            "legacy_route": lf.get("route_aware_novel_acc"),
            "legacy_cond": lf.get("conditional_novel_acc"),
            "legacy_all": lf.get("all_track_acc"),
            "tau": c.get("tau"),
            "n_tracks": p.get("n_tracks"),
            "n_aligned": p.get("n_aligned_tracks"),
        })
    return rows


def md_table(rows, cols):
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    return "\n".join([header, sep] + body)


def main():
    main_names = ["tse_main", "tse_v2"]
    v0_names = ["v0_traj_t045", "v0_traj_cal", "v0_frame_t045"]
    abl_names = ["tse_main_frame", "tse_abl_no_mnn", "tse_abl_no_frame",
                 "tse_abl_no_pres"]
    frontier_names = ["tse_main_kt0.80_nt0.55_mg0.20",
                      "tse_v2_kt0.80_nt0.55_mg0.20",
                      "tse_v2_kt0.85_nt0.60_mg0.20"]
    main_rows = run_table(main_names)
    v0_rows = run_table(v0_names)
    abl_rows = run_table(abl_names)
    frontier_rows = run_table(frontier_names)
    phase6b_s, phase6b_lf = load_strict("phase6b")

    md = []
    md.append("# TrackOCD ICLR 2027 — Phase 6C Complete Report")
    md.append("")
    md.append("## Open-World Trajectory Semantic Representation Reset")
    md.append("")
    md.append("Date: 2026-08-18. Project root: "
              "`/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT`.")
    md.append("")
    md.append("## 1. Final status")
    md.append("")
    md.append("**FINAL_STATUS: `TRAJECTORY_SEMANTIC_RECOVERY_PARTIAL` / "
              "`TRUE_OOV_SEMANTIC_SUPERVISION_REMAINS_LIMITING` / "
              "`TRACKOCD_NOT_YET_ICLR_LEVEL`**")
    md.append("")
    md.append("Phase 6C recovers known semantics (strict Known 0.671, legacy"
              " all-track 0.235 vs Phase 6B 0.0/0.031) and restores a usable"
              " novel memory (birth 0.222, count error 1, NMI 0.820), but"
              " correct cross-physical novel reuse remains 0 under the"
              " calibrated protocol; it only becomes >0 with a known-accuracy"
              " sacrificing policy (frontier diagnostics). The legal TRAIN"
              " unlabeled pool is too small/noisy for true-OOV cross-track"
              " category learning, so the overall system is not yet at ICLR"
              " level.")
    md.append("")
    md.append("## 2. Frozen conclusions from Phase 6B")
    md.append("")
    md.append("- Semantic mapping/evaluator/NEW path/novel-memory legality: no bug;")
    md.append("- JCDQ shared-hidden semantic formulation failed;")
    md.append("- DSCT physical/objectness is a positive signal and is preserved"
              " unchanged in Phase 6C;")
    md.append("- Strict causal protocol stays frozen (immediate decision, no"
              " future, no retroactive relabel, Physical ID != Semantic ID,"
              " NEW birth then reusable);")
    md.append("- Phase 6B bottleneck: Q1 Known acc = 0, RN-Acc = 0,"
              " cross-physical reuse = 0; TRAIN semantics separable but TAO Q1"
              " semantic geometry mismatched because the frame-level semantic"
              " head drifted far from the foundation space (inference"
              " max-known-sim mean -3.68).")
    md.append("")
    md.append("## 3. Phase 6C core question and hypothesis")
    md.append("")
    md.append("Can TrackOCD learn a transferable category-level semantic space"
              " from supported-known labels + unlabeled TRAIN trajectories + a"
              " generic pretrained foundation model, such that known categories"
              " are preserved, true-OOV categories are discovered online, and"
              " the same novel category is reused across different physical"
              " trajectories?")
    md.append("")
    md.append("Hypothesis: replace the Phase 6B frame-level semantic head with a"
              " **Trajectory Semantic Encoder (TSE)** anchored in the frozen"
              " DINOv2 track-feature geometry, trained with known-anchor"
              " preservation + same-track temporal consistency + cross-track"
              " mutual-nearest-neighbor discovery on real TRAIN unlabeled"
              " trajectories, and evaluated with an online B2-style memory.")
    md.append("")
    md.append("## 4. Prior art (verified)")
    md.append("")
    md.append("Full verified list: `docs/iclr27_phase6c/PRIOR_ART.md`. Key items:")
    md.append("")
    md.append("- TRACT / TraCLIP (ICCV 2025, arXiv 2503.08145): trajectory"
              " feature aggregation + semantic enrichment for open-vocabulary"
              " tracking;")
    md.append("- GET (CVPR 2025, arXiv 2403.09974): CLIP multi-modal GCD with"
              " text-embedding synthesis;")
    md.append("- Prior-Constrained Association Learning (AAAI 2025, arXiv"
              " 2502.09501): association between known and unlabeled instances;")
    md.append("- DiffGRE / TTD / TALON / Video-GCD MCCL / NC-GCD (verified in"
              " Phase 6B docs): online attach/create memory, calibrated logits,"
              " temporal consistency, ETF anchors;")
    md.append("- Internal frozen V0 baseline: DINOv2 track-mean + train-known"
              " prototypes + B2 online memory (Known 0.486 / route-novel 0.256"
              " on the full TrackOCD-v1 protocol); DINOv3 showed no clear gain"
              " over DINOv2, so DINOv2 remains the foundation backbone.")
    md.append("")
    md.append("## 5. Method")
    md.append("")
    md.append("### 5.1 Trajectory Semantic Encoder (TSE)")
    md.append("")
    md.append("- Input: frozen DINOv2 ViT-B/14 bbox-crop features, L2-normalized"
              " per frame (768-d).")
    md.append("- Encoder: PCA-initialized linear base (128-d) + zero-initialized"
              " residual MLP (128-256-128), L2-normalized frame embeddings;"
              " track embedding = masked mean of frame embeddings, L2-normalized.")
    md.append("- 48 known anchors initialized from PCA-projected TRAIN class"
              " means; learnable with an anchor-preservation MSE.")
    md.append("")
    md.append("### 5.2 Objectives")
    md.append("")
    md.append("1. Known CE + anchor attraction on 2196 supported-known TRAIN"
              " tracks;")
    md.append("2. Same-track temporal InfoNCE: every frame pulled to its own"
              " track mean, repelled by other tracks and anchors;")
    md.append("3. Cross-track discovery: on 11,377 unlabeled TRAIN trajectories"
              " (model-generated stream, GT ignored), each track is pulled to"
              " the mean of its mutual top-k neighbors;")
    md.append("4. Anchor preservation: residual drift + anchor MSE keep the"
              " learned space close to the frozen PCA/foundation geometry.")
    md.append("")
    md.append("### 5.3 Online semantic memory (B2-style, strict causal)")
    md.append("")
    md.append("- Per physical track: causal EMA of TSE frame embeddings"
              " (alpha=0.30);")
    md.append("- KNOWN if max known-anchor cosine >= tau;")
    md.append("- EXISTING if max novel-memory cosine >= tau (EMA update);")
    md.append("- NEW otherwise: birth a novel slot, reusable by later physical"
              " tracks;")
    md.append("- tau calibrated on the legal proxy split (24 held-out"
              " supported-known classes as proxy-novel, Hungarian ACC).")
    md.append("")
    md.append("## 6. Training protocol")
    md.append("")
    md.append("- Data: 2196 known tracks (48 classes) + 11,377 Phase4T TRAIN"
              " unlabeled pred tracks (>=3 rows) + 374 GT-identity-grouped"
              " TRAIN trajectories (identity only; category labels ignored);"
              " per-track up to 8 sampled frames.")
    md.append("- Batch: class-balanced known tracks + 64 unlabeled pred tracks"
              " (main) or 48 pred + 48 GT-grouped tracks (TSE-v2) per step;"
              " 250 steps/epoch; 60 epochs; AdamW lr=1e-3 (anchors 5e-4),"
              " cosine schedule, grad clip 5.0.")
    md.append("- Weights (main): w_attr=1.0, w_frame=0.5, w_mnn=1.0, w_pres=0.1,"
              " temperature 0.07. TSE-v2 adds open-space repulsion on"
              " self-selected likely-novel unlabeled tracks"
              " (open_thresh=0.85, open_margin=0.55, w_open=2.0) and stronger"
              " MNN (w_mnn=5.0, k=10).")
    md.append("- Resources: 1 A100 40GB per run (GPU 0/7/9), ~10 min/run.")
    md.append("")
    md.append("## 7. Results on the frozen Q1 strict stream")
    md.append("")
    md.append("Physical stream: Phase 6B DSCT final (2947 rows / 373 tracks /"
              " 710 frames / 45 aligned GT tracks). Only semantic columns were"
              " recomputed; physical numbers are unchanged by construction.")
    md.append("")
    md.append("### 7.1 Phase 6C main vs Phase 6B")
    md.append("")
    md.append(md_table(main_rows, [
        "run", "known", "birth", "reuse", "cross", "born", "nmi", "ari",
        "frag", "dup", "legacy_known", "legacy_route", "legacy_cond",
        "legacy_all", "tau"]))
    md.append("")
    md.append("Phase 6B reference (strict occurrence-level):")
    md.append("")
    for k in ("known_occurrence_acc", "first_novel_birth_acc", "novel_reuse_acc",
              "cross_physical_reuse_acc", "n_born_novel_states", "novel_nmi",
              "novel_ari", "mean_fragmentation", "duplicate_creation_rate"):
        md.append(f"- {k}: {fmt(phase6b_s.get(k))}")
    md.append("")
    md.append("Phase 6B legacy first-frame: known "
              f"{fmt(phase6b_lf.get('overall_known_acc'))}, route-novel "
              f"{fmt(phase6b_lf.get('route_aware_novel_acc'))}, all-track "
              f"{fmt(phase6b_lf.get('all_track_acc'))}.")
    md.append("")
    md.append("### 7.2 Non-parametric V0 baselines (no training)")
    md.append("")
    md.append(md_table(v0_rows, [
        "run", "known", "birth", "reuse", "cross", "born", "nmi", "ari",
        "frag", "dup", "legacy_known", "legacy_route", "legacy_all", "tau"]))
    md.append("")
    md.append("### 7.3 Key ablations")
    md.append("")
    md.append(md_table(abl_rows, [
        "run", "known", "birth", "reuse", "cross", "born", "nmi", "ari",
        "frag", "dup", "legacy_known", "legacy_route", "legacy_all", "tau"]))
    md.append("")
    md.append("Required ablation mapping: frame vs trajectory ="
              " `v0_frame_t045` vs `v0_traj_t045`, and TSE `tse_main_frame` vs"
              " `tse_main`; no cross-track discovery = `tse_abl_no_mnn`; no"
              " same-track consistency = `tse_abl_no_frame`; no known-anchor"
              " preservation = `tse_abl_no_pres`; Phase 6B semantic vs Phase 6C"
              " = Phase 6B row vs `tse_main`.")
    md.append("")
    md.append("### 7.4 Policy frontier diagnostics (not the main protocol)")
    md.append("")
    md.append("These runs use separate known/novel thresholds + a relative"
              " novel margin. They show that cross-physical reuse becomes >0"
              " only when the known threshold is raised far beyond the legal"
              " calibration, at the cost of known accuracy and massive"
              " over-birth — exactly the forbidden `known-for-novel` trade-off.")
    md.append("")
    md.append(md_table(frontier_rows, [
        "run", "known", "birth", "reuse", "cross", "born", "cnt_err",
        "nmi", "ari", "legacy_known", "legacy_route", "legacy_all"]))
    md.append("")
    md.append("## 8. Failure analysis and scientific judgment")
    md.append("")
    md.append("### 8.1 What was achieved")
    md.append("")
    md.append("- Known recovery: TSE raises strict Known occurrence accuracy"
              " from Phase 6B 0.0 -> V0 0.277 -> **TSE 0.671** (TSE-v2 0.720)"
              " and legacy first-frame known from 0.0 -> 0.224, while keeping"
              " physical metrics and the causal contract unchanged (373 tracks,"
              " frag 8, dup 2, no-future/no-relabel/first-frame-immediacy all"
              " pass).")
    md.append("- Representation quality: known leave-one-out accuracy rises"
              " 0.694 (raw DINOv2) -> 0.988 (TSE); known intra/inter class"
              " cosine ratio rises 1.32 -> 4.11; Q1 known-track prototype"
              " accuracy 0.559 -> 0.676.")
    md.append("- Novel memory mechanics: 10 born novel states (count error 1"
              " vs 9 for Phase 6B), NMI 0.820 / ARI 0.620 on aligned novel"
              " rows; 59 semantic slots are shared across physical tracks in"
              " the full stream (memory reuse mechanism works).")
    md.append("")
    md.append("### 8.2 Why cross-physical reuse stays 0 in the main protocol")
    md.append("")
    md.append("1. **Aligned data is nearly empty of cross-track novel"
              " categories**: Q1 aligned stream has 11 novel tracks / 9"
              " categories; only categories 611 and 817 have two physical"
              " tracks (max 21 correct cross rows out of 102 reuse rows).")
    md.append("2. **First-frame known/novel overlap**: in TSE space, Q1 known"
              " first-row max-known-sim median 0.740 vs novel 0.623 (ranges"
              " overlap heavily; a known track is as low as 0.019 and a novel"
              " track as high as 0.893). The calibrated threshold cannot"
              " separate them; raising it births more novel but collapses"
              " known.")
    md.append("3. **FP-born slots interfere**: early low-confidence FP tracks"
              " birth novel slots that absorb the true novel tracks later"
              " (`existing` instead of `new`), so the first correct birth for"
              " a novel category often never happens on an aligned track.")
    md.append("4. **One of the two cross-capable categories is intrinsically"
              " hard**: offline cross-track cosine for cat 817 is 0.474 (TSE)"
              " / 0.250 (raw); first-row-vs-mean is 0.329 / 0.193, below any"
              " usable novel threshold. Cat 611 reaches 0.878 but its first"
              " track is absorbed by known anchor 1057.")
    md.append("5. **Legal unlabeled TRAIN pool is too small**: only 39 clean"
              " novel GT-grouped tracks (27 categories) and 81 fragmented"
              " novel-role pred tracks are available in the 60-video Phase 4T"
              " subset; MNN trained on this pool did not transfer to Q1"
              " (no-mnn ablation ≈ main), and the open-space repulsion"
              " (TSE-v2) improved known accuracy but did not improve novel"
              " reuse.")
    md.append("")
    md.append("### 8.3 Judgment")
    md.append("")
    md.append("The Phase 6C representation reset supports the first half of"
              " the hypothesis (foundation trajectory features + known-anchor"
              " preservation recover known semantics and a usable novel"
              " memory) but does NOT support the second half (true-OOV"
              " category-level semantics transferable across physical"
              " trajectories under strict causality). With the legal supervision"
              " available, the Q1 stream cannot be solved without sacrificing"
              " known accuracy. This is a data/representation limit, not a"
              " wiring bug.")
    md.append("")
    md.append("**`TRUE_OOV_SEMANTIC_SUPERVISION_REMAINS_LIMITING`** and"
              " **`TRACKOCD_NOT_YET_ICLR_LEVEL`** are therefore written"
              " explicitly.")
    md.append("")
    md.append("## 9. ICLR novelty assessment")
    md.append("")
    md.append("- Contributions with evidence: (a) strict-causal trajectory"
              " semantic encoder anchored in a frozen foundation space;"
              " (b) showing that known-anchor preservation is what recovers"
              " known semantics (0 -> 0.67) whereas the Phase 6B frame-level"
              " head destroyed them; (c) an honest negative result that the"
              " legal unlabeled TRAIN pool cannot support true-OOV"
              " cross-physical reuse on Q1.")
    md.append("- Missing for ICLR: a positive cross-physical novel reuse"
              " result without known collapse, and evidence that novel"
              " categories transfer from TRAIN unlabeled streams. Until then,"
              " the paper is a partial system + negative analysis, not a"
              " complete ICLR contribution.")
    md.append("")
    md.append("## 10. Next steps")
    md.append("")
    md.append("- Obtain a larger legal TRAIN unlabeled pool (full TAO TRAIN"
              " with identity-only grouping + DINOv2 features) so cross-track"
              " novel discovery has real signal.")
    md.append("- Investigate FP-slot suppression that is causal-safe (e.g.,"
              " track-quality priors for birth, without retroactive relabel).")
    md.append("- Consider a two-stage open-world objective where known anchors"
              " are frozen after Stage A and only novel-side memory is learned"
              " from the unlabeled stream.")
    md.append("- Re-run the strict protocol on Q2 / larger dev splits before"
              " any further ICLR claim.")
    md.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
