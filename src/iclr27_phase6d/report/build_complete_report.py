"""Build docs/iclr27_phase6d/PHASE6D_COMPLETE_COPYABLE_REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "docs" / "iclr27_phase6d" / "PHASE6D_COMPLETE_COPYABLE_REPORT.md"
EVAL = ROOT / "outputs" / "iclr27_phase6d" / "eval"


def fmt(v, nd=3):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def load_strict(name):
    p = EVAL / name / "strict" / "summary.json"
    if not p.exists():
        return {}, {}
    d = json.loads(p.read_text())
    return d.get("strict", {}), d.get("legacy_first_frame", {})


def load_cal(name):
    p = EVAL / name / "calibration.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def run_table(names):
    rows = []
    for n in names:
        s, lf = load_strict(n)
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
            "legacy_all": lf.get("all_track_acc"),
            "tau": c.get("tau"),
        })
    return rows


def md_table(rows, cols):
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def main():
    stats = json.loads(
        (ROOT / "outputs/iclr27_phase6d/assets/full_tao_stats.json").read_text())
    main_rows = run_table(["gmna_main", "gmna_open"])
    best_rows = run_table(["gmna_small_pool"])
    abl_rows = run_table(["gmna_no_discovery", "gmna_no_teacher"])
    c6_s, c6_lf = load_strict("phase6c_tse_main")
    md = []
    md.append("# TrackOCD ICLR 2027 — Phase 6D Complete Report")
    md.append("")
    md.append("## Full-TAO Cross-Physical Novel Category Discovery")
    md.append("")
    md.append("Date: 2026-08-18. Project root: "
              "`/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT`.")
    md.append("")
    md.append("## 1. Final status")
    md.append("")
    md.append("**FINAL_STATUS: `CROSS_PHYSICAL_TRUE_OOV_REUSE_NOT_SOLVED` / "
              "`TRUE_OOV_SEMANTIC_SUPERVISION_REMAINS_LIMITING` / "
              "`TRACKOCD_NOT_YET_ICLR_LEVEL`**")
    md.append("")
    md.append("Full-TAO GMNA and its known-filtered variant (OpenGMNA) both "
              "collapse true-OOV into known classes (cross reuse 0, novel "
              "birth ~0, NMI < 0.45). The small-pool GMNA run is the only "
              "configuration with a nonzero strict cross-physical reuse "
              "(0.014, one correct row for category 611) while keeping known "
              "at 0.769, but RN-Acc is not improved (legacy route 0.091 vs "
              "Phase 6C 0.273) and the effect is not robust. The large legal "
              "unlabeled pool does not enable transferable true-OOV "
              "cross-physical category reuse.")
    md.append("")
    md.append("## 2. Frozen conclusions from Phase 6C")
    md.append("")
    md.append("- TSE direction is effective: strict Known 0.671, NMI 0.820, "
              "ARI 0.620, novel count error 1;")
    md.append("- physical/objectness + strict-causal DSCT framework preserved;")
    md.append("- true-OOV cross-physical reuse remains 0;")
    md.append("- simple in-batch MNN is ineffective (no-MNN ≈ main);")
    md.append("- threshold/margin tuning is abandoned (known collapses or "
              "novel birth explodes).")
    md.append("")
    md.append("## 3. Full-TAO legal trajectory pool")
    md.append("")
    md.append("Source: TAO TRAIN annotation file; boxes and GT track identity "
              "only. Category labels are used only for statistics and the "
              "supported-known subset (48 classes); they are never used as "
              "novel supervision. All 18,274 frames verified present.")
    md.append("")
    for k, v in stats.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("Feature extraction: DINOv2 ViT-B/14 bbox crops (518px, +10% "
              "context, <=8 frames/track), 3 GPUs; known-track caches reused "
              "after exact sample-id verification (2196/2196), remaining "
              "tracks extracted fresh. Assembled 2644 tracks (99.9% coverage; "
              "3 tracks had no valid crops).")
    md.append("")
    md.append("## 4. Prior art (verified)")
    md.append("")
    md.append("Full list: `docs/iclr27_phase6d/PRIOR_ART.md`. Key items: "
              "OCGCD/DEAN (ECCV 2024, github.com/KHU-AGI/OCGCD), MCCL "
              "(Video-GCD 2025), Beyond Known Clusters (2024), plus Phase 6C "
              "verified TRACT/GET/Prior-Constrained Association.")
    md.append("")
    md.append("## 5. Method")
    md.append("")
    md.append("### 5.1 Architecture 1: Global Memory-Bank Neighborhood "
              "Aggregation (GMNA)")
    md.append("")
    md.append("One cross-track discovery principle replacing simple MNN:")
    md.append("")
    md.append("1. Momentum teacher encoder (EMA 0.999 of the Phase 6C TSE "
              "student);")
    md.append("2. Global memory bank of teacher trajectory embeddings over "
              "the legal pool, refreshed per epoch (bank EMA 0.90);")
    md.append("3. Confidence-aware pseudo-pairs: mutual top-k cross-video "
              "neighbors (k=10) in the bank, softmax-weighted aggregation;")
    md.append("4. Student losses: known CE + anchor attraction (fidelity), "
              "neighborhood-target attraction (w=2.0), teacher-student "
              "consistency (w=0.5), anchor-preservation MSE (w=0.1).")
    md.append("")
    md.append("### 5.2 Architecture 2 (one allowed switch): OpenGMNA")
    md.append("")
    md.append("Same student/teacher/bank, plus:")
    md.append("- known-filtered bank: only self-selected likely-novel tracks "
              "(teacher max-known-sim < 0.75) participate in the graph;")
    md.append("- open-space separation: hinge loss pushing likely-novel track "
              "embeddings below 0.50 max-known similarity (w=2.0).")
    md.append("")
    md.append("## 6. Training protocol")
    md.append("")
    md.append("- Full pool: 2644 tracks (2196 known / 448 unlabeled), track "
              "mean DINOv2 features; 64 class-balanced known + 96 unlabeled "
              "tracks per step; 125 steps/epoch; 60 epochs; AdamW lr=1e-3 "
              "(anchors 5e-4), cosine, grad clip 5.0.")
    md.append("- Small pool (ablation): Phase 6C pool, 11,751 tracks, same "
              "architecture.")
    md.append("- Resources: 1 A100 per run; GPUs 0/7/9 (other users' jobs "
              "untouched).")
    md.append("")
    md.append("## 7. Results on the frozen Q1 strict stream")
    md.append("")
    md.append("Physical stream identical to Phase 6B/6C (2947 rows / 373 "
              "tracks / 45 aligned). Only semantic columns recomputed.")
    md.append("")
    md.append("### 7.1 Phase 6D main (full-TAO pool)")
    md.append("")
    md.append(md_table(main_rows, [
        "run", "known", "birth", "reuse", "cross", "born", "cnt_err",
        "nmi", "ari", "frag", "dup", "legacy_known", "legacy_route",
        "legacy_all", "tau"]))
    md.append("")
    md.append("Phase 6C TSE reference: strict known "
              f"{fmt(c6_s.get('known_occurrence_acc'))}, birth "
              f"{fmt(c6_s.get('first_novel_birth_acc'))}, reuse "
              f"{fmt(c6_s.get('novel_reuse_acc'))}, cross "
              f"{fmt(c6_s.get('cross_physical_reuse_acc'))}, NMI "
              f"{fmt(c6_s.get('novel_nmi'))}; legacy all-track "
              f"{fmt(c6_lf.get('all_track_acc'))} / route "
              f"{fmt(c6_lf.get('route_aware_novel_acc'))}.")
    md.append("")
    md.append("Phase 6B reference: strict known 0.000, first-novel-birth "
              "0.000, novel reuse 0.000, cross reuse 0.000, NMI 0.437; "
              "legacy all-track 0.031 / route 0.136.")
    md.append("")
    md.append("### 7.2 Best-config run (small pool, diagnostic)")
    md.append("")
    md.append("The same GMNA architecture trained on the Phase 6C small pool "
              "(11,751 fragmented/GT-grouped TRAIN tracks) is the only run "
              "with nonzero strict cross-physical reuse:")
    md.append("")
    md.append(md_table(best_rows, [
        "run", "known", "birth", "reuse", "cross", "born", "cnt_err",
        "nmi", "ari", "frag", "dup", "legacy_known", "legacy_route",
        "legacy_all", "tau"]))
    md.append("")
    md.append("This shows the online mechanism can fire (category 611: first "
              "track births slot 100000; one later row of the second track "
              "correctly reuses it), but the effect is a single row "
              "(cross acc = 1/74) and RN-Acc is worse than Phase 6C, so it "
              "is not a solution.")
    md.append("")
    md.append("### 7.3 Ablations")
    md.append("")
    md.append(md_table(abl_rows, [
        "run", "known", "birth", "reuse", "cross", "born", "cnt_err",
        "nmi", "ari", "frag", "dup", "legacy_known", "legacy_route",
        "legacy_all", "tau"]))
    md.append("")
    md.append("Mapping: `gmna_small_pool` = small pool under the same GMNA "
              "architecture (small vs full TAO); `gmna_no_discovery` = no "
              "cross-track objective; `gmna_no_teacher` = no momentum teacher "
              "/ global bank; Phase 6C TSE vs Phase 6D = reference row above; "
              "old MNN vs Phase 6D method = Phase 6C `tse_main` (in-batch "
              "MNN) vs `gmna_small_pool`.")
    md.append("")
    md.append("## 8. Comparison with Phase 5A / 6B / 6C")
    md.append("")
    md.append("| run | known | birth | reuse | cross | born | NMI | all |")
    md.append("|---|---|---|---|---|---|---|---|")
    md.append("| Phase 5A | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.262 | 0.031 |")
    md.append("| Phase 6B | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0.437 | 0.031 |")
    md.append("| Phase 6C TSE | 0.671 | 0.222 | 0.196 | 0.000 | 10 | 0.820 | 0.235 |")
    md.append("| Phase 6D GMNA full | 0.813 | 0.111 | 0.000 | 0.000 | 1 | 0.271 | 0.245 |")
    md.append("| Phase 6D OpenGMNA | 0.732 | 0.000 | 0.000 | 0.000 | 0 | 0.448 | 0.204 |")
    md.append("| Phase 6D GMNA small | 0.769 | 0.222 | 0.265 | 0.014 | 6 | 0.627 | 0.255 |")
    md.append("")
    md.append("(Phase 5A/6B legacy values from project reports; strict "
              "occurrence-level rows use the same evaluator.)")
    md.append("")
    md.append("## 9. Failure analysis and scientific judgment")
    md.append("")
    md.append("### 9.1 What was tested")
    md.append("")
    md.append("- Full legal TAO TRAIN pool: 500 videos / 18,274 frames / "
              "2,647 tracks (2,196 known / 451 unlabeled, 168 unlabeled "
              "categories, 1,219 novel same-category track pairs); DINOv2 "
              "features for 2,644 tracks (99.9% coverage).")
    md.append("- Architecture 1 (GMNA): momentum teacher + global memory bank "
              "+ mutual-neighbor aggregation over the full pool.")
    md.append("- Architecture 2 (OpenGMNA, the one allowed switch): "
              "known-filtered bank (novel-thresh 0.75) + open-space "
              "repulsion, same student/teacher.")
    md.append("")
    md.append("### 9.2 Results")
    md.append("")
    md.append("- GMNA full pool: known 0.813 but novel collapse (birth 0.111, "
              "reuse 0, NMI 0.271); the 83%-known bank absorbs novel tracks.")
    md.append("- OpenGMNA: known 0.732, birth 0, NMI 0.448; open-space "
              "repulsion did not transfer novelness to Q1 (calibrated tau "
              "dropped to 0.50 and all first novel occurrences still attached "
              "to known/FP slots).")
    md.append("- GMNA small pool: known 0.769, birth 0.222, reuse 0.265, "
              "cross 0.014 (1 row), NMI 0.627 — better than full-pool GMNA "
              "but RN-Acc lower than Phase 6C and effect tiny.")
    md.append("- TRAIN-side post-hoc diagnostic (448 novel tracks / 167 "
              "categories, KMeans): raw DINOv2 NMI 0.817 / ARI 0.201; GMNA "
              "NMI 0.818 / ARI 0.251; OpenGMNA NMI 0.814 / ARI 0.252. The "
              "learned objective does not sharpen true-OOV clustering beyond "
              "the frozen foundation.")
    md.append("")
    md.append("### 9.3 Judgment")
    md.append("")
    md.append("Large-scale legal unlabeled trajectories do NOT enable "
              "transferable true-OOV category structure across different "
              "physical tracks under strict causal TrackOCD with this "
              "training protocol. The foundation geometry already carries "
              "most of the novel structure (raw NMI 0.817); without novel "
              "labels the extra objectives either leave it unchanged or pull "
              "novel toward the known-dominated pool. The one nonzero cross "
              "row comes from the noisy small pool and is not a robust "
              "mechanism.")
    md.append("")
    md.append("**`CROSS_PHYSICAL_TRUE_OOV_REUSE_NOT_SOLVED`**, "
              "**`TRUE_OOV_SEMANTIC_SUPERVISION_REMAINS_LIMITING`**, and "
              "**`TRACKOCD_NOT_YET_ICLR_LEVEL`** are written explicitly.")
    md.append("")
    md.append("## 10. ICLR novelty assessment")
    md.append("")
    md.append("- Evidence: full-TAO legal pool construction; GMNA/OpenGMNA "
              "negative results; a controlled demonstration that the "
              "known-dominated global bank collapses OOV while the small "
              "noisy pool preserves a fragile cross-reuse signal; TRAIN-side "
              "diagnostics showing the learned objective adds ~0 ARI/NMI.")
    md.append("- Missing for ICLR: any robust positive cross-physical reuse "
              "with improved RN-Acc. The paper remains a system + negative "
              "analysis, not an ICLR contribution.")
    md.append("")
    md.append("## 11. Next steps")
    md.append("")
    md.append("- Explore novel-only self-selection at inference (e.g., "
              "foundation-space novelty prior from full-TAO clusters) with "
              "strict causal legality.")
    md.append("- Investigate FP-slot suppression / track-quality birth priors "
              "that are causal-safe.")
    md.append("- Consider a larger identity-only TRAIN tracker output (not GT "
              "identity) to enlarge the novel pool beyond 451 tracks.")
    md.append("- Re-evaluate on Q2 / full dev before any ICLR claim.")
    md.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
