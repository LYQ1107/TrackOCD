"""Build the self-contained Phase 7A complete copyable report."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def load(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else {}


def fmt(v, digits=4):
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def strict_table(name, split):
    d = load(f"outputs/iclr27_phase7a/eval/{name}/{split}_strict/summary.json")
    s = d.get("strict", {})
    keys = [
        "n_aligned_occurrences", "n_known_occurrences", "n_novel_occurrences",
        "n_first_novel_occurrences", "n_novel_reuse_occurrences",
        "known_occurrence_acc", "first_novel_birth_acc", "novel_reuse_acc",
        "cross_physical_reuse_acc", "cross_physical_reuse_share",
        "n_born_novel_states", "n_true_novel_categories",
        "novel_count_abs_error", "novel_nmi", "novel_ari",
        "mean_fragmentation", "duplicate_creation_rate",
        "semantic_switch_rate", "known_to_new_rate", "known_to_existing_rate",
    ]
    return " | ".join(fmt(s.get(k)) for k in keys), keys


def main():
    runs = {
        "ema_baseline": "simple EMA baseline (Phase 6C B2-style)",
        "racc_main": "RACC-Memory (Phase 7A main)",
    }
    lines = []
    ap = lines.append
    ap("# TrackOCD ICLR 2027 — Phase 7A Complete Report")
    ap("")
    ap("## Reliability-Aware Causal Category Memory under Strict TrackOCD")
    ap("")
    ap("Date: 2026-08-19. Project root: "
       "`/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT`.")
    ap("")
    # status placeholder (filled after final results)
    status = "FINAL_STATUS_PLACEHOLDER"
    ap(f"## 1. Final status\n\n**FINAL_STATUS: `{status}`**\n")
    ap("## 2. Strict TrackOCD protocol\n")
    ap((ROOT / "docs/iclr27_phase7a/PROTOCOL.md").read_text())
    ap("## 3. OCD compatibility\n")
    ap("Phase 7A is compatible with standard OCD: it consumes only "
       "supported-known labels + unlabeled online streams, and reports "
       "Known/RN-Acc/NMI/ARI/count error. It adds TrackOCD's per-frame "
       "immediate decisions and the CT-Reuse metric (see below), which "
       "standard OCD protocols do not define.\n")
    ap("## 4. Q1 DEV / locked final split\n")
    ap("Q1 DEV is used for diagnostics; the locked heldout split (24 videos, "
       "3037 rows) is evaluated once after freezing. Because this project "
       "historically used TAO validation annotations, the heldout split is "
       "declared a locked-with-history split, not a truly untouched test "
       "split.\n")
    ap("## 5. 2025/2026 prior art\n")
    ap((ROOT / "docs/iclr27_phase7a/PRIOR_ART.md").read_text())
    ap("## 6. Method\n")
    ap((ROOT / "docs/iclr27_phase7a/PROTOCOL.md").read_text().split(
        "## 5. Ablations")[0].split("## 4. Method summary")[-1])
    ap("## 7. Training\n")
    ap("Proxy episodes are built from the corrected Phase 4T TRAIN stream "
       "(bbox bug fixed in `stream_data.py`; DINOv2 features re-extracted), "
       "class-level anchor hiding + leave-k-out episodic pseudo-novel, "
       "balanced track-level sampling, reliability-weighted memory updates, "
       "and frozen TSE. The only trained module is the attach-or-create "
       "head.\n")
    ap("## 8. Results\n")
    for name, label in runs.items():
        ap(f"### {label} ({name})")
        for split in ("dev", "heldout"):
            row, keys = strict_table(name, split)
            ap(f"**{split}**: | " + " | ".join(keys) + " |")
            ap("| " + row + " |\n")
    ap("## 9. CT-Reuse detailed cases\n")
    for split in ("dev", "heldout"):
        p = ROOT / f"outputs/iclr27_phase7a/eval/racc_main/{split}_ctreuse.json"
        if p.exists():
            c = json.loads(p.read_text())
            ap(f"### {split}\n")
            ap(f"- cross rows: {c['n_cross_rows']}, correct: "
               f"{c['n_correct']}, CT-Reuse: {c['cross_physical_reuse_acc']}")
            ap("")
            ap("```json")
            ap(json.dumps(c["cases"][:20], indent=1))
            ap("```\n")
    ap("## 10. Memory contamination\n")
    ap("Measured by mean fragmentation, duplicate creation rate, "
       "semantic switch rate, and new-precision on aligned rows (tables "
       "above).\n")
    ap("## 11. Ablations\n")
    ap("See `outputs/iclr27_phase7a/eval/*_abl*` (filled after runs).\n")
    ap("## 12. Comparison with Phase 6C/6D\n")
    ap("Phase 6C main: Known 0.671 / first-birth 0.222 / reuse 0.196 / "
       "cross 0 / NMI 0.820. Phase 6D small-pool: Known 0.769 / cross 0.014. "
       "Phase 7A numbers above.\n")
    ap("## 13. ICLR novelty assessment\n")
    ap("PLACEHOLDER\n")
    ap("## 14. Final innovation point\n")
    ap("PLACEHOLDER\n")
    ap("## 15. Next steps\n")
    ap("PLACEHOLDER\n")
    out = ROOT / "docs/iclr27_phase7a/PHASE7A_COMPLETE_COPYABLE_REPORT.md"
    out.write_text("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
