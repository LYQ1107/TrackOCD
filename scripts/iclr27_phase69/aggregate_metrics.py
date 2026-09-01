#!/usr/bin/env python3
"""Aggregate Phase69 fold diagnostics without rerunning evaluation."""
from __future__ import annotations
import hashlib, json, os, pathlib, statistics, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase69/metrics/phase69_aggregate.json"


def atomic_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def parse_summary(path: pathlib.Path):
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2: return None
    h, v = lines[-2].split(), lines[-1].split()
    out = {}
    for k, x in zip(h, v):
        try: out[k] = float(x)
        except ValueError: out[k] = None
    return out


def main() -> None:
    fold_rows = []
    for fold in range(4):
        metric_path = ROOT / f"outputs/iclr27_phase69/metrics/fold{fold}_eval/full_sequence_metrics.json"
        metric = json.loads(metric_path.read_text())
        pred = ROOT / f"outputs/iclr27_phase69/metrics/fold{fold}_eval/teta_results/tao_track.json"
        summary_paths = sorted((ROOT / f"outputs/iclr27_phase69/trackeval/fold{fold}/results/PHASE69_F{fold}").glob("*_summary.txt"))
        rows = [parse_summary(p) for p in summary_paths]
        rows = [r for r in rows if r is not None]
        keys = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "LocA", "OWTA", "CLR_Re", "CLR_Pr", "IDSW", "Frag"]
        macro = {k: statistics.fmean([r[k] for r in rows if r.get(k) is not None]) if any(r.get(k) is not None for r in rows) else None for k in keys}
        sums_keys = ["CLR_TP", "CLR_FN", "CLR_FP", "IDTP", "IDFN", "IDFP", "Dets", "GT_Dets", "IDs", "GT_IDs", "IDSW", "Frag"]
        sums = {k: sum((r.get(k) or 0.0) for r in rows) for k in sums_keys}
        gt, det = sums["GT_Dets"], sums["Dets"]
        weighted = {
            "CLR_Re": sums["CLR_TP"] / gt if gt else 0.0,
            "CLR_Pr": sums["CLR_TP"] / det if det else 0.0,
            "IDR": sums["IDTP"] / (sums["IDTP"] + sums["IDFN"]) if sums["IDTP"] + sums["IDFN"] else 0.0,
            "IDP": sums["IDTP"] / (sums["IDTP"] + sums["IDFP"]) if sums["IDTP"] + sums["IDFP"] else 0.0,
        }
        top20 = metric["recall"]["topk"]["20"]
        fold_rows.append({
            "fold": fold,
            "prediction": {"path": str(pred), "bytes": pred.stat().st_size, "sha256": hashlib.sha256(pred.read_bytes()).hexdigest(), "count": metric["prediction"]["count"]},
            "gt_rows": metric["gt"]["annotations"],
            "top20_recall": {k: v["recall"] for k, v in top20["thresholds"].items()},
            "top20_mean_best_iou": top20["mean_best_iou"],
            "top20_median_best_iou": top20["median_best_iou"],
            "track_continuity_proxy": metric["track_continuity_proxy"],
            "trackeval_summary_count": len(rows),
            "trackeval_macro": macro,
            "trackeval_count_sums": sums,
            "trackeval_count_weighted": weighted,
        })
    def get_path(obj, path):
        for key in path:
            obj = obj[key]
        return obj
    def mean(path):
        vals = [get_path(x, path) for x in fold_rows]
        return statistics.fmean(vals)
    agg = {
        "protocol": "trackocd_phase69_ovtr_class_agnostic_full_sequence_validation",
        "folds": fold_rows,
        "fold_mean": {
            "top20_recall_iou03": mean(("top20_recall", "0.3")),
            "top20_recall_iou05": mean(("top20_recall", "0.5")),
            "top20_recall_iou07": mean(("top20_recall", "0.7")),
            "top20_mean_best_iou": mean(("top20_mean_best_iou",)),
            "top20_median_best_iou": mean(("top20_median_best_iou",)),
            "continuity_mean_reliable_fraction": mean(("track_continuity_proxy", "mean_reliable_fraction")),
            "trackeval_macro_HOTA": statistics.fmean(x["trackeval_macro"]["HOTA"] for x in fold_rows),
            "trackeval_macro_DetA": statistics.fmean(x["trackeval_macro"]["DetA"] for x in fold_rows),
            "trackeval_macro_AssA": statistics.fmean(x["trackeval_macro"]["AssA"] for x in fold_rows),
            "trackeval_macro_MOTA": statistics.fmean(x["trackeval_macro"]["MOTA"] for x in fold_rows),
            "trackeval_macro_IDF1": statistics.fmean(x["trackeval_macro"]["IDF1"] for x in fold_rows),
            "trackeval_macro_IDSW": statistics.fmean(x["trackeval_macro"]["IDSW"] for x in fold_rows),
            "trackeval_macro_Frag": statistics.fmean(x["trackeval_macro"]["Frag"] for x in fold_rows),
        },
        "notes": [
            "Top-k values are class-agnostic IoU diagnostics on validation annotations.",
            "TrackEval values are unweighted means over the 298 non-empty TAO class summary files per fold; exact files remain authoritative.",
            "No public/Q1/sealed labels were used as model inputs; annotations are post-hoc scoring only.",
        ],
    }
    atomic_json(OUT, agg)
    print(json.dumps({"out": str(OUT), "fold_mean": agg["fold_mean"]}, indent=2))


if __name__ == "__main__": main()
