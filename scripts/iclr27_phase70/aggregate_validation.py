#!/usr/bin/env python3
"""Aggregate Phase70 TRAIN/validation full-sequence and TrackEval diagnostics."""
from __future__ import annotations
import hashlib, json, pathlib, statistics, tempfile, os

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/iclr27_phase70/validation/joint_d_repair1"
OUT = BASE / "validation_aggregate.json"

def atomic_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def parse_summary(p: pathlib.Path):
    lines = p.read_text(errors="replace").strip().splitlines()
    if len(lines) < 2: return None
    header, values = lines[-2].split(), lines[-1].split()
    out = {}
    for k, v in zip(header, values):
        try: out[k] = float(v)
        except ValueError: out[k] = None
    return out

def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main() -> None:
    rows = []
    for fold in range(4):
        metric_path = BASE / f"fold{fold}_metrics.json"
        metric = json.loads(metric_path.read_text())
        pred = BASE / f"fold{fold}_eval/teta_results/tao_track.json"
        summaries = [parse_summary(p) for p in sorted((BASE / f"trackeval/fold{fold}/results").glob("**/*_summary.txt"))]
        summaries = [x for x in summaries if x is not None]
        keys = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Frag"]
        macro = {k: statistics.fmean([x[k] for x in summaries if x.get(k) is not None]) if any(x.get(k) is not None for x in summaries) else None for k in keys}
        sum_keys = ["CLR_TP", "CLR_FN", "CLR_FP", "IDTP", "IDFN", "IDFP", "Dets", "GT_Dets", "IDSW", "Frag"]
        sums = {k: sum((x.get(k) or 0.0) for x in summaries) for k in sum_keys}
        gt, det = sums["GT_Dets"], sums["Dets"]
        weighted = {
            "CLR_Re": sums["CLR_TP"] / gt if gt else 0.0,
            "CLR_Pr": sums["CLR_TP"] / det if det else 0.0,
            "IDR": sums["IDTP"] / (sums["IDTP"] + sums["IDFN"]) if sums["IDTP"] + sums["IDFN"] else 0.0,
            "IDP": sums["IDTP"] / (sums["IDTP"] + sums["IDFP"]) if sums["IDTP"] + sums["IDFP"] else 0.0,
        }
        top = metric["recall"]["topk"]["20"]
        rows.append({
            "fold": fold,
            "prediction": {"path": str(pred.resolve()), "bytes": pred.stat().st_size, "sha256": sha256(pred), "count": metric["prediction"]["count"]},
            "gt_rows": metric["gt"]["annotations"],
            "top20_recall": {k: v["recall"] for k, v in top["thresholds"].items()},
            "top20_mean_best_iou": top["mean_best_iou"],
            "top20_median_best_iou": top["median_best_iou"],
            "track_continuity_proxy": metric["track_continuity_proxy"],
            "trackeval_summary_count": len(summaries),
            "trackeval_macro": macro,
            "trackeval_count_sums": sums,
            "trackeval_count_weighted": weighted,
        })
    def mean(path):
        vals = []
        for row in rows:
            x = row
            for k in path: x = x[k]
            vals.append(x)
        return statistics.fmean(vals)
    agg = {
        "protocol": "trackocd_phase70_joint_d_repair1_train_validation_full_sequence",
        "folds": rows,
        "fold_mean": {
            "top20_recall_iou03": mean(("top20_recall", "0.3")),
            "top20_recall_iou05": mean(("top20_recall", "0.5")),
            "top20_recall_iou07": mean(("top20_recall", "0.7")),
            "top20_mean_best_iou": mean(("top20_mean_best_iou",)),
            "top20_median_best_iou": mean(("top20_median_best_iou",)),
            "continuity_mean_reliable_fraction": mean(("track_continuity_proxy", "mean_reliable_fraction")),
            "trackeval_macro_HOTA": mean(("trackeval_macro", "HOTA")),
            "trackeval_macro_DetA": mean(("trackeval_macro", "DetA")),
            "trackeval_macro_AssA": mean(("trackeval_macro", "AssA")),
            "trackeval_macro_MOTA": mean(("trackeval_macro", "MOTA")),
            "trackeval_macro_IDF1": mean(("trackeval_macro", "IDF1")),
            "trackeval_macro_IDSW": mean(("trackeval_macro", "IDSW")),
            "trackeval_macro_Frag": mean(("trackeval_macro", "Frag")),
        },
        "references": {
            "phase68_q0_trackeval": str((ROOT / "outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval_aggregate.json").resolve()),
            "phase69_class_agnostic": str((ROOT / "outputs/iclr27_phase69/metrics/phase69_aggregate.json").resolve()),
        },
        "labels_used_for_model": False,
        "sealed_public_q1_accessed": False,
        "held_event_gt_used_for_model": False,
    }
    atomic_json(OUT, agg)
    print(json.dumps({"out": str(OUT), "fold_mean": agg["fold_mean"]}, indent=2))

if __name__ == "__main__": main()
