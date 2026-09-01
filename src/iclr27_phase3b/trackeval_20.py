"""Preliminary 20-video SimOWT vs ByteTrack TrackEval comparison."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TRACKEVAL_ROOT = ROOT / "third_party" / "TrackEval"
sys.path.insert(0, str(TRACKEVAL_ROOT))

import trackeval  # noqa: E402

GT_FOLDER = ROOT / "outputs" / "iclr27_phase3a" / "trackeval" / "gt"
TRACKERS_FOLDER = ROOT / "outputs" / "iclr27_phase3b" / "trackeval_20"
TRACKERS = ["simowt", "bytetrack"]


def build_tracker(name, input_dir):
    anns = []
    for p in sorted(Path(input_dir).glob("*.json")):
        anns.extend(json.loads(p.read_text()))
    anns.sort(key=lambda a: (a["video_id"], a["image_id"], a["track_id"]))
    out = TRACKERS_FOLDER / name / "data" / "pred.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(anns, separators=(",", ":")))
    print("built", name, len(anns))


def main():
    TRACKERS_FOLDER.mkdir(parents=True, exist_ok=True)
    build_tracker("simowt", ROOT / "outputs/iclr27_phase3a/trajectories/instrumented_online_20")
    build_tracker("bytetrack", ROOT / "outputs/iclr27_phase3b/bytetrack_smoke/predictions")
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_ONLY_COMBINED"] = True
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    dataset_config = trackeval.datasets.TAO_OW.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(GT_FOLDER)
    dataset_config["TRACKERS_FOLDER"] = str(TRACKERS_FOLDER)
    dataset_config["TRACKERS_TO_EVAL"] = TRACKERS
    dataset_config["TRACKER_SUB_FOLDER"] = "data"
    dataset_config["SUBSET"] = "all"
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.TAO_OW(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    output_res, _ = evaluator.evaluate([dataset], metrics)
    rows = []
    for tracker in TRACKERS:
        combined = output_res["TAO_OW"][tracker]["COMBINED_SEQ"]["cls_comb_cls_av"]
        h = combined["HOTA"]
        c = combined["CLEAR"]
        i = combined["Identity"]
        deta = float(np.mean(h["DetA"])) * 100.0
        assa = float(np.mean(h["AssA"])) * 100.0
        rows.append({
            "tracker": tracker,
            "HOTA": h["HOTA(0)"], "DetA": deta, "AssA": assa,
            "LocA": h["LocA(0)"],
            "MOTA": c["MOTA"], "IDF1": i["IDF1"], "IDSW": c["IDSW"],
            "Frag": c["Frag"], "CLR_TP": c["CLR_TP"], "CLR_FP": c["CLR_FP"],
            "CLR_FN": c["CLR_FN"],
        })
    out_dir = ROOT / "outputs" / "iclr27_phase3b" / "tracking"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "frontend_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "frontend_metrics_20.json").write_text(
        json.dumps({t: {m: {k: (v.tolist() if hasattr(v, "tolist") else v)
                             for k, v in output_res["TAO_OW"][t]["COMBINED_SEQ"]["cls_comb_cls_av"][m].items()}
                        for m in output_res["TAO_OW"][t]["COMBINED_SEQ"]["cls_comb_cls_av"]}
                   for t in TRACKERS}, indent=1, default=str))
    print(rows)


if __name__ == "__main__":
    main()
