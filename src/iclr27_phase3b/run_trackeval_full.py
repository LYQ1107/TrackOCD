"""Full TrackEval for SimOWT I and ByteTrack on the frozen detection stream."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TRACKEVAL_ROOT = ROOT / "third_party" / "TrackEval"
sys.path.insert(0, str(TRACKEVAL_ROOT))

import trackeval  # noqa: E402

GT_FOLDER = TRACKEVAL_ROOT / "data" / "gt" / "tao" / "tao_training"
TRACKERS_FOLDER = ROOT / "outputs" / "iclr27_phase3b" / "trackeval"
TRACKERS = ["simowt_instrumented", "bytetrack"]


def main():
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
    summary = {}
    per_seq = {}
    for tracker in TRACKERS:
        combined = output_res["TAO_OW"][tracker]["COMBINED_SEQ"]["cls_comb_cls_av"]
        summary[tracker] = {m: {k: (v.tolist() if hasattr(v, "tolist") else v)
                                for k, v in combined[m].items()} for m in combined}
        for seq, seq_res in output_res["TAO_OW"][tracker].items():
            if seq == "COMBINED_SEQ":
                continue
            cls = seq_res.get("object", {})
            h = cls.get("HOTA", {})
            c = cls.get("CLEAR", {})
            i = cls.get("Identity", {})
            per_seq.setdefault(seq, {})[tracker] = {
                "HOTA": h.get("HOTA(0)"),
                "MOTA": c.get("MOTA"),
                "IDF1": i.get("IDF1"),
                "IDSW": c.get("IDSW"),
                "Frag": c.get("Frag"),
            }
    out = ROOT / "outputs" / "iclr27_phase3b" / "tracking" / "frontend_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1, default=str))
    (ROOT / "outputs/iclr27_phase3b/tracking/per_sequence_comparison.json").write_text(
        json.dumps(per_seq, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
