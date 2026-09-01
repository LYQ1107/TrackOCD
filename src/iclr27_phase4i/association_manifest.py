"""Association-code manifest for the Phase 4I tracker audit."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / "association_code_manifest.csv"

ROWS = [
    ("detector_output", "IDOL.track_eval",
     "third_party/SimOWT/projects/IDOL/idol/idol.py",
     "box_pred/mask_pred/reid_pred produced by frozen DETR detector",
     "frozen (Phase 3A export)"),
    ("pre_assoc_filter", "IDOL.track_eval",
     "third_party/SimOWT/projects/IDOL/idol/idol.py",
     "score threshold inference_select_thres + batched NMS 0.9 before association",
     "frozen (pre_assoc_detections)"),
    ("pre_assoc_export", "IDOL.track_eval",
     "third_party/SimOWT/projects/IDOL/idol/idol.py",
     "SIMOWT_EXPORT_DIR writes det_bboxes/labels/masks/track_feats/indices per frame",
     "frozen (replay_packages)"),
    ("mask_nms_pre", "IDOL_Tracker.match -> mask_nms",
     "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
     "mask IoU NMS before association (nms_thr_pre=0.5)",
     "frozen for B0/B1; kept for B2"),
    ("appearance_matrix", "IDOL_Tracker.match",
     "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
     "feats @ memo_embeds.T then bisoftmax (d2t+t2d)/2",
     "B0/B1 identical; B2 adds semantic term"),
    ("frame_weight", "IDOL_Tracker.match",
     "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
     "multiply scores by memo_exist_frame for candidates >0.5",
     "frozen"),
    ("greedy_assignment", "IDOL_Tracker.match",
     "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
     "per detection max score; assign if > match_score_thr; zero assigned memo column",
     "frozen"),
    ("new_track_birth", "IDOL_Tracker.match",
     "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
     "unmatched ids==-2 with bbox score > addnew_score_thr become new tracklets",
     "frozen"),
    ("mask_nms_post", "IDOL_Tracker.match -> mask_iou",
     "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
     "unmatched low-score detections become -1 backdrops unless mask IoU overlap",
     "frozen"),
    ("memo_update", "IDOL_Tracker.update_memo",
     "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
     "EMA embed (memo_momentum=0.8), velocity, exist_frame, long_embed history len 3",
     "frozen for B0/B1; prefix state added for semantics (does not change physical state)"),
    ("memo_read", "IDOL_Tracker.memo",
     "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
     "long_match=True: score-weighted sum of long_embed history",
     "frozen"),
    ("track_lifecycle", "IDOL_Tracker.update_memo",
     "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
     "drop tracklets unseen for memo_tracklet_frames=10; backdrops limited",
     "frozen"),
    ("motion_state", "IDOL_Tracker.update_memo",
     "third_party/SimOWT/projects/IDOL/idol/models/tracker.py",
     "velocity stored per tracklet but NOT used in match scores",
     "frozen (audit finding)"),
    ("output_writer", "IDOL.track_eval",
     "third_party/SimOWT/projects/IDOL/idol/idol.py",
     "per-image JSON with bbox/track_id/score/mask",
     "frozen for B0/B1/B2 output"),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "function", "file", "behavior", "phase4i_status"])
        w.writerows(ROWS)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
