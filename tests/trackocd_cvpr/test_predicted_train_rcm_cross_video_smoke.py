import json
from pathlib import Path


def test_cross_video_predicted_rcm_smoke_is_lifecycle_only():
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads(
        (
            root
            / "outputs/trackocd_cvpr/cvpr2/predicted_train_rcm_cross_video_smoke_v1/result.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "PASS_CATEGORY_FREE_CROSS_VIDEO_RCM_LIFECYCLE_SMOKE_NO_TRAINING"
    assert receipt["lifecycle"]["videos"] == 2
    assert receipt["lifecycle"]["cross_video_token_count"] == 3
    assert receipt["lifecycle"]["optimizer_steps"] == 0
    assert receipt["boundaries"]["labels_or_GT_read"] is False
    assert receipt["boundaries"]["scientific_acceptance"] is False
