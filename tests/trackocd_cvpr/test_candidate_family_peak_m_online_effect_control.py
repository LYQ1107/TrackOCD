import json

from scripts.trackocd_cvpr import candidate_family_peak_m_online_effect_control as control


def test_paired_online_m_effect_control_is_fixed_and_not_pred_training():
    config_path = control.ROOT / "configs/trackocd_cvpr/candidate_family_peak_m_online_effect_control_v1.json"
    output_path = control.ROOT / "outputs/trackocd_cvpr/cvpr2/candidate_family_peak_m_online_effect_control_v1/result.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert config["frames_per_video"] is None
    assert result["status"] == "PASS_FIXED_M_ONLINE_EFFECT_CONTROL"
    assert result["scope"] == {
        "videos": 2,
        "frozen_frames": 79,
        "online_frames": 79,
        "frozen_events": 215,
        "online_events": 215,
        "complete_fixed_clock": True,
    }
    assert result["arms"]["frozen_m"]["optimizer_steps"] == 0
    assert result["arms"]["online_m"]["optimizer_steps"] == 2
    assert result["arms"]["online_m"]["parameters_changed"] is True
    assert result["boundaries"]["predicted_track_training"] is False
    assert result["paired_effect"]["arm_selected"] is False
