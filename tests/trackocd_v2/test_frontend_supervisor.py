import json

from scripts.trackocd_v2 import autonomous_supervisor as supervisor
from src.trackocd_v2.protocol import STAGES


def test_incomplete_frontend_bakeoff_rewinds_before_selection(monkeypatch, tmp_path):
    output = tmp_path / "project_outputs"
    audit = output / "audit"
    audit.mkdir(parents=True)
    monkeypatch.setattr(supervisor, "OUTPUT_TARGET", output)
    monkeypatch.setattr(supervisor, "STATE_PATH", audit / "autonomous_state.json")

    for slug in ("simowt", "ovtr", "covtrack_native", "covtrack_nosem"):
        (audit / f"frontend_{slug}.json").write_text(
            json.dumps({"status": "NATIVE_STREAM_NORMALIZED"}), encoding="utf-8"
        )

    frontend_start = STAGES.index("FRONTEND_AUDIT")
    completed = list(STAGES[:frontend_start]) + [
        "FRONTEND_AUDIT", "FRONTEND_SIMOWT", "FRONTEND_OVTR",
        "FRONTEND_COVTRACK_NATIVE", "FRONTEND_COVTRACK_NOSEM", "FRONTEND_SELECTION",
    ]
    state = {
        "state": "REPRESENTATION_DECISION",
        "status": "WAITING_FRONTEND_EXECUTION",
        "completed_stages": completed,
        "pending_stages": [stage for stage in STAGES if stage not in completed],
        "artifacts": {},
    }

    assert supervisor._rewind_incomplete_frontend_bakeoff(state) is True
    assert state["state"] == "FRONTEND_SIMOWT"
    assert state["status"] == "RUNNABLE"
    assert "FRONTEND_SIMOWT" not in state["completed_stages"]
    assert "FRONTEND_SELECTION" not in state["completed_stages"]
    assert state["pending_stages"][0] == "FRONTEND_SIMOWT"
    written = json.loads((audit / "autonomous_state.json").read_text(encoding="utf-8"))
    assert written["state"] == "FRONTEND_SIMOWT"


def test_idle_gpu_indices_uses_uuid_occupancy():
    preflight = {
        "resources": {
            "idle_gpu_count": 1,
            "gpu_rows": ["0, GPU-0, A100, 100, 100, 0", "1, GPU-1, A100, 100, 100, 0"],
            "compute_apps": [{"gpu_uuid": "GPU-0", "pid": 7}],
        }
    }
    assert supervisor._idle_gpu_indices(preflight) == [1]
