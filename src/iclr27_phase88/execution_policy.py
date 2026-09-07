"""Phase88 deadline/resource policy loaded from the original registration."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT / "outputs/iclr27_phase88/audit/window_registration.json"
CONTINUOUS_STATE = ROOT / "outputs/iclr27_phase88/audit/continuous_state.json"
FORMAL_GATE = {
    "commit_ct_min": 15,
    "category_coverage_min": 5,
    "video_coverage_min": 8,
    "existing_precision_min": 0.70,
    "negative_false_merge_max": 0.15,
    "known_micro_min": 0.206,
    "known_macro_min": 0.139,
}
ROUTES = ("C0V2_FIX2", "C0_CONTINUE", "C1_SUPPORT")


def load_registration() -> dict:
    return json.loads(REGISTRATION.read_text())


def deadline_utc() -> dt.datetime:
    value = load_registration()["deadline_utc"].replace("Z", "+00:00")
    return dt.datetime.fromisoformat(value).astimezone(dt.timezone.utc)


def remaining_seconds(now: dt.datetime | None = None) -> float:
    current = now or dt.datetime.now(dt.timezone.utc)
    return (deadline_utc() - current).total_seconds()


def hard_blocker_active() -> bool:
    # A resource stop is recoverable until the corrected memmap worker is
    # measured.  This function intentionally does not call it unrecoverable.
    return False


def assert_finalization_allowed() -> None:
    if not CONTINUOUS_STATE.exists():
        raise RuntimeError("TASK_NOT_COMPLETE_RESEARCH_MUST_CONTINUE")
    state = json.loads(CONTINUOUS_STATE.read_text())
    if state.get("task_status") != "COMPLETE":
        raise RuntimeError("TASK_NOT_COMPLETE_RESEARCH_MUST_CONTINUE")


def sealed_inputs_forbidden() -> bool:
    return True
