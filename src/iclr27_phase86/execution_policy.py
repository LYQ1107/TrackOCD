from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase86"
REGISTRATION = OUT / "audit/window_registration.json"
HARD_BLOCKER = OUT / "audit/hard_blocker.json"
FINALIZATION_LEAD_SECONDS = 45 * 60
ALLOWED_HARD_BLOCKERS = {
    "DATA_UNAVAILABLE",
    "PROTOCOL_IMPOSSIBILITY",
    "SAFETY_BOUNDARY",
    "RESOURCE_UNRECOVERABLE",
}

def parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))

def load_registration() -> dict:
    return json.loads(REGISTRATION.read_text(encoding="utf-8"))

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def deadline_utc() -> dt.datetime:
    return parse_utc(load_registration()["deadline_utc"])

def remaining_seconds() -> float:
    return (deadline_utc() - now_utc()).total_seconds()

def hard_blocker_active() -> bool:
    if not HARD_BLOCKER.exists():
        return False
    obj = json.loads(HARD_BLOCKER.read_text(encoding="utf-8"))
    return obj.get("active") is True and obj.get("reason") in ALLOWED_HARD_BLOCKERS

def finalization_allowed() -> bool:
    return remaining_seconds() <= FINALIZATION_LEAD_SECONDS or hard_blocker_active()

def assert_finalization_allowed() -> None:
    if not finalization_allowed():
        raise RuntimeError(
            "FINALIZATION_TOO_EARLY_RESEARCH_MUST_CONTINUE "
            f"remaining_seconds={remaining_seconds():.0f}"
        )
