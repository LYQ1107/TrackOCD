"""Shared state-machine transitions for training and runtime."""
from __future__ import annotations

from typing import Any

from src.iclr27_phase88.memory import StateMemoryV2, TargetSession


def apply_session_action(session: TargetSession, action: str,
                         state_index: int | None = None,
                         known_index: int | None = None,
                         global_state_index: int | None = None) -> None:
    if action == "EXISTING":
        session.committed_action = "EXISTING"
        session.committed_state_index = state_index
        session.committed_global_index = global_state_index
        session.committed_known_index = None
        session.provisional_new = False
    elif action == "NEW":
        session.committed_action = "NEW"
        session.committed_state_index = None
        session.committed_global_index = None
        session.committed_known_index = None
        session.provisional_new = True
    elif action == "KNOWN":
        session.committed_action = "KNOWN"
        session.committed_known_index = known_index
        session.committed_state_index = None
        session.committed_global_index = None
        session.provisional_new = False
    elif action == "RESET":
        session.reset(reason=session.reset_reason or "wrong_existing")
    elif action == "DEFER":
        return
    else:
        raise ValueError(f"unknown action {action}")


def finalize_session(memory: StateMemoryV2, session: TargetSession,
                     final_vector, observations: int,
                     oracle_category: int | None = None) -> None:
    memory.finalize_track(session, final_vector, observations, oracle_category)
