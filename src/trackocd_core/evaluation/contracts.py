"""Sealed prediction ledger and strictly evaluator-only fixed GT joins.

Sealing accepts no GT rows, geometry matches, Novel labels or vocabulary.
Metrics can examine labels only after the prediction ledger is immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

PREFIXES = (1, 2, 4, 8, 16)
Kind = Literal["KNOWN", "NEW", "EXISTING", "WAIT"]
Role = Literal["known", "novel", "distractor"]


@dataclass(frozen=True, slots=True)
class TrackKey:
    video_id: int
    local_track_id: str

    def __post_init__(self):
        if type(self.video_id) is not int or type(self.local_track_id) is not str or not self.local_track_id:
            raise ValueError("Track identity must be (integer video_id, nonempty local string)")


@dataclass(frozen=True, slots=True)
class Target:
    key: TrackKey
    category_id: int
    role: Role

    def __post_init__(self):
        if not isinstance(self.key, TrackKey) or type(self.category_id) is not int:
            raise ValueError("GT key/category types must be explicit")


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    sequence: int
    physical_key: TrackKey
    observed_prefix: int
    kind: Kind
    known_category_id: int | None = None
    token: str | None = None

    def __post_init__(self):
        if (not isinstance(self.physical_key, TrackKey) or type(self.sequence) is not int
                or self.sequence < 0 or type(self.observed_prefix) is not int):
            raise ValueError("Decision identity/order/prefix types must be explicit integers")
        if self.known_category_id is not None and type(self.known_category_id) is not int:
            raise ValueError("Known ID must be an exact integer, not a float/bool alias")


@dataclass(frozen=True, slots=True)
class Commit:
    event: DecisionEvent
    prior_token_member_count: int


@dataclass(frozen=True, slots=True)
class SealedReplay:
    video_order: tuple[int, ...]
    prefix_cap: int
    known_ids: frozenset[int]
    events: tuple[DecisionEvent, ...]
    commits: tuple[Commit, ...]
    created_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationJoin:
    replay: SealedReplay
    targets: tuple[Target, ...]
    matches: tuple[tuple[TrackKey, TrackKey | None], ...]


def seal_decisions(events: Sequence[DecisionEvent], *, video_order: Sequence[int],
                   prefix_cap: int, known_ids: Sequence[int]) -> SealedReplay:
    """Reject future token access, ID aliasing and post-commit error erasure.

    WAIT may become a first commitment as observations/memory arrive. A semantic
    commitment is irrevocable within a replay; an identical confirmation is a
    no-op and never adds the member twice. Each order/prefix starts a new ledger.
"""
    order, known = tuple(video_order), frozenset(known_ids)
    if (type(prefix_cap) is not int or prefix_cap not in PREFIXES
            or any(type(v) is not int for v in order) or len(set(order)) != len(order)
            or any(type(k) is not int for k in known_ids)):
        raise ValueError("Invalid prefix cap or duplicate video in registered order")
    ranks = {v: i for i, v in enumerate(order)}
    members: dict[str, int] = {}
    committed: dict[TrackKey, Commit] = {}
    observed: dict[TrackKey, int] = {}
    previous_sequence = -1
    previous_rank = -1
    sealed_events = tuple(events)
    for event in sealed_events:
        if not isinstance(event.sequence, int) or event.sequence <= previous_sequence:
            raise ValueError("Decision sequence must be strictly increasing; no silent reordering")
        previous_sequence = event.sequence
        rank = ranks.get(event.physical_key.video_id)
        if rank is None or rank < previous_rank:
            raise ValueError("Decision videos violate the registered causal order")
        previous_rank = rank
        if (not 1 <= event.observed_prefix <= prefix_cap
                or event.observed_prefix < observed.get(event.physical_key, 0)):
            raise ValueError("Future or decreasing observation prefix")
        observed[event.physical_key] = event.observed_prefix
        if event.kind not in {"KNOWN", "NEW", "EXISTING", "WAIT"}:
            raise ValueError("Unknown decision kind")
        if event.kind == "KNOWN":
            if event.known_category_id not in known or event.token is not None:
                raise ValueError("KNOWN must name an allowed Known prototype, never an anonymous token")
        elif event.kind in {"NEW", "EXISTING"}:
            if not isinstance(event.token, str) or not event.token or event.known_category_id is not None:
                raise ValueError("Anonymous decision must name only a nonempty token")
        elif event.token is not None or event.known_category_id is not None:
            raise ValueError("WAIT cannot secretly assign a class/token")
        prior_commit = committed.get(event.physical_key)
        if prior_commit is not None:
            old = prior_commit.event
            if (event.kind, event.known_category_id, event.token) != (old.kind, old.known_category_id, old.token):
                raise ValueError("Cannot revise or hide an irrevocable semantic commitment")
            continue
        if event.kind == "WAIT":
            continue
        prior_member_count = 0
        if event.kind == "NEW":
            if event.token in members:
                raise ValueError("NEW cannot recycle an existing token")
            members[event.token] = 0
        elif event.kind == "EXISTING":
            if event.token not in members:
                raise ValueError("EXISTING references a token not yet created")
            prior_member_count = members[event.token]
        committed[event.physical_key] = Commit(event, prior_member_count)
        if event.kind in {"NEW", "EXISTING"}:
            members[event.token] += 1
    return SealedReplay(order, prefix_cap, known, sealed_events,
                        tuple(committed.values()), tuple(members))


def join_evaluation(replay: SealedReplay, targets: Sequence[Target],
                    matches: Mapping[TrackKey, TrackKey | None]) -> EvaluationJoin:
    """Posthoc one-to-one geometry join; missing GT targets remain present.

    All predicted identities that occur in the ledger must have an explicit
    geometry result (target or None). This function never calls a live policy.
    Distractor/unknown token members are retained for purity diagnostics.
"""
    universe = tuple(targets)
    by_key = {target.key: target for target in universe}
    if len(by_key) != len(universe):
        raise ValueError("Duplicate fixed GT target identity")
    ranks = set(replay.video_order)
    roles_by_category: dict[int, str] = {}
    for target in universe:
        if target.role not in {"known", "novel", "distractor"} or target.key.video_id not in ranks:
            raise ValueError("GT role/video outside registered evaluation universe")
        if target.role == "known" and target.category_id not in replay.known_ids:
            raise ValueError("Known GT category absent from registered Known list")
        if target.role != "known" and target.category_id in replay.known_ids:
            raise ValueError("Known/Novel/distractor roles overlap")
        if roles_by_category.setdefault(target.category_id, target.role) != target.role:
            raise ValueError("A category has conflicting frozen roles")
    copied_matches = dict(matches)
    assigned = set()
    for physical, gt_key in copied_matches.items():
        if physical.video_id not in ranks:
            raise ValueError("Predicted video outside registered order")
        if gt_key is not None:
            if gt_key not in by_key or gt_key.video_id != physical.video_id:
                raise ValueError("Geometry join outside fixed GT universe or video")
            if gt_key in assigned:
                raise ValueError("Geometry join must be one-to-one, not duplicate-count a GT")
            assigned.add(gt_key)
    if any(event.physical_key not in copied_matches for event in replay.events):
        raise ValueError("Every predicted event needs an explicit geometry join, including unmatched")
    return EvaluationJoin(replay, universe, tuple(copied_matches.items()))


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
