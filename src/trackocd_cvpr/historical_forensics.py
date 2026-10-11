"""Offline, evaluator-only forensics of already sealed decisions.

This module is not imported by any live policy. Unknown physical matches are
explicitly not counted as proven pollution. No retrospective repair is made.
"""
from collections import Counter
from dataclasses import dataclass, field

KINDS = ("KNOWN", "NEW", "EXISTING", "WAIT")  # Historical Parquet encoding.


@dataclass
class History:
    categories: set = field(default_factory=set)
    videos: set = field(default_factory=set)
    unknown: bool = False
    mixed: bool = False
    writes: int = 0
    first_bad: int | None = None


class SealedForensics:
    """Consumes events after sealing; labels never enter prediction state."""

    def __init__(self, targets, expected_events, bins=16):
        self.targets = targets
        self.expected_events = expected_events
        self.counts = Counter()
        self.temporal = [Counter() for _ in range(bins)]
        self.histories = {}
        self.anonymous_categories = set()
        self.first_bad_write = None
        self.first_proven_contamination = None
        self.events = 0

    def consume(self, video, physical, kind, known, token):
        if kind not in range(len(KINDS)):
            raise ValueError("Invalid sealed action")
        action = KINDS[kind]
        target = self.targets.get((int(video), int(physical)))
        role, category = target if target is not None else ("unknown", None)
        if role not in {"known", "novel", "unknown"}:
            raise ValueError("Unexpected target role")
        self.counts[action] += 1
        self.counts[f"{role}_{action}"] += 1
        bin_id = min(self.events * len(self.temporal) // self.expected_events, len(self.temporal) - 1)
        self.temporal[bin_id][action] += 1
        self.temporal[bin_id][f"{role}_{action}"] += 1
        if action == "KNOWN":
            self.counts["wrong_known"] += target is not None and (role != "known" or known != category)
            self.counts["correct_known"] += role == "known" and known == category
        if action in {"NEW", "EXISTING"}:
            if token < 0:
                raise ValueError("Missing anonymous token")
            if action == "NEW":
                if token in self.histories:
                    raise ValueError("Recycled anonymous identity")
                self.counts["duplicate_birth"] += role == "novel" and category in self.anonymous_categories
                self.histories[token] = History()
            elif token not in self.histories:
                raise ValueError("Absent/future EXISTING token")
            h = self.histories[token]
            before_mixed = h.mixed
            conflicting = target is not None and bool(h.categories - {category})
            wrong_existing = action == "EXISTING" and target is not None and (role == "known" or conflicting)
            bad_write = role == "known" or conflicting
            self.counts["wrong_existing"] += wrong_existing
            self.counts["unverified_existing"] += action == "EXISTING" and h.unknown
            self.counts["reuse_after_proven_contamination"] += action == "EXISTING" and before_mixed
            self.counts["proven_contamination_write_events"] += conflicting
            self.counts["unknown_anonymous_writes"] += target is None
            if bad_write and self.first_bad_write is None:
                self.first_bad_write = self.events
            if conflicting and self.first_proven_contamination is None:
                self.first_proven_contamination = self.events
            if bad_write and h.first_bad is None:
                h.first_bad = self.events
            if target is None:
                h.unknown = True
            else:
                h.categories.add(category)
                self.anonymous_categories.add(category)
            h.mixed = len(h.categories) > 1
            h.videos.add(video)
            h.writes += 1
        self.events += 1

    def finish(self):
        if self.events != self.expected_events:
            raise ValueError("Truncated or expanded sealed event universe")
        return {
            "events": self.events,
            "counts": dict(self.counts),
            "temporal_bins": [dict(c) for c in self.temporal],
            "anonymous_states": len(self.histories),
            "mixed_states": sum(h.mixed for h in self.histories.values()),
            "states_with_unknown_members": sum(h.unknown for h in self.histories.values()),
            "first_bad_write_sequence": self.first_bad_write,
            "first_proven_contamination_sequence": self.first_proven_contamination,
            "bad_write_definition": "matched Known assigned anonymously or matched label conflicts with prior matched member",
            "unknown_members_are_not_proven_pollution": True,
            "logits_and_candidate_ranks": "NOT_RETAINED_IN_HISTORICAL_LEDGER",
            "first_error_intervention_performed": False,
        }
