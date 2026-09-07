"""Balanced target-track/event sampler for Phase88 TRAIN."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


class BalancedCausalEventSampler:
    def __init__(self, events: list[dict[str, Any]], seed: int = 88001) -> None:
        self.events = list(events)
        self.rng = random.Random(seed)
        self.by_polarity = {
            "positive": [e for e in self.events if e.get("polarity") == "positive"],
            "negative": [e for e in self.events if e.get("polarity") == "negative"],
        }
        self.by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in self.events:
            self.by_target[str(event["target_track_key"])].append(event)
        self.targets = sorted(self.by_target)

    def sample(self) -> dict[str, Any]:
        polarity = "positive" if self.rng.random() < 0.5 else "negative"
        pool = self.by_polarity[polarity] or self.events
        if not pool:
            raise RuntimeError("empty Phase88 event sampler")
        target = self.rng.choice(self.targets)
        choices = [e for e in self.by_target[target] if e.get("polarity") == polarity]
        return self.rng.choice(choices or pool)

    def stats(self) -> dict[str, int]:
        return {
            "events": len(self.events),
            "targets": len(self.targets),
            "positive": len(self.by_polarity["positive"]),
            "negative": len(self.by_polarity["negative"]),
        }
