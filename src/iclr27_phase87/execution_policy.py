"""Phase87 fixed execution policy; no implicit route or threshold expansion."""
from __future__ import annotations

WINDOW_DEADLINE_UTC = "2026-09-07T03:53:10.043296+00:00"
FOLD_GPU = {0: 5, 1: 6, 2: 7, 3: 8}
FORMAL_GATE = {
    "commit_ct_min": 15,
    "category_coverage_min": 5,
    "video_coverage_min": 8,
    "existing_precision_min": 0.70,
    "negative_false_merge_max": 0.15,
    "known_micro_min": 0.206,
    "known_macro_min": 0.139,
}
ROUTES = ("C0_CAUSAL_PERSISTENT_CONTROLLER", "C0_FALSE_MERGE_REPAIR1", "C1_SUPPORT_INTEGRATION")


def sealed_inputs_forbidden() -> bool:
    return True
