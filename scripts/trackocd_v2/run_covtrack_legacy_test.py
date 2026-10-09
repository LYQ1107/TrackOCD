#!/usr/bin/env python3
"""Launch the pinned COVTrack test script with its optional BDD shim.

The COVTrack repository imports its BDD evaluator at module import time even
for TAO evaluation.  The registered legacy environment does not ship the
optional ``bdd100k`` package, and the TAO route never uses that evaluator.
Install only the two import-time symbols as a child-process shim, then execute
the unchanged upstream script with ``runpy``.  If a BDD route is ever selected,
the explicit stub raises instead of silently producing a result.
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
import runpy
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COVTRACK_TEST = ROOT / "third_party/research_refs_phase4n/COVTrack/tools/test.py"
CLIP_PACKAGE = Path(
    "/home/lwr/anaconda3/envs/ovtrack/lib/python3.11/site-packages/clip"
)


def _unsupported_bdd(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError("the optional bdd100k evaluator is not part of the TAO COVTrack route")


def _install_bdd_import_shim() -> None:
    bdd100k = types.ModuleType("bdd100k")
    common = types.ModuleType("bdd100k.common")
    common_utils = types.ModuleType("bdd100k.common.utils")
    common_utils.load_bdd100k_config = _unsupported_bdd
    label = types.ModuleType("bdd100k.label")
    to_scalabel = types.ModuleType("bdd100k.label.to_scalabel")
    to_scalabel.bdd100k_to_scalabel = _unsupported_bdd
    sys.modules.update({
        "bdd100k": bdd100k,
        "bdd100k.common": common,
        "bdd100k.common.utils": common_utils,
        "bdd100k.label": label,
        "bdd100k.label.to_scalabel": to_scalabel,
    })
    scalabel = types.ModuleType("scalabel")
    scalabel_eval = types.ModuleType("scalabel.eval")
    scalabel_mot = types.ModuleType("scalabel.eval.mot")
    scalabel_mot.acc_single_video_mot = _unsupported_bdd
    scalabel_mot.evaluate_track = _unsupported_bdd
    scalabel_label = types.ModuleType("scalabel.label")
    scalabel_io = types.ModuleType("scalabel.label.io")
    scalabel_io.group_and_sort = _unsupported_bdd
    scalabel_io.load = _unsupported_bdd
    sys.modules.update({
        "scalabel": scalabel,
        "scalabel.eval": scalabel_eval,
        "scalabel.eval.mot": scalabel_mot,
        "scalabel.label": scalabel_label,
        "scalabel.label.io": scalabel_io,
    })
    # The upstream offline TETA helper is imported by the dataset package but
    # is not reached by the TAO ``--format-only`` route.  The compatible
    # environment intentionally keeps TETA separate; a real TETA call must
    # use the dedicated v2 reference runner instead of this shim.
    sys.modules.setdefault("teta", types.ModuleType("teta"))


def _install_clip_import_shim() -> None:
    """Load existing CLIP without exposing its whole donor environment."""

    if "clip" in sys.modules:
        return
    init_file = CLIP_PACKAGE / "__init__.py"
    if not init_file.is_file():
        raise ModuleNotFoundError(
            f"COVTrack requires clip; expected existing package at {init_file}"
        )
    spec = importlib.util.spec_from_file_location(
        "clip",
        init_file,
        submodule_search_locations=[str(CLIP_PACKAGE)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load CLIP package from {CLIP_PACKAGE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["clip"] = module
    spec.loader.exec_module(module)


def _install_sklearn_import_shim() -> None:
    """Expose explicit stubs for unused optional clustering helpers."""

    def unsupported(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(
            "the optional sklearn clustering helpers are not part of the TAO "
            "COVTrack --format-only route"
        )

    class UnsupportedCluster:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def fit(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError(
                "the optional sklearn clustering helpers are not part of the "
                "TAO COVTrack --format-only route"
            )

    sklearn = types.ModuleType("sklearn")
    cluster = types.ModuleType("sklearn.cluster")
    cluster.KMeans = UnsupportedCluster
    cluster.DBSCAN = UnsupportedCluster
    metrics = types.ModuleType("sklearn.metrics")
    metrics.adjusted_rand_score = unsupported
    metrics.normalized_mutual_info_score = unsupported
    sys.modules.update({
        "sklearn": sklearn,
        "sklearn.cluster": cluster,
        "sklearn.metrics": metrics,
    })


def _install_lap_import_shim() -> None:
    """Provide the missing ``lapjv`` API using SciPy's exact assignment."""

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def lapjv(
        cost_matrix: object,
        extend_cost: bool = False,
        cost_limit: float = np.inf,
        *_args: object,
        **kwargs: object,
    ) -> object:
        costs = np.asarray(cost_matrix, dtype=float)
        if costs.ndim != 2:
            raise ValueError("lapjv expects a two-dimensional cost matrix")
        rows, cols = costs.shape
        row_assignment = np.full(rows, -1, dtype=int)
        col_assignment = np.full(cols, -1, dtype=int)
        if rows == 0 or cols == 0:
            result = (0.0, row_assignment, col_assignment)
            return result if kwargs.get("return_cost", True) else result[1:]

        finite = np.isfinite(costs)
        if not finite.any():
            result = (0.0, row_assignment, col_assignment)
            return result if kwargs.get("return_cost", True) else result[1:]
        finite_values = costs[finite]
        penalty = max(
            float(np.max(finite_values)),
            float(cost_limit) if np.isfinite(cost_limit) else 0.0,
            0.0,
        ) + max(float(np.ptp(finite_values)), 1.0) * 4.0
        solver_costs = np.where(finite, costs, penalty)
        assigned_rows, assigned_cols = linear_sum_assignment(solver_costs)
        selected = costs[assigned_rows, assigned_cols]
        accepted = np.isfinite(selected) & (selected <= cost_limit)
        row_assignment[assigned_rows[accepted]] = assigned_cols[accepted]
        col_assignment[assigned_cols[accepted]] = assigned_rows[accepted]
        total_cost = float(np.sum(selected[accepted]))
        result = (total_cost, row_assignment, col_assignment)
        return result if kwargs.get("return_cost", True) else result[1:]

    lap = types.ModuleType("lap")
    lap.lapjv = lapjv
    lap.__spec__ = ModuleSpec("lap", loader=None)
    sys.modules["lap"] = lap


def main() -> None:
    if not COVTRACK_TEST.is_file():
        raise FileNotFoundError(COVTRACK_TEST)
    _install_bdd_import_shim()
    _install_clip_import_shim()
    _install_sklearn_import_shim()
    _install_lap_import_shim()
    runpy.run_path(str(COVTRACK_TEST.resolve()), run_name="__main__")


if __name__ == "__main__":
    main()
