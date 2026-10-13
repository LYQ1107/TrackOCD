#!/usr/bin/env python3
"""Exact original M evaluation driver with explicit qualified local cache scope.

Own process module constants select this wrapper as supervisor/worker command;
original source/control files unchanged. Full original data/clock/model/guards
and actualID/future probes retained; only outputnamespace/Raw reduction differ.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import evaluate_M_development as original
from src.trackocd_cvpr.M_cached_development_profile import cached_configuration, verify_actual_shadow, SOURCES
from src.trackocd_cvpr.qualified_raw_cache_scope import qualified_raw_cache_scope
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/evaluate_M_development_cached.py'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True); p.add_argument('--git-proxy', default='')
    p.add_argument('--model', choices=original.MODELS, required=True); p.add_argument('--seed', type=int, choices=original.SEEDS, required=True)
    mode = p.add_mutually_exclusive_group(); mode.add_argument('--worker', action='store_true'); mode.add_argument('--supervisor', action='store_true')
    args = p.parse_args()
    cfg = cached_configuration(ROOT, original.configuration(ROOT, args), args)
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    # Explicit per-process wrapper binding; never edits original files, model
    # kernels, another worker or systemwide module state.
    original.SCRIPT = SCRIPT
    original.SOURCES = tuple(dict.fromkeys((*original.SOURCES, SCRIPT, *SOURCES)))
    original_gate = original.source_gate
    checked = {}
    def qualified_gate(args, cfg, safety):
        result = original_gate(args, cfg, safety)
        checked.update(verify_actual_shadow(ROOT))
        return result
    original.source_gate = qualified_gate  # explicit PUBLIC qualifier after fresh source gate
    if args.worker:
        with qualified_raw_cache_scope() as cache:
            original.worker(args, cfg, safety)
            out = ROOT / cfg['output_directory']
            atomic_json(out / 'execution_profile.json', {**checked,
                'profile': 'EXPLICIT_EQUIVALENT_CPU_FROZEN_RAW_STATISTICS_CACHE',
                'retained_derived_cache_bytes': cache.retained_bytes,
                'result_sha256': sha256_file(out / 'result.json'),
                'original_data_models_orders_prefixes_thresholds_and_resource_guards_unchanged': True})
        return 0
    if args.supervisor: return original.supervise(args, cfg, safety)
    return original.launch(args, cfg, safety)


if __name__ == '__main__': sys.exit(main())
