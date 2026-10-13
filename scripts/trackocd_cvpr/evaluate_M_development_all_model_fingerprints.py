#!/usr/bin/env python3
"""Full frozen-M dev, fail closed until actual ALL11 support-hash parity.

Same original worker/all orders/ID/reverse/future probes, cohort and guards.
Only NEW exact Raw fingerprint scope and output namespace. Not training or
partial-order salvage; both full result/profile plus normal terminal required.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import evaluate_M_development as original
from src.trackocd_cvpr.M_all_model_fingerprint_development_profile import (
    all_model_configuration, verify_all_model_shadow, SOURCES, PROFILE)
from src.trackocd_cvpr.qualified_M_fingerprint_cache_scope import qualified_M_fingerprint_cache_scope
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/evaluate_M_development_all_model_fingerprints.py'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True); p.add_argument('--git-proxy', default='')
    p.add_argument('--model', choices=original.VARIANTS, required=True)
    p.add_argument('--seed', type=int, choices=original.SEEDS, required=True)
    mode = p.add_mutually_exclusive_group(); mode.add_argument('--worker', action='store_true'); mode.add_argument('--supervisor', action='store_true')
    args = p.parse_args()
    cfg = all_model_configuration(ROOT, original.configuration(ROOT, args), args)
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    original.SCRIPT = SCRIPT
    original.SOURCES = tuple(dict.fromkeys((*original.SOURCES, SCRIPT, *SOURCES)))
    original_gate, checked = original.source_gate, {}
    def gate(args, cfg, safety):
        result = original_gate(args, cfg, safety)  # fresh source/live registration/authority before private inputs
        checked.update(verify_all_model_shadow(ROOT))
        return result
    original.source_gate = gate
    if args.worker:
        with qualified_M_fingerprint_cache_scope() as (raw, fingerprints):
            original.worker(args, cfg, safety)
            if raw.retained_bytes > 33554432 or fingerprints.retained_bytes > 33554432 or not fingerprints.hits:
                raise ValueError('Actual weighted-M exact Raw hash profile within unchanged shared64MiB')
            out = ROOT / cfg['output_directory']
            atomic_json(out / 'execution_profile.json', {**checked, 'profile': PROFILE,
                'Raw_statistics_bytes': raw.retained_bytes, 'fingerprint_bytes': fingerprints.retained_bytes,
                'fingerprint_hits': fingerprints.hits, 'fingerprint_misses': fingerprints.misses,
                'result_sha256': sha256_file(out / 'result.json'),
                'original_data_models_orders_prefixes_thresholds_history_numeric_kernels_and_resource_guards_unchanged': True})
        return 0
    if args.supervisor: return original.supervise(args, cfg, safety)
    return original.launch(args, cfg, safety)


if __name__ == '__main__': sys.exit(main())
