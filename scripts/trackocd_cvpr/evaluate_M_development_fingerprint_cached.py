#!/usr/bin/env python3
"""Same full M evaluation, new qualified exact fingerprint cache namespace."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import evaluate_M_development as original
from src.trackocd_cvpr.M_fingerprint_development_profile import fingerprint_configuration, verify_fingerprint_shadow, SOURCES
from src.trackocd_cvpr.qualified_fingerprint_cache_scope import qualified_fingerprint_cache_scope
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/evaluate_M_development_fingerprint_cached.py'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--preregistration-commit', required=True); p.add_argument('--git-proxy', default='')
    p.add_argument('--model', choices=original.MODELS, required=True); p.add_argument('--seed', type=int, choices=original.SEEDS, required=True)
    mode = p.add_mutually_exclusive_group(); mode.add_argument('--worker', action='store_true'); mode.add_argument('--supervisor', action='store_true')
    args = p.parse_args()
    cfg = fingerprint_configuration(ROOT, original.configuration(ROOT, args), args)
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    original.SCRIPT = SCRIPT
    original.SOURCES = tuple(dict.fromkeys((*original.SOURCES, SCRIPT, *SOURCES)))
    original_gate, checked = original.source_gate, {}
    def gate(args, cfg, safety):
        result = original_gate(args, cfg, safety)  # fresh remote/source/authority first
        checked.update(verify_fingerprint_shadow(ROOT))
        return result
    original.source_gate = gate
    if args.worker:
        with qualified_fingerprint_cache_scope() as (raw, fingerprints):
            original.worker(args, cfg, safety)
            if raw.retained_bytes + fingerprints.retained_bytes > 67108864:
                raise ValueError('Original combined derived-cache ceiling')
            out = ROOT / cfg['output_directory']
            atomic_json(out / 'execution_profile.json', {**checked,
                'profile': 'EXACT_BYTE_SEGMENT_SHA_MEMO_32MiB_AND_FROZEN_RAW_32MiB_TOTAL_64MiB',
                'Raw_statistics_bytes': raw.retained_bytes, 'fingerprint_bytes': fingerprints.retained_bytes,
                'fingerprint_hits': fingerprints.hits, 'fingerprint_misses': fingerprints.misses,
                'result_sha256': sha256_file(out / 'result.json'),
                'original_data_models_orders_prefixes_thresholds_and_resource_guards_unchanged': True})
        return 0
    if args.supervisor: return original.supervise(args, cfg, safety)
    return original.launch(args, cfg, safety)


if __name__ == '__main__': sys.exit(main())
