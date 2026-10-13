#!/usr/bin/env python3
"""Same complete frozen-M dev; refuses pending actual input-memo qualification."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr import evaluate_M_development as original
from src.trackocd_cvpr.M_member_input_development_profile import configuration, verify_member_input_shadow, SOURCES, PROFILE
from src.trackocd_cvpr.frozen_member_feature_cache_scope import frozen_member_feature_cache_scope
from src.trackocd_v2.io import atomic_json, sha256_file

SCRIPT = 'scripts/trackocd_cvpr/evaluate_M_development_member_input_memo.py'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preregistration-commit', required=True); p.add_argument('--git-proxy', default='')
    p.add_argument('--model', choices=original.VARIANTS, required=True)
    p.add_argument('--seed', type=int, choices=original.SEEDS, required=True)
    g = p.add_mutually_exclusive_group(); g.add_argument('--worker', action='store_true'); g.add_argument('--supervisor', action='store_true')
    args = p.parse_args()
    cfg = configuration(ROOT, original.configuration(ROOT, args), args)
    safety = json.loads((ROOT / 'configs/trackocd_cvpr/autonomous_research.json').read_text())
    original.SCRIPT = SCRIPT
    original.SOURCES = tuple(dict.fromkeys((*original.SOURCES, SCRIPT, *SOURCES)))
    original_gate, checked = original.source_gate, {}

    def gate(args, cfg, safety):
        checked.update(verify_member_input_shadow(ROOT))  # Public missing/partial proof refuses first.
        return original_gate(args, cfg, safety)  # Still fresh exact remote/source before any private input.

    original.source_gate = gate
    if args.worker:
        with frozen_member_feature_cache_scope() as (raw, sha, features, index):
            original.worker(args, cfg, safety)  # Three FULL orders plus original FULL ID/reverse/future probes.
            if (raw.retained_bytes > 33554432 or sha.retained_bytes > 16252928
                    or features.retained_bytes > 16777216 or index.peak_bytes > 524288
                    or index.snapshot is not None or index.retained_bytes or not features.hits or not index.hits):
                raise ValueError('Actual same64MiB input-only memo/current-refresh index used')
            out = ROOT / cfg['output_directory']
            atomic_json(out / 'execution_profile.json', {**checked, 'profile': PROFILE,
                'Raw_statistics_bytes': raw.retained_bytes, 'fingerprint_bytes': sha.retained_bytes,
                'fingerprint_hits': sha.hits, 'fingerprint_misses': sha.misses,
                'member_input_bytes': features.retained_bytes, 'member_input_hits': features.hits,
                'member_input_misses': features.misses, 'temporary_index_hits': index.hits,
                'temporary_index_peak_bookkeeping_bytes': index.peak_bytes,
                'result_sha256': sha256_file(out / 'result.json'),
                'original_models_orders_prefixes_thresholds_history_probes_and_guards_unchanged': True,
                'learned_outputs_or_partial_metrics_cached': False})
        return 0
    if args.supervisor: return original.supervise(args, cfg, safety)
    return original.launch(args, cfg, safety)


if __name__ == '__main__':
    sys.exit(main())
