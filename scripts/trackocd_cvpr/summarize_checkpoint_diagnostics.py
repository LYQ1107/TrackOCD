#!/usr/bin/env python3
"""Emit aggregate CSV views of a completed registered diagnostic, no model."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr.audit_historical_full import csv_write


def main():
    out = ROOT / 'outputs/trackocd_cvpr/cvpr0/checkpoint_replay_v1'
    d = json.loads((out / 'result.json').read_text())
    if d['status'] != 'COMPLETE_TRAIN_CHECKPOINT_SUBSTEP_NOT_COMPLETE_CVPR0':
        raise ValueError('Completed bounded checkpoint result required')
    rows = []
    for r in d['geometry']:
        rows.append({**{k: r[k] for k in ('representation', 'seed', 'partition', 'prefix', 'queries',
            'supported_queries', 'known_macro_rank1', 'pseudo_novel_macro_rank1', 'known_supported',
            'pseudo_novel_supported', 'mean_vector_norm_anisotropy', 'cross_video_positive_cosine_mean',
            'cross_video_negative_cosine_mean', 'hard_negative_cosine_mean', 'errors_corrected_vs_raw', 'new_errors_vs_raw')},
            'uncentered_effective_rank': r['spectra']['uncentered']['effective_rank'],
            'centered_effective_rank': r['spectra']['centered']['effective_rank'],
            'uncentered_top_eigenvalue_share': r['spectra']['uncentered']['top_eigenvalue_share']})
    csv_write(out / 'representation_geometry.csv', rows)
    rows = []
    for r in d['checkpoint_gradient_replays']:
        rows.append({**{k: r[k] for k in ('representation', 'seed', 'policy', 'prefix', 'wait_bias', 'delayed_events')},
            **{f'loss_{k}': v for k, v in r['loss_magnitudes'].items()},
            **{f'gradient_norm_{k}': v for k, v in r['gradient_norms'].items()},
            **{f'gradient_cosine_{k}': v for k, v in r['gradient_cosines'].items()},
            'interpretation': r['status'],
            'D1_cost_terms_not_used_in_original_training': r['D1_cost_gradients_are_hypothetical_not_original_contributions']})
    csv_write(out / 'checkpoint_gradient_conflict.csv', rows)
    print(json.dumps({'geometry_rows': len(d['geometry']), 'gradient_rows': len(rows), 'no_model_or_new_data': True}))


if __name__ == '__main__':
    main()
