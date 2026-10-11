#!/usr/bin/env python3
"""Public aggregate table from an actual receipt; no raw predictions or labels."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_cvpr.audit_historical_full import csv_write


def summarize(result):
    if len(result['cases']) != 45 or len({(c['representation'], c['seed'], c['prefix'])
                                        for c in result['cases']}) != 45:
        raise ValueError('Actual finite registered cases required')
    rows = []
    for case in result['cases']:
        for side, count in (('predicted', 16933), ('GT_controlled', 258)):
            item = case[side]
            standard = item['evaluation']['standard']
            persistent = item['evaluation']['persistent']
            if (standard['old_denominator'], standard['new_denominator'],
                persistent['commit_ct_denominator']) != (249, 9, 5):
                raise ValueError('Full fixed selected-reference denominators required')
            if sum(item['action_counts'].values()) != count:
                raise ValueError('All input tracks required')
            rows.append({**{k: case[k] for k in ('representation', 'seed', 'prefix')},
                'input': side, 'input_tracks': count,
                'Old': standard['old_acc'], 'New': standard['new_acc'],
                'H': standard['h_score'], 'correct_commit_CT': persistent['correct_commit_ct'],
                'reference_non_wait_coverage': standard['non_wait_coverage'],
                'reference_prediction_coverage': standard['prediction_coverage'],
                'reuse_effective_commit_coverage': persistent['effective_commit_coverage'],
                'input_commit_coverage': item['runtime']['commitment_coverage'],
                'Known_denominator': 249, 'pseudo_Novel_denominator': 9, 'CT_denominator': 5,
                **{a: item['action_counts'].get(a, 0) for a in ('KNOWN', 'NEW', 'EXISTING', 'WAIT')},
                'selected_capped_reference_only': True, 'main_acceptance': False})
    return rows


if __name__ == '__main__':
    directory = ROOT / 'outputs/trackocd_cvpr/cvpr0/matched_train_semantics_v1'
    rows = summarize(json.loads((directory / 'result.json').read_text()))
    csv_write(directory / 'summary.csv', rows)
    print(json.dumps({'actual_aggregate_rows': len(rows), 'main_acceptance': False}))
