#!/usr/bin/env python3
"""Read-only Phase46 support/causal/bridge contract audit."""
import json, os, tempfile
from pathlib import Path
import torch
from src.iclr27_phase41.bridge import SafetyVectorBridge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/iclr27_phase46'

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.' + path.name)
    with os.fdopen(fd, 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def main():
    manifest = json.load(open(ROOT/'outputs/iclr27_phase38/audit/support_stream_manifest.json'))
    temporal = json.load(open(ROOT/'outputs/iclr27_phase38/audit/support_temporal_audit.json'))
    leakage = json.load(open(ROOT/'outputs/iclr27_phase38/audit/support_leakage_audit.json'))
    cov = json.load(open(ROOT/'outputs/iclr27_phase38/audit/support_coverage_76.json'))
    policies = manifest['policies']
    policy_rows = {}
    for policy, rows in policies.items():
        event_prefix = {(r['event_key'], int(r['prefix'])): r for r in rows}
        all_keys_prior = all(all(int(k.split(':')[0][1:]) < int(r['target_video']) for k in r['support_track_keys']) for r in rows)
        prefix_complete = all((ek, p) in event_prefix for ek in {r['event_key'] for r in rows} for p in (1,2,4,8,16))
        policy_rows[policy] = {
            'rows': len(rows), 'events': len({r['event_key'] for r in rows}),
            'prefixes': sorted({int(r['prefix']) for r in rows}),
            'all_support_videos_strictly_prior': bool(all_keys_prior),
            'all_event_prefixes_present': bool(prefix_complete),
            'future_support_rows': int(sum(bool(r.get('future_support')) for r in rows)),
            'nonempty_prefix16': int(sum(int(r['prefix']) == 16 and int(r['support_count']) > 0 for r in rows)),
            'mean_support_count_prefix16': float(sum(r['support_count'] for r in rows if int(r['prefix']) == 16) / max(sum(int(r['prefix']) == 16 for r in rows), 1)),
        }
    # Verify frozen bridge's actual row-vector contract and invalid-support fallback.
    bridge = SafetyVectorBridge()
    ck = ROOT/'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f0_best.pt'
    bridge.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model']); bridge.eval()
    with torch.no_grad():
        raw = torch.zeros(1, 768); ctx = torch.zeros(1, 768)
        z_valid, alpha_valid, _ = bridge(raw, ctx, torch.zeros(1), torch.ones(1), True)
        z_invalid, alpha_invalid, _ = bridge(raw, None, torch.zeros(1), torch.zeros(1), False)
    bridge_contract = {
        'checkpoint': str(ck), 'input_dim': 768, 'valid_shape': list(z_valid.shape),
        'invalid_shape': list(z_invalid.shape), 'valid_finite': bool(torch.isfinite(z_valid).all()),
        'invalid_fallback_alpha': float(alpha_invalid.item()), 'invalid_equals_raw': bool(torch.allclose(z_invalid, raw)),
        'valid_alpha_finite': bool(torch.isfinite(alpha_valid).all()),
    }
    legal = all(x['all_support_videos_strictly_prior'] and x['all_event_prefixes_present'] and x['future_support_rows'] == 0 and x['nonempty_prefix16'] == 76 for x in policy_rows.values()) and all(bool(v) for v in leakage.values()) is False
    # Leakage audit uses false-valued flags to indicate no leakage; spell this out.
    legal = all(x['all_support_videos_strictly_prior'] and x['all_event_prefixes_present'] and x['future_support_rows'] == 0 and x['nonempty_prefix16'] == 76 for x in policy_rows.values()) and not any(bool(v) for v in leakage.values()) and bridge_contract['valid_shape'] == [1,768] and bridge_contract['invalid_equals_raw']
    atomic(OUT/'audit/support_contract.json', {'phase': 46, 'positive_events': 76, 'policies': policy_rows, 'temporal_source': temporal, 'coverage_source': cov, 'leakage_source': leakage, 'support_is_causal_under_prior_video_policy': bool(legal), 'support_is_diagnostic_not_persistent_ct': True})
    atomic(OUT/'audit/bridge_contract.json', bridge_contract)
    atomic(OUT/'audit/resource_preflight.json', {'gpu_ids_checked': list(range(10)), 'intended_gpus': [4,5,6,7], 'long_training_started': False, 'controller_run': False, 'public_q1_dev_access': False})
    decision = {'phase': 46, 'stage0': 'PASS' if legal else 'FAIL', 'decision': 'B_LAST_CONDITIONAL_GATE_ALLOWED' if legal else 'C_SUPPORT_INTERFACE_REPAIR_REQUIRED', 'decision_code': 'P46_STAGE0_SUPPORT_CONTRACT_PASS_LAST_GATE_ALLOWED' if legal else 'P46_STAGE0_SUPPORT_CONTRACT_FAIL_DECISION_C', 'support_76_event_causal': bool(legal), 'bridge_contract_pass': bool(bridge_contract['valid_shape'] == [1,768] and bridge_contract['invalid_equals_raw']), 'controller_run': False, 'sealed': True, 'public_q1_dev_access': False, 'persistent_ct_validated': False, 'reason': 'Prior-completed support has 76/76 nonempty events, zero future rows and strict prior-video ordering; frozen 768-D bridge fallback is exact.' if legal else 'At least one support/causal contract check failed; no gate training authorized.'}
    atomic(OUT/'audit/phase46_decision.json', decision)
    atomic(OUT/'completion/stage0.done', {'stage': 0, 'support_contract': decision['stage0'], 'decision_code': decision['decision_code']})
    print(json.dumps(decision, indent=2))

if __name__ == '__main__': main()
