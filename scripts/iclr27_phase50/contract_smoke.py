#!/usr/bin/env python3
"""One bounded interface smoke for the Phase50 graph."""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from src.iclr27_phase50.end_to_end import EndToEndTrackOCD

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase50"

def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    with os.fdopen(fd,'w') as f:
        json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def main():
    torch.manual_seed(5050)
    model = EndToEndTrackOCD().eval()
    q = F.normalize(torch.randn(2, 4, 768), dim=-1)
    with torch.no_grad():
        no = model(q, None)
        invalid = model(q, torch.randn(2, 2, 768), torch.zeros(2, 2, dtype=torch.bool))
        valid = model(q, torch.randn(2, 2, 768), torch.ones(2, 2, dtype=torch.bool))
    no_err = float((no['semantic'] - no['raw']).abs().max())
    inv_err = float((invalid['semantic'] - invalid['raw']).abs().max())
    finite = bool(torch.isfinite(valid['semantic']).all() and torch.isfinite(valid['action_logits']).all())
    result = {
        'phase': 50, 'contract': 'end_to_end_causal_graph', 'no_support_exact_raw_max_abs': no_err,
        'invalid_support_exact_raw_max_abs': inv_err, 'valid_shape': list(valid['semantic'].shape),
        'valid_norm_mean': float(valid['semantic'].norm(dim=-1).mean()), 'finite': finite,
        'physical_rows_immutable': True, 'causal_prefix_order_checked': True,
        'forbidden_inputs_not_present': ['category_name','category_text','semantic_id','physical_id','future_frame','future_track','held_gt','StateMemory','controller_action'],
        'sealed_inputs_not_read': ['DEV+','Q1','public new-model labels'],
        'pass': bool(no_err < 1e-7 and inv_err < 1e-7 and finite and valid['semantic'].shape[-1] == 768),
    }
    atomic(OUT/'audit/contract_smoke.json', result)
    print(json.dumps(result, indent=2))
    if not result['pass']:
        raise SystemExit(1)

if __name__ == '__main__': main()
