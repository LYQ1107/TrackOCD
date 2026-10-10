import json
from pathlib import Path
import numpy as np
from src.trackocd_core.limited_masa_bank import FrozenVideoBank
from src.trackocd_core.limited_masa_features import LimitedMasaVideoCache
from src.trackocd_core.train_first_experiment import frozen_model,evidence_bank
from src.trackocd_core.sealed_parquet import write_sealed,read_sealed
from src.trackocd_core.evaluation import DecisionEvent,TrackKey,seal_decisions


def test_streamed_frozen_bank_matches_every_field_and_uses_true_prefix():
    root=Path(__file__).resolve().parents[2];out=root/'outputs/trackocd_core/features/masa_limited_prefix_v1'
    r=json.loads((out/'integration_manifest.json').read_text())['records'][0]
    for name in ('A0_RAW','A1_SELECTED','A2_EVIDENCE'):
        model,_=frozen_model(root,name,1027);cache=LimitedMasaVideoCache(out/'shards'/r['npz_filename'],r['video_id'])
        ref=evidence_bank(cache,cache.rows,model);bank=FrozenVideoBank(out/'shards',[r],model)
        for cap in (1,2,4,8,16):
            items=list(bank.videos(r['video_id'],cap));assert len(items)==r['tracks']
            for row,item in items:
                expected=ref[cap][row['key']]
                for k,v in expected.items():
                    assert np.array_equal(item[k],v) if isinstance(v,np.ndarray) else item[k]==v
        bank.close();cache.close()


def test_private_compact_ledger_roundtrip_preserves_tokens_wait_order(tmp_path):
    events=[DecisionEvent(0,TrackKey(4,'12'),1,'NEW',token='S:0'),DecisionEvent(1,TrackKey(4,'77'),1,'WAIT'),
        DecisionEvent(2,TrackKey(20,'3'),2,'EXISTING',token='S:0'),DecisionEvent(3,TrackKey(20,'90'),1,'KNOWN',known_category_id=17)]
    s=seal_decisions(events,video_order=[4,20],prefix_cap=2,known_ids=[17]);p=tmp_path/'seal.parquet'
    write_sealed(p,s);assert read_sealed(p,[4,20],[17])==s
    import pytest
    with pytest.raises(ValueError):write_sealed(p,s)
