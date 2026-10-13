import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from src.trackocd_cvpr.fresh_A_baseline import fresh_model
from src.trackocd_cvpr.fresh_A_development_encoding import encode_native_prefixes,offline_fit5_native_prototypes
from scripts.trackocd_cvpr.smoke_selected_frozen_encoder import digest


def toy_tracks():
    spec=importlib.util.spec_from_file_location('encoding_fixture',Path(__file__).with_name('test_fresh_A_baseline.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.tracks()


@pytest.mark.parametrize('family',['A1_ADAPTER','A2_EVIDENCE','A2_CAPACITY_CONTROL'])
def test_complete_prefix_native_frozen_and_offline_canonical_fit_only(family):
    rows=toy_tracks();known=rows[:-1];model=fresh_model(family,1027).eval().requires_grad_(False);before=digest(model)
    e,n=encode_native_prefixes(model,known,4,lambda:None)
    assert e.shape==(10,256) and n==[min(4,t.count()) for t in known]
    p,scope=offline_fit5_native_prototypes(model,rows+[rows[0]],lambda:None)
    assert p.shape==(5,256) and np.allclose(np.linalg.norm(p,axis=1),1)
    assert scope['scope']['physical_tracks']==12 and scope['scope']['canonical_known_groups']==10
    assert scope['scope']['masked_no_fit5_target_tracks']==1 and scope['canonical_members_per_external_fit_slot']==[2]*5
    assert not scope['dev_labels_or_prototypes_or_old_checkpoints_used']
    assert digest(model)==before


def test_no_training_no_empty_population_or_undeclared_prefix():
    rows=toy_tracks()[:-1];model=fresh_model('A1_ADAPTER',1027)
    with pytest.raises(ValueError):encode_native_prefixes(model,rows,1,lambda:None)
    model.eval().requires_grad_(False)
    with pytest.raises(ValueError):encode_native_prefixes(model,[],1,lambda:None)
    with pytest.raises(ValueError):encode_native_prefixes(model,rows,3,lambda:None)
    with pytest.raises(ValueError):encode_native_prefixes(model,rows,True,lambda:None)
    with pytest.raises(ValueError):encode_native_prefixes(model,rows,16.0,lambda:None)


def test_numeric_outputs_invariant_to_physical_routing_alias_change_and_unseen_suffix():
    rows=toy_tracks()[:-1];model=fresh_model('A2_EVIDENCE',1027).eval().requires_grad_(False)
    before,n=encode_native_prefixes(model,rows,1,lambda:None)
    for t in rows:t.video+=100
    after,m=encode_native_prefixes(model,rows,1,lambda:None)
    assert np.array_equal(before,after) and m==n
    # Underlying fixture suffix changes; one-observation input never reads it.
    mutated=0
    for t in rows:
        p=t.visible(t.count())
        if t.count()>1:
            changed=torch.flip(p.visual[1:],dims=[1]);mutated+=int((changed!=p.visual[1:]).any())
            p.visual[1:]=changed
    assert mutated>0
    changed,k=encode_native_prefixes(model,rows,1,lambda:None)
    assert np.array_equal(before,changed) and k==n
