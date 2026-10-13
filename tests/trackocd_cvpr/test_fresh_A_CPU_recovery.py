import json
from pathlib import Path
import pytest
from src.trackocd_cvpr.fresh_A_CPU_recovery import (
    ORIGINAL,RETRY,verify_original,recovery_configuration,validate_thread_inventory)

ROOT=Path(__file__).resolve().parents[2]


def test_actual_original_thread_stop_preserved_and_only_new_namespace_changes():
    proof=verify_original(ROOT)
    assert proof['original_preregistration_commit']=='bcc2044d88250eb5d5b3bc5544374c5cbdc2b697'
    base=json.loads((ROOT/'configs/trackocd_cvpr/fresh_A_optimizer_pilot_v1.json').read_text())
    new=recovery_configuration(ROOT,base,'cpu_visibility_retry1')
    assert base['output_directory']==ORIGINAL and new['output_directory']==RETRY
    assert {k for k in base if base[k]!=new[k]}=={'output_directory'}
    with pytest.raises(ValueError): recovery_configuration(ROOT,new,'cpu_visibility_retry1')
    with pytest.raises(ValueError): recovery_configuration(ROOT,base,'unbounded_retry')


def test_one_compute_worker_plus_identified_bounded_native_driver_not_autograd_pool():
    main={'tid':10,'name':'python','CPU_ticks':100}
    helper={'tid':11,'name':'cuda00001400006','CPU_ticks':2}
    assert validate_thread_inventory([main],10,100)==0
    assert validate_thread_inventory([main,helper],10,100)==1
    for name in ('pt_autograd_0','python','Thread-1','cuda_not_pinned'):
        with pytest.raises(ValueError): validate_thread_inventory([main,{**helper,'name':name}],10,100)
    with pytest.raises(ValueError): validate_thread_inventory([main,helper,{**helper,'tid':12}],10,100)
    with pytest.raises(ValueError): validate_thread_inventory([main,{**helper,'CPU_ticks':11}],10,100)
    with pytest.raises(ValueError): validate_thread_inventory([helper],10,100)
