#!/usr/bin/env python3
"""Read-only all-ID validation of completed predicted prefix features, no GT."""
from pathlib import Path
import json
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    from scripts.trackocd_core.extract_limited_masa_features import inputs,CONFIG,barrier
    from src.trackocd_core.limited_masa_features import completed_feature,select_all_prefixes
    from src.trackocd_v2.io import sha256_file,atomic_json
    cfg,_,plan=inputs();out=ROOT/cfg['output_directory'];manifest_path=out/'full_manifest.json'
    manifest=json.loads(manifest_path.read_text());sources=json.loads((ROOT/cfg['prediction_run']/'prediction_manifest.json').read_text())
    physical={r['video_id']:r for r in sources['videos']};metadata={v['video_id']:v for v in plan['videos']}
    receipt=json.loads((out/'supervisor_full_c1d166eedd9d4ec98109f35f0d5be52c.json').read_text())
    if receipt['error'] is not None or receipt['owned_child_returncodes']!=[0]*8:raise ValueError('Actual successful supervisor required')
    if (manifest['status'],manifest['videos'],manifest['tracks'],manifest['observations'])!=('COMPLETE_FULL_LIMITED_MASA_ALL_ID_PREFIX_FEATURES',988,304561,1294110):raise ValueError('Full feature manifest incomplete')
    barrier();started=time.monotonic();tracks=observations=singletons=reused=payload=0
    for r in manifest['records']:
        if completed_feature(out/'shards',r['video_id'],sha256_file(CONFIG),physical[r['video_id']]['npz_sha256'])!=r:raise ValueError('Atomic marker/payload/manifest differs')
        p=ROOT/cfg['prediction_run']/'shards'/physical[r['video_id']]['npz_filename']
        if sha256_file(p)!=r['input_npz_sha256']:raise ValueError('Frozen physical input differs')
        with np.load(p,allow_pickle=False) as archive,np.load(out/'shards'/r['npz_filename'],allow_pickle=False) as a:
            # NpzFile re-decompresses on every __getitem__; materialize one
            # bounded video once. Check values/selection remain unchanged.
            raw={k:archive[k] for k in archive.files}
            rows,n=select_all_prefixes(metadata[r['video_id']],raw,cfg['maximum_observations'])
            assert np.array_equal(a['track_id'],[row['physical_track_id'] for row in rows])
            assert np.array_equal(a['offsets'],[row['observation_offset'] for row in rows]+[n])
            lengths=np.diff(a['offsets']);assert np.all((lengths>=1)&(lengths<=16))
            assert a['visual'].dtype==np.float16 and a['visual'].shape==(n,768) and np.isfinite(a['visual']).all()
            assert a['geometry'].shape==(n,4) and np.isfinite(a['geometry']).all() and np.isfinite(a['quality']).all()
            images=metadata[r['video_id']]['images'];source_rows=[o for row in rows for o in row['observations']]
            assert np.array_equal(a['frame_index'],[images[o['image_position']]['frame_index'] for o in source_rows])
            assert np.array_equal(a['image_id'],[images[o['image_position']]['image_id'] for o in source_rows])
            assert np.array_equal(a['quality'],[raw['score'][o['source_offset']] for o in source_rows])
            assert (len(rows),n)==(r['tracks'],r['observations'])
            singletons+=int(np.count_nonzero(lengths==1))
        tracks+=r['tracks'];observations+=r['observations'];reused+=r['reused_existing_smoke_observations'];payload+=r['npz_bytes']
    assert (tracks,observations,singletons,reused)==(304561,1294110,114380,8)
    result={'status':'PASS_ALL_988_ATOMIC_FEATURE_SHARDS_AND_TRUE_FIRST_PREFIXES','videos':988,'tracks':tracks,'observations':observations,
        'singletons_retained':singletons,'reused_smoke_observations':reused,'compressed_payload_bytes':payload,
        'full_feature_manifest_sha256':sha256_file(manifest_path),'validation_source_sha256':sha256_file(Path(__file__).resolve()),
        'all_protected_Train_hashes_verified':True,'all_actual_input_NPZ_and_feature_payload_hashes_verified':True,
        'all_predicted_IDs_and_frame_image_score_prefixes_exact':True,'Val_GT_read':False,'TAO_Test_access':False,
        'wall_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    atomic_json(ROOT/'outputs/trackocd_core/LIMITED_MASA_FULL_FEATURE_VALIDATION.json',result);print(json.dumps(result))


if __name__=='__main__':main()
