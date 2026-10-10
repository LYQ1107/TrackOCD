"""Fixed comparisons from registered case IDs, never from observed metrics."""


def registered_pairs(cases,cfg):
    index={(r['method'],r['seed'],r['order'],r['prefix'],r['coverage_target']):r for r in cases}
    comparisons=[('B1_track_nearest','FULL'),('B2_track_dpmeans','FULL'),('A1_SELECTED','A2_EVIDENCE'),('A1_SELECTED','A2_CAPACITY_CONTROL'),
        ('D1_SIMPLE_MLP','D2_RISK_AWARE'),('WITHOUT_RISK','FULL'),('WITHOUT_TEMPORAL','FULL')]
    for left,right in comparisons:
        targets=cfg['coverage_targets'] if not left.startswith(('A','B')) else [1.]
        for seed in cfg['seeds']:
            for order in cfg['orders']:
                for cap in cfg['prefixes']:
                    for target in targets:
                        a=index[(left,None if left.startswith('B') else seed,order,cap,None if left.startswith(('A','B')) else target)]
                        b=index[(right,seed,order,cap,None if right.startswith('A') else target)]
                        yield left,right,seed,order,cap,target,a,b
