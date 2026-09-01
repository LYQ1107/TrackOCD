# OCD_OVMOT — MOT + On-the-fly Category Discovery

Architecture 1 verification: **Track-then-Discover** on TAO-OW, with SimOWT as
the tracking frontend, PHE as the discovery frontend (adapted to track
embeddings), frozen DINOv2/CLIP features, and official TrackEval for tracking.

## Project layout

```text
configs/          dataset / feature / OCD / tracking configs
data/raw/tao      symlink to read-only TAO data
data/tao_ow_ocd_v1  public/private splits, manifests, stats
data/caches/features  per-track frozen embeddings
data/external_annotations  official SimOWT/TrackEval annotations
docs/             code review, dataset card, protocol, final report
environments/     conda env definitions + pip freeze
outputs/metrics/  tracking and discovery metrics
patches/          third-party patches
scripts/          run_arch1_blocking.sh, merge, evaluate
src/              data / features / ocd / evaluation / tracking code
third_party/      pinned official repositories
runs/             stage logs, checkpoints, prediction logs
```

## Reproduce

```bash
# 1. environments (see environments/*.yml)
conda env create -f environments/discovery_environment.yml
conda env create -f environments/simowt_environment.yml

# 2. one blocking end-to-end run (resumable, stage markers)
bash scripts/run_arch1_blocking.sh 2>&1 | tee runs/arch1_main/blocking.log
```

SimOWT additionally requires the compiled Multi-Scale Deformable Attention op:

```bash
cd third_party/SimOWT/projects/IDOL/idol/models/ops
CUDA_HOME=/usr/local/cuda-11.6 python setup.py install --user
```

## Key results

Historical experiment reports and generated metric tables remain in the
private working tree and are not part of this public source snapshot.

## Public source snapshot

The public repository tracks the project source, scripts, configuration,
tests, environment definitions, and maintenance patches. Local experiment
outputs, checkpoints, raw data, runtime logs, caches, and vendored upstream
repositories are intentionally excluded; obtain those dependencies from
their upstream projects and use the pinned references in `assets/` and
`patches/` where applicable.
