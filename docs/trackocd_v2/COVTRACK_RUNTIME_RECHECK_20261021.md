# COVTrack runtime recheck — 2026-10-21

This is a compatibility and asset recheck only. It did not read TAO Test,
start training, start a new COVTrack inference run, or promote the old partial
shard-1 output.

## Evidence

- The recovered public source remains the pinned local COVTrack tree. Its
  original `/data3/liuyeqiang/.venvs/covtrack-public/bin/python` symlink points
  through the removed `cfp-fishmot` interpreter and is not executable.
- The retained package overlay is internally compatible when launched through
  `scripts/trackocd_v2/covtrack_python_compat.py`: PyTorch `2.0.0+cu117`,
  torchvision `0.15.0+cu117`, MMCV `1.7.2`, MMDetection `2.28.2`, CLIP,
  LVIS, and pycocotools import successfully, including `mmcv.ops`.
- The upstream `tools/test.py --help` path imports successfully with the
  BDD/LAP/optional-metrics guards. The focused repository tests passed: `5
  passed`.
- Correct COVTrack model registration was reached with the pinned config, but
  model construction fails because the standalone OpenAI CLIP
  `ViT-B/32` weight is absent. The attempted network fetch ended with
  `SSL: UNEXPECTED_EOF_WHILE_READING`. No substitute backbone is admissible.
- `motmetrics` is not present locally. The compatibility shim only permits the
  native `--format-only` route and raises explicitly if an OCD/MOT evaluation
  path is selected; it cannot create metrics.
- The previous shard-1 native log is retained as historical partial evidence:
  it reached `78342/120066` observations before the frozen RAM-reserve pause.
  Its partial native output was not promoted and must not be counted as a
  completed shard.
- Read-only NAS access was attempted using the supplied address
  `192.168.31.57:22`; the direct TCP connection timed out from this server.
  No NAS files were read or changed.

## Boundary and next action

The current v2 formal OCD/MOT result remains pending; this recheck produces no
new metric. When a local or NAS copy of the exact CLIP weight becomes
available, re-run an asset hash check, then launch only a fresh namespace for
the incomplete shard. GPU ownership and the 25% RAM guard must be rechecked at
launch; external processes remain untouched.
