# TrackOCD v2 takeover recheck — 2026-10-16

This is a live recheck of the new server. It supplements the historical
takeover snapshot and does not rewrite its older evidence.

## Verified local state

- Working tree: `/data3/liuyeqiang/TrackOCD`
- Branch: `research/cvpr-trackocd-autonomous-v2`
- HEAD at recheck: `aa16a7c` (`separate recovered historical MOT and OCD metrics`)
- `source /data3/liuyeqiang/trackocd_a100_env.sh && PYTHONPATH=. python -m pytest -q tests/trackocd_v2`: **35 passed** in 3.72 seconds.
- The current-source CVPR M endpoint is a separate CPU-only owned run; this
  recheck did not start, stop, or inspect its private inputs.

## Missing v2 formal assets on this host

The following expected v2 inputs were absent at recheck time:

- `data/raw/tao/annotations/{train,validation}.json`
- `data/tao_ow_ocd_v1/splits/{known_ids,unknown_ids_val,distractor_ids}.json`
- `data/tao_ow_ocd_v1/public/pred_track_stream.jsonl`
- `/data2/usr_for_deadline/trackocd_v2/project_outputs`
- the formal predicted feature cache and `audit/frontend_selection.json`

The existing `/data3/liuyeqiang/TAO-Amodal/{annotations,frames}` tree is
present, but it is not silently substituted for the missing v2 lineage,
role splits, predicted stream, or formal cache. No bootstrap output was
created by this recheck.

## NAS connectivity evidence

The configured target `192.168.31.57` was probed read-only using the local
NAS read-only key. TCP/22 timed out. The host currently has only the
`202.205.84.0/24` IPv4 route and no route for `192.168.31.0/24`; bounded
probes to ports 22, 445, 2049, 5000, and 5001 were all unreachable. No NAS
file was read, changed, or copied, and no historical PID/session was used.

## Consequence

The v2 code is locally testable, but formal v2 OCD/MOT cannot be claimed on
this host until the exact v2 role files, predicted physical stream, selected
frontend provenance, and complete feature cache are recovered and verified.
Historical PANDAS/MASA metrics and the current CVPR M diagnostics remain
separately scoped and must not be relabeled as that missing formal result.

## Post-recheck migration repair — 2026-10-16

- The configured `nas57` alias is not installed in this server's SSH config;
  the available read-only key was tried directly against `192.168.31.57:22`
  and the connection timed out. No NAS file or external process was changed.
- The local core evidence contains byte-recovered role files with a passing
  historical machine audit: `known=78`, `unknown_val=209`, `distractor=45`.
  Their recorded hashes are preserved at
  `outputs/trackocd_core/audit/role_recovery.json`; they are candidate
  Train/Val protocol inputs, not a substitute for the missing v2 public
  predicted stream or frontend cache.
- The existing `/data3/liuyeqiang/TAO-Amodal` Train and Validation annotation
  files match the recorded hashes (`train=7eb551...921d0`,
  `validation=041488...76ee`). This establishes a reversible candidate
  mapping for future GT-only audits, but no mapping, cache, training, or
  formal evaluator was started in this recheck.
- The recovered `src/trackocd_v2/io.py` remains byte-identical to the NAS
  source and still names `/data2/usr_for_deadline/trackocd_v2/project_outputs`.
  Direct v2 writers therefore remain unrun on this host; any future local
  output relocation must be implemented in an external, explicitly recorded
  wrapper or after a separately registered source patch. It does not authorize
  predicted features, frontend selection, TAO Test access, or formal
  evaluation.

## Existing metrics, kept in their original scope

- The preserved historical COVTrack-native Val reference reports category-free
  tracking `HOTA=0.15412`, `AssA=0.44569`, `DetA=0.055848`,
  `DetRe=0.54448`, `LocA=0.76654`, and `OWTA=0.48228`; its derived novel and
  persistent observability are `191/819=0.233211` and `88/527=0.166983`.
  The same audit records `legal_gate=FAIL_NO_NOVEL_VOCABULARY`,
  `recomputed_on_a100=false`, and `not_new_ocd_or_commit_ct_result=true`.
- The preserved full-Val class-agnostic RPN/ByteTrack quality audit reports
  `HOTA=0.145054`, `AssA=0.440879`, `DetA=0.048489`, and `DetRe=0.631809`,
  with reliable physical coverage `1400/4413` Known and `189/819` Novel.
  Its status explicitly says quality audit only, not semantic OCD/M9 or
  automatic primary qualification.
- The limited Train-first annotated-cadence OCD development table reports at
  prefix 16: A1-Selected `Old ACC=0.078103`, `New ACC=0.113350`,
  `H-score=0.089303`, `Commit-CT=0`, and A2-Evidence
  `Old ACC=0.050816`, `New ACC=0.123423`, `H-score=0.071512`,
  `Commit-CT=0`. These are existing core development results with a
  scientific-negative conclusion, not formal v2 or TAO Test results.

## Current NAS re-probe — 2026-10-16T03:59:24+08:00

- The local proxy listeners `127.0.0.1:7890` and `127.0.0.1:17899` are
  present, but direct TCP probes to `192.168.31.57` ports `22`, `445`,
  `5000`, `5001`, and `8080` all timed out.
- HTTP CONNECT through either local proxy returned `502`; SSH through either
  proxy did not complete the SSH banner exchange. The `nas57` SSH alias still
  resolves locally to the literal, unavailable hostname `nas57`, not to the
  user-provided `192.168.31.57` configuration.
- This was a bounded read-only probe using the local NAS read-only key. No
  NAS file, SSH configuration, external process, or historical PID/session
  was changed or reused. No formal v2 asset was copied or inferred.
- The formal evaluator remains closed: the predicted physical stream,
  selected frontend provenance, and complete predicted feature cache are
  still absent. The next valid transition is to recover and hash those exact
  assets, then run the registered v2 preflight; local PANDAS/Core/history
  metrics remain separate evidence.
