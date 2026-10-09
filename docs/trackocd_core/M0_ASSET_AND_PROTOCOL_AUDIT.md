# M0 — TrackOCD core asset and protocol audit

Initial audit: 2026-10-09 (Asia/Shanghai); continuation recorded in UTC below.
Engineering audit and prior GitHub delivery: **PASS**. Exact later v2 source:
**81/81 NAS/Git/A100 byte identities verified**. Historical runtime assets:
**PARTIAL; metadata, physical streams and feature payloads not transferred**.
Algorithm hypotheses in **M0 alone**: **NOT TESTED**. Later bounded Train
Known GT feasibility has now run; see the final dated handover section.
No new training, feature extraction, tracking inference or TAO Test access has
been performed in M0.

## Code and preservation

The new branch is `codex/trackocd-track-conditioned-persistent-discovery`,
created from PANDAS delivery `7e0d4b6cf8046d173afedb0f16e138d45c43a416`.
Its merge base with `codex/trackocd-v2` is
`cac66467af7ca137fc53bdbef34896dde47030d8`, also verified using `git ls-remote`.
Thus the real pushed v2 source is inherited; main's Phase88/89/90 source is
not substituted for v2. The completed PANDAS reports remain in this branch.

The initial worktree was clean, with one worktree and no existing attached
research worktree. No reset, stash application, force push, deletion or
external-process intervention was used. Two recovery stashes remain intact;
both contain the 138-line historical research-log addition. Local ignored
`research_log.md` and `docs/trackocd_v2/TAKEOVER_20260929.md` remain untouched.

The named v2 protocol/schema, three baseline methods, two evaluators and five
pipeline scripts were read. Their exact SHA256 values are recorded in
`outputs/trackocd_core/audit/assets.json`. The later old-server cross-track
batching and prefix-atomic execution fixes are not established as restored:
the pushed builder flushes within one track, writes per-track JSON/markers,
and the predicted runner seals all prefixes only after the complete stream.
These scripts will not be relaunched unchanged on the new server.

## Inherited category protocol recovered, not redefined

The complete machine audit printed in the old session journal at line 297,
timestamp `2026-09-15T17:25:29.838Z`, contains all original role IDs and hashes.
Those IDs were recovered verbatim into `configs/trackocd_core/roles.json`:
78 Known, 209 observed Val Novel, 45 distractor categories. The initial M0
commit `44615cc` recovered ID content but not byte identity. A follow-up
inspection found the original serialization recipe in
`src/data/build_protocol.py:225`: `json.dumps(ids, indent=1)` without a final
newline. All three payloads produced the **exact original SHA256 hashes**.

`scripts/trackocd_core/recover_role_files.py` now reconstructs only these three
tiny assets under `outputs/trackocd_core/recovered_splits/`, validating all
payloads and any existing target before writing. The successful manifest is
`outputs/trackocd_core/audit/role_recovery.json`. Their combined size is only
2,031 bytes (471 / 1,284 / 276); the original worktree and other runtime assets
are still not recovered. No IDs, category roles or performance-derived
selection were changed.

Known IDs and distractor IDs are equal to the official maps in the existing
Open-World-Tracking checkout at
`060dd193127ac1f2ae8a580ce4382d152c35e0fc`. Official Known map SHA256 is
`a95567f04769319bfa65a4fbe6a0791b745123f947e6b7861c8705e54f7718a2`;
distractor map SHA256 is
`b999eee97cb569d9f411ee2060582b04a0fe6fd98c5134b75132dc42c2fc33da`.
The original v2 Novel set is the 209 observed categories, not PANDAS's
1107-category metadata complement. All observed Val annotation categories
have one inherited role. Nothing was selected from performance.

## Actual local data and supervision

| Input | Videos | Annotated images | Annotations | GT tracks | Availability |
|---|---:|---:|---:|---:|---|
| TAO Train | 500 | 18,274 | 54,639 | 2,647 | present; all annotated frame paths exist |
| TAO Validation | 988 | 36,375 | 113,112 | 5,485 | present; all annotated frame paths exist |

Train SHA256:
`7eb551fdeeeebc76b876ae255f91dc5662c7270a125955c5f1be2d9bd30921d0`.
Val SHA256:
`0414885ee2702c2d3176cf6184e7811a7bd1c1347a157fef57a91020976776ee`.
These equal the old canonical Train/Val lineage evidence. The existing frame
tree is about 354 GiB and is reused in place; it was not copied or rewired.

Only 48/78 Known categories have Train supervision. Train contains 43,380
Known annotation observations, but also 7,374 observations of inherited Val
Novel categories and 2,006 train-unassigned observations. The latter two
populations are **excluded from representation/policy training**. This is
stricter than merely saying "Train-only": truly Novel labels must not enter
training because they happen to exist in the Train JSON. Pseudo-Novel episodes
will use only the 48 supported Known categories, with category/video-disjoint
selection or preregistered cross-validation if support is insufficient.

The remaining 30 Known categories have no Train prototype supervision. Every
method must disclose the same availability and zero-shot limitations; none
may manufacture prototypes from Val labels or silently redefine Old/New.

No TAO Test annotation was opened. The core config forbids Test access for the
entire goal even if a legacy `FINAL_FREEZE` marker later appears. Legacy
Test-after-freeze behavior is not the new task's authority.

## Assets, reuse and reconstruction scope

| Asset | Actual new-server status | Reuse / risk / minimal action |
|---|---|---|
| v2 source and tests | present, real pushed line | reuse helpers/methods; preserve historical evaluator |
| TAO Train/Val frames and annotations | present and lineage verified | reuse in place; evaluator/training whitelist separation |
| old GT-track DINO features | not found in project/cache/backup audit | do not declare recovered; obtain manifest first, otherwise small compact re-extraction |
| selected predicted COVTrack-native stream | not found locally | recover exact tracks/config/provenance; supervision and causality unverified |
| old 649,378-track SimOWT/Q0 stream | not found locally | historical count is not a recovered frontend; fragment audit required |
| formal COVTrack-native shards | not found locally | inspect NAS manifest, done markers, hashes and last update before any resume |
| DINOv2 B/14 checkpoint and hub | not found locally | official checkpoint source, 346,378,731 bytes and hash are pinned in existing asset manifest |
| SimOWT checkpoint/environment | checkpoint missing; Python 3.7 environment absent | restore cached outputs first; avoid unnecessary full legacy reinstall |
| PHE checkpoint | missing | historical 48/78 Known coverage; pure CLS compatibility needs independent proof |
| Phase90 H3 checkpoint | missing; old feature protocol incompatible | INCOMPARABLE, not a substitute for the proposed evidence/policy model |
| PANDAS original / FG-0/1/2/3 full-Val results | present and frozen | external reference only; no extra threshold sweep/inference/training |
| Dedup-C 20+20 diagnostics | present; Novel calibration recall gate failed | preserve subset label; never call 0.289395 full-Val HOTA |
| InterMOT models/features | DanceTrack/person-ReID assets, not common TAO DINO descriptors | INCOMPARABLE; do not relabel them as TAO features or alter that project |
| MASA / AED | no usable local TAO checkpoint or output established | not selected by model name; no new large bake-off |

The available 36-GiB `datasets/codex-backup/full_20260920_140444` tree contains
Codex history and unrelated IR checkpoints, not a recovered v2 runtime tree.
No files in that backup were modified. Targeted hidden/no-ignore searches
found no required v2 feature/model/stream asset. Foreign personal directories
on `/data1` and `/data2` were not searched or modified.

## NAS and last historical state

No NAS mount exists here. A bounded BatchMode SSH probe to
`192.168.31.57:22` timed out; independent bounded TCP probes to ports 22, 445
and 2049 also timed out. No authentication/configuration changes were made.
Candidate old roots remain the user-provided `/home/Valadmin/A100` project,
`usr_for_deadline/trackocd_v2`, `trackocd_phase90` and `trackocd_archive` paths.
Their current symlink mappings, sizes and contents are **unverified**.

A subsequent read-only inspection of the app's NAS thread `盘点当前文件夹`
(`01a0c851-ce6c-76e2-ac44-d6b0e3618ceb`, host `remote-ssh-discovered:nas57`)
recovered its completed inventory from **2026-09-22**. That command output
lists the OCD-root DINOv2 B/14 checkpoint (331 MiB), SimOWT checkpoint
(518 MiB), COVTrack checkpoints and three DINO-PHE checkpoints (61 MiB each).
These are stronger location leads than a chat recollection, but they are
**historical listings**, not current file/hash verification or restored assets.
The pinned DINO/SimOWT hashes must still be checked at the source and destination.
An existing Phase90 tree and a 146-G historical archive do not prove recovery
of the later v2 code, selected stream or atomic shards. The task's available
app tools can inspect peer thread output but provide no cross-host shell or
file-copy operation; this does not establish A100-to-NAS connectivity.

`configs/trackocd_core/minimal_asset_recovery.json` now names those exact
checkpoint candidates and the unresolved v2 code/metadata/cache locations.
It prioritizes small code/metadata and DINO before optional legacy checkpoints,
requires manifests and sizes before any feature transfer, and excludes whole
archives, frame duplication and Test assets. No bulk migration was started.

The last machine output in the old journal at line 22062,
`2026-09-17T18:28:14.579Z`, reports 65 done atomic shards / 247,756 observations
and cache status RUNNING. This is historical evidence only; the actual NAS
cache may have advanced. No old PID/session was used and no extraction
completion is claimed. Valid complete shards should be reused once accessible.

## Protocol defects that must not migrate into new main results

1. Persistent success currently checks whether any previous member of a token
   had the correct category in another video; wrong members do not poison
   that success. The new evaluator must require pure correct reuse and report
   contamination explicitly.
2. Wrong NEW, EXISTING and KNOWN are mixed in one false-assignment number.
   The new evaluator must separate fragmentation, wrong merge and wrong Known.
3. Category eligibility counts local physical IDs without a video namespace.
4. Predicted semantic metrics use only the post-hoc matched join. Missing
   predicted tracks therefore disappear from their primary denominators.
5. Standard OCD globally remaps Known and anonymous tokens together. Known
   output identity must not be "repaired" by a post-hoc anonymous mapping.
6. Whole-track prefix replay reads complete track descriptors and groups track
   steps without an event-time scheduler. Its prefix-causal claim is not proof
   of truly frame-online decisions or cross-track event ordering.
7. `TrackSample.model_view()` exposes the complete feature matrix. A caller's
   prefix discipline is not an automatic future-field whitelist.

Historical v2 numbers remain historical and are not silently rescored or used
as the new baseline table. New schema, denominator formulas, leakage guards
and synthetic evaluator tests belong in `src/trackocd_core/evaluation/`.

## Storage and next gate

Initial `df -h` showed approximately 91 GiB available; statvfs at completion
of the annotation audit measured 90.13 GiB. GPU 0–4 and 6–9 were essentially
idle at the initial snapshot; GPU 5 held an external job and was untouched.
No GPU worker was started in M0. Resource selection must be refreshed before
execution. New task soft/hard budgets are 15/30 GiB.

Recomputable features for the 43,380 legal Train Known observations require
only 66,631,680 raw FP16 descriptor bytes (768 dimensions), before compact
metadata. This is a size estimate, **not an extracted cache**. It does not
justify reconstructing millions of predicted crops or per-track JSONs.

M1 must first audit any available lawful physical output. The historical
selected frontend is asset-blocked and supervision-unverified. The local
PANDAS reference has full-Val HOTA 0.110113 / DetA 0.037554 / AssA 0.330246,
with 2,370,335 physical tracks; it cannot be promoted as a reliable frontend
without quality/coverage evidence. If no qualified frontend can be frozen,
record `BLOCKED_FRONTEND_QUALITY` and distinguish any permissible GT-track
feasibility work from an end-to-end result. No architecture training or
positive scientific claim is authorized by this M0 engineering pass alone.

Validation: nine new asset/role/forbidden-annotation tests plus six historical
v2 tests passed (`15 passed`). New recovery tests verify original byte hashes,
idempotent reuse, pre-write rejection of corrupt input and refusal to overwrite
a different existing asset. The first audit invocation correctly stopped
on a source-map mismatch caused by parsing distractor-group keys as category
IDs. The parser was corrected to flatten official category lists; no role
was changed. A complete successful audit and source hashes are retained.

## M0 delivery continuation

The first delivery and continuation both observed GitHub HTTPS TLS failures
and NAS SSH timeouts. A no-proxy GitHub connection and SSH-over-443 also
timed out. No proxy daemon, network configuration or external process was
changed. These are execution prerequisites, not negative algorithm results.
Every local M0 commit remains preserved for an ordinary non-force push;
M1–M10 have not been skipped because of failed remote verification.

The user's GitHub SSH-key screenshot was checked against the local **public**
key fingerprint; they match. No private key was printed, exported or changed.
The identity is an existing GitHub key, not proof of a NAS login route.
Bounded GitHub probes using that key failed before authentication: direct port
22 timed out; SSH-over-443 through the existing HTTP/SOCKS proxies and port
22 through the HTTP proxy timed out during banner exchange. This is not an
observed key rejection and not proof that the repository or remote branch is
absent. No proxy service, SSH config, host-key checking or unrelated process
was changed. At that point, the requested NAS connection fields were
HostName/User/Port and any jump/proxy route in the user's `Host nas57` stanza.

The user subsequently provided that client stanza: `Valadmin@192.168.31.57:22`,
client identity `id_ed25519_nas_192_168_31_57`, and
`RemoteForward 17890 127.0.0.1:7890`. This is a client-to-NAS login plus a
client-proxy listener on NAS, not an A100 jump route. A fresh A100 direct
probe still timed out, and the named identity is absent here. No identity
was copied and the NAS stanza was not installed into A100's SSH config.
Existing A100 HTTP/SOCKS proxies accepted a CONNECT request for NAS but did
not return any SSH banner; accepted proxy replies are not treated as a
verified NAS connection.

Read-only listener/controller inspection identifies A100's 7890/7891 as
the existing local `mihomo`, not the client's NAS reverse forward. Its
selected `hostwind` leaf reported `alive=false` with recent zero-delay
health observations, while the GitHub HTTPS probe failed. This supplies
an upstream-network lead, not proof of GitHub key rejection or authority
to change routing, restart services or interfere with other jobs.
A100 loopback port 17890 has no listener at this snapshot. A separate
client-opened loopback SSH reverse forward could supply a task-scoped
proxy without changing the current proxy; it has not been created.

On this follow-up, the worktree was clean before documentation edits and the
same 15 tests passed again. About 91 GiB remains free. Training, large feature
extraction and scientific evaluation have still not started; M0 remote
delivery and current NAS asset verification remain required before advancing.

## Verified delivery and current NAS evidence — 2026-10-09T16:16:35Z

This supersedes the earlier transport/listener snapshots, not their history.
The client-maintained A100 loopback relay at `127.0.0.1:17890` is now present.
An ordinary non-force push published all four M0 commits through
`38be5dd867c7c2ab2dee5f090a2c05366b12c96a`; an independent `git ls-remote`
returned that exact SHA for the new research branch. The remote v2 branch
still points to `cac66467af7ca137fc53bdbef34896dde47030d8`. No global proxy,
SSH configuration, proxy daemon or external process was modified.

The user started a new **read-only source audit** in NAS thread
`盘点当前文件夹`, turn `01a1216a-ce14-7de0-9eec-36e9dd8af799`.
Its completed command outputs, rather than the old journal, now establish:

| Asset | Current source-side evidence | Recovery consequence |
|---|---|---|
| v2 Python source | 16 source + 49 script files, 611,764 bytes; static AST passes | Actual later source is located, **not transferred**. A100 has 27 tracked v2 Python files and lacks the sharded builder/reader. Exact source Git HEAD and file hashes remain unverified. |
| DINOv2 B/14 weight | 346,378,731 bytes; source SHA and ZIP CRC match pinned official asset | Canonical weight can be recovered independently; old hub code is still missing. |
| COVTrack physical stream | 369,794 tracks / 1,133,841 observations, no duplicate sample keys; SHA `63945d2ae19f63390930b3ad0ed1971d64c4d3ff1067fff2842317f31d8fe2f1` matches cache input | Located on NAS, not restored on A100 or qualified for the new frozen frontend. |
| Formal shards | **255 of 723 required**, 130,560 tracks / 561,672 observations; 2,087,621,953 payload bytes | Partial cache, 35.3061% track coverage; retain and validate existing units before any future resume. |
| GT cache | Train 2,555 and Val 5,232 payload keys / done markers exactly match their manifests; 1,350,657,893 / 2,762,499,153 JSON bytes | File coverage is complete; contents were only sampled. This is GT input, not predicted tracking. Avoid raw JSON migration; later prefer compact, Train-Known-only export after lineage checks. |

The formal manifest still says `RUNNING`; its `shard_count=255` is the existing
count, not the required whole-cache count. With 512 tracks/shard, the latter
is `ceil(369794/512)=723`. Missing units are **shard-000255 through
shard-000722** (468 shards / 239,234 tracks). The latest completed marker is
`2026-09-17T21:29:32.138874+00:00`, later than the old 65-shard journal entry.
There is **no evidence of a currently live feature extractor**.

All 255 existing units have their done marker and four payload files.
Done/manifest agreement, NPY FP16 shapes and byte lengths, and Parquet envelope
checks passed. **Parquet row contents, numerical array values and individual
payload checksums were not validated**. They are structurally consistent
reuse candidates, not yet a fully validated/recovered cache. The legacy
reader correctly rejects a non-`COMPLETE` cache; do not edit that marker to
pretend completion or restart an old supervisor/PID.

The applicable NAS `AGENTS.md` (155 lines / 9,872 bytes) was read in full by
the main agent from the complete source command output. No inspected v2
subdirectory override was found. Its old Luna/Phase24 workflow was not
started; the user's current M0–M10 goal and absolute no-TAO-Test constraint
remain authoritative. During the NAS audit initial directory enumeration
showed Test-related filenames/sizes, but no Test content was opened; subsequent
checks excluded those assets.

Current source paths are explicit in
`outputs/trackocd_core/audit/nas_source_audit.json`. Old absolute `/data2`
output paths and NAS broken symlinks must not be mistaken for valid new-server
locations. The formerly relocated `/home/lwr/trackocd_v2_cache/pred` target
is absent on NAS. This does not affect the separately located formal shards.

### Weight recovered on A100; remaining environment gap

The official DINOv2 weight was downloaded through the task-scoped relay into
a task-owned temporary file. Its exact byte size and SHA were validated before
an atomic no-overwrite hard-link installation at
`/data3/liuyeqiang/TrackOCD/checkpoints/dinov2_vitb14_pretrain.pth`.
The destination hash matches both the pinned manifest and current NAS weight:
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`.
Only the verified download temporary file/empty directory was removed; no
pre-existing checkpoint or cache was touched. The checkpoint is ignored and
will not be pushed to GitHub. New large-asset storage is 346,378,731 bytes
(about 330.33 MiB), below the 15/30-GiB limits.

The existing environment still supplies torch 2.6.0+cu118, torchvision
0.21.0+cu118, NumPy 2.2.6 and pandas 2.3.3. **pyarrow is absent** and is a
concrete gap for reading the formal Parquet indices. Old DINO hub code remains
unavailable on both checked source hub paths and A100. OmegaConf, Hydra and
xformers are also absent, but are not declared mandatory without the recovered
runtime path. No package installation, CUDA worker, model inference or training
was performed in this continuation. CPU asset checks are not a DINO inference
smoke or a feature-contract compatibility proof.

### Next smallest recovery action

First transfer a **small code/config/test/provenance bundle**, not old archives,
datasets, all GT JSON caches or the incomplete formal payloads. The complete
Python inventory alone is only 611,764 bytes. Read source Git status and hash
the named files; preserve exact bytes separately on A100 before comparing or
merging unpublished work. The v2 sharded builder, reader and prefix-atomic
runner must be recovered from real source, not rewritten from this audit.
Remaining config/test/doc/metadata sizes require a source-side inventory.

The source-side audit is now completed/idle, **not a live worker to wait on**.
This A100 chat can read peer output but still lacks a message-dispatch or
cross-host transfer tool. A client-mediated small transfer or source-side
export is needed. See `docs/trackocd_core/NAS_MINIMAL_RECOVERY_REQUEST.md`.
No bulk copying or new feature extraction is authorized by these source
findings. M1–M10 have not been run, and no scientific PASS is claimed.

An A100 file-by-file comparison against the actual NAS inventory confirms
**38 missing files and eight shared files with different byte sizes**, including
`persistent.py`, `protocol.py`, `build_common_features.py`,
`build_predicted_evaluator_join.py`, `build_gt_benchmark_table.py`,
`audit_canonical_tao.py`, `autonomous_supervisor.py` and `run_pred_baselines.py`.
The other 19 shared files have equal sizes, which is **not proof of byte
identity**. This demonstrates a real source-version gap, not just missing
runtime assets. The comparison and local SHA values are reproducible in
`scripts/trackocd_core/audit_assets.py`; NAS source SHA values remain pending.
In particular, conclusions about the pushed old persistent evaluator must not
be presented as a completed audit of the changed NAS evaluator.

## Git-first recovery and minimum Parquet dependency — 2026-10-09T16:45:22Z

The user chose **NAS → GitHub recovery branch → A100 targeted fetch**. The
source-only recovery request now uses a temporary isolated clone and a new
`codex/trackocd-v2-nas-recovery` branch, preserving the original NAS worktree,
main, old v2 and current research branch. It excludes data, weights, feature
payloads and Test outputs, forbids force push, and requires source hashes and
independent remote HEAD verification. The previous no-Git archive instruction
is superseded; a source archive is fallback only if NAS Git authentication
is unavailable. Neither a source push nor NAS authentication has yet been
verified. At the fresh check, the NAS peer was completed/idle and this recovery
branch was absent; no live upload was being waited on.

Reading the actual NAS `sharded_cache.py` output establishes the missing
backend precisely: `_load_index()` imports `pyarrow.parquet`, otherwise raising
`pyarrow is required to read the sharded feature index`. The existing A100
environment has no pip module, so the installed `uv` was used to add **only
`pyarrow==25.0.1`**, with `--no-deps --no-cache --only-binary=:all:` and the
per-command 17890 proxy. Official [installation guidance](https://arrow.apache.org/docs/python/install.html)
and the [25.0.1 release](https://pypi.org/project/pyarrow/25.0.1/) identify the
supported binary distribution/Python compatibility; they do not prove our
NAS cache integrity. No other packages or shared service settings changed.

`environments/trackocd_core_extra_requirements.txt` pins this minimal addition.
The reproducible CPU smoke in `scripts/trackocd_core/check_parquet_dependency.py`
passed for **two synthetic tracks / five observations**: nested Parquet rows,
offset/count indexing into a `(5,768)` FP16 NPY memmap, and separate videos
retaining the same local physical ID. It did **not** import the unrecovered
legacy reader, validate actual NAS shards, run a model, or open annotations.
The independent report is `outputs/trackocd_core/audit/parquet_dependency_smoke.json`.
The one new regression plus the earlier asset and v2 tests passed (**16 total**).

The smoke confirmed torch `2.6.0+cu118`, torchvision `0.21.0+cu118`, NumPy
`2.2.6` and pandas `2.3.3` unchanged. PyArrow's installed distribution manifest
accounts for 157,010,660 bytes; together with the recovered DINO checkpoint,
named added large assets/dependencies account for 503,389,391 bytes (about
480.1 MiB), well within 15/30 GiB. The 12,488-byte synthetic temporary payload
was task-owned and removed automatically; no existing cache was removed.
Old hub code and the exact later v2 source/metadata remain pending.
This is **M0 engineering progress only**, not completion of M1 or any scientific
gate, and the real 255-shard cache remains partial and unvalidated in content.

## Exact NAS source recovery — 2026-10-10 (Asia/Shanghai)

Remote branch `codex/trackocd-v2-nas-recovery` independently resolves to
`604d7eff50e2db386b597ec81690f16d0b40885e`. Only that branch was fetched.
The source-side delivery turn `01a121ab-161b-77a0-8c92-d74972d8306a` recorded
Git blob hashes before and after delivery: all 81 agree. NAS original HEAD
remains `24907fe1d6a2c296de06732392deb1c2eda9c23d`; its index digest was also
reported unchanged. This recovers source bytes, **not the full old Git history**.

The recovery commit's actual parent is main `7b1d3c0`, not the requested old-v2
base. Its exact diff nevertheless contains only the registered 81 v2 files:
65 implementation/entrypoint files, 15 synthetic test files, one protocol JSON,
642,256 bytes. No blanket merge was performed. Only these exact files were
materialized into the existing research branch: 50 added, nine updated, 22
already identical. Main, original v2, prior research commits and both recovery
stashes are preserved. `nas_source_delivery.json` records the source evidence;
`verify_recovered_source.py` independently checks Git tree identity and all
A100 bytes, adding SHA256 values in `source_recovery.json`.

The recovered sharded builder really contains cross-track batching; the
predicted runner really seals/resumes each prefix atomically. The newer
persistent evaluator fixes track namespacing and the runner supplies unmatched
GT targets as DEFER under a fixed denominator. These findings supersede the
initial audit of `cac6646`. Remaining limitations are real: Standard OCD still
remaps Known tokens by Hungarian; contaminated-token reuse is still accepted;
wrong NEW/EXISTING/KNOWN are not separately scored; track-prefix replay is not
a proof of event-time online causality. M3 must use a separate core evaluator,
not silently reinterpret historical scores or overwrite the legacy evaluator.

First A100 synthetic run: **43 PASS / 2 FAIL**. The failures expose old server
assumptions: the sharded preflight test calls `ensure_output_layout()` before
its gate and attempts the old `/data2` mount (permission denied; no output
created), and the ownership test hardcodes the old root. A separate
`tests/trackocd_v2/conftest.py` redirects synthetic I/O to pytest tmp directories
and simulates that ownership fixture's root. All 81 recovered files remain
byte-identical; no old runner was launched. The corrected migration suite plus
source-identity tests now reports **48 PASS** on the existing Python 3.10 /
torch 2.6 / PyArrow 25 stack. This is compatibility evidence, not model inference.

No training, full cache extraction, tracking run or TAO Test data access took
place. The old output paths, missing native environments/hub code, legacy
Test-after-freeze stages and historical PID markers must not be executed as
new-server recovery. DINO weight and PyArrow are available; exact embedding
code lineage and actual cache numerical/Parquet values remain unverified.
`research_log.md`, project reports and runtime frontend/representation provenance
were excluded from the source commit and are still required as a small,
Val-only metadata follow-up. No cache completion or frontend freeze is claimed.

The subsequent M1 continuation recovered all five named Val metadata files
byte-for-byte and inspected the actual source/configuration and runtime log.
Its [qualification report](M1_FRONTEND_AUDIT.md) supersedes treating historical
COVTrack "clean" flags as proof: the detector uses Novel vocabulary, so the
native stream/cache is retained as a reference, not resumed as core input.

## Current recovery handover — 2026-10-10 (Asia/Shanghai)

The code mismatch is resolved: exact 81-file v2 recovery from `604d7eff`,
including actual cross-track batching and atomic prefix resume. Source files
remain byte-identical in the current research branch. Main, old v2, both
recovery stashes, ignored user histories and historical evaluators are intact.
Earlier missing-code/environment snapshots above are history, not current gaps.

Restored/reused: pinned DINOv2 B/14 weight and official source `7764ea0`,
existing Python 3.10 / torch 2.6 CUDA environment, minimal PyArrow addition,
in-place Train/Val data, exact role bytes, five COV runtime metadata files and
four private SimOWT source/metadata files. No old SimOWT 3.7 environment,
physical stream payload, COV weights or NAS feature payload was copied.

Actual last cache state remains **255/723 formal shards**, 561,672
observations; not historical 65 and not extraction complete. Headers/envelopes
and markers passed source-side checks, not all numerical/Parquet content/SHA.
Native COVTrack uses Novel vocabulary, so keep these as reference assets and
do not resume them as clean primary input. A new DINO pin/preprocessing is
not proven identical to old NAS descriptor code; do not mix cache lineages.

Main M1 is still `BLOCKED_FRONTEND_QUALITY`. PANDAS Novel coverage is weak;
COVTrack fails no-Novel-vocabulary; SimOWT has median-one fragmentation and
score-contract issues, with complete supervision/runtime binding unresolved.
The completed additional NAS native-row audit verifies 1,853,369 scores:
1,802,527 in the repeated-sigmoid region, 50,842 near .7501, no scores below
.5. This is artifact consistency, not proof of lawful historical training or
all fragmentation causality. The NAS audits are completed, not live jobs.

Permitted GT exception progressed beyond synthetic tests: 64 deterministic
Train Known tracks / 1,024 observations, globally video-disjoint partitions,
12 classes split into four adaptation/four policy pseudo-Novel/four heldout
pseudo-Novel. Original TAO roles unchanged. Four exact smoke tracks reused.
Compact pilot payload 2,114,366 B, frozen inference 38.13 s; no full cache.
Separate evaluator handles fixed GT universes and absorbing contamination.
Actual fixed baselines: 120 replays; original representations: 280 replays;
single preregistered R1: another 280 replays. These are **Train GT only**, not
new TAO Val detection/tracking or formal main M4/M5/M8/M9 completion.

Six original plus six R1 small adapter/evidence fits used only the 12 legal
fit tracks / 192 observations. Each six-fit job took ~13 seconds, not long
training; DINO/detector/tracker stayed frozen. Private checkpoints total
25,645,464 B. Original unknown discovery failed; R1 partially improved H,
but did not establish stable A2-over-A1 temporal benefit. Preserve negative
results and the simpler A1 candidate. The sole correction round is consumed;
no R2, GT-as-predicted claim or M11 authorization. Regression: 124 pass.

Named new weight/dependency/feature/checkpoint payload is 531,287,428 B,
excluding small source/receipts/vendor checkout; far below 15/30 GiB. This
does not include the pre-existing ~354-GiB dataset, which was reused in place.
Current disk snapshots are ~88 GiB free, not the initial 295-GiB user estimate.
No Test access, external process interference, destructive reset or full job.

Continue from: M1 lawful primary provenance/assets and independent geometry
qualification. Official SimOWT [model instructions](https://github.com/22109095/SimOWT/blob/753b2ac0082976753350d1b74a4d332088889e57/assets/train%26test%26model_zoo.md)
and [data preparation](https://github.com/22109095/SimOWT/blob/753b2ac0082976753350d1b74a4d332088889e57/assets/data.md)
do not seal the checkpoint's complete training lineage. Do not fetch their
whole class-agnostic annotation pack, which has not been proven Val/Train-only.
The [official conference abstract](https://www.sigmm.org/opentoc/MM2023-TOC)
confirms the self-training method, not exact supervision splits. Full paper or
model training evidence was requested; no source claim is inferred from an
unavailable full text. Prefer small provenance/geometry summaries before a
conditional 329-MB stream or 542-MB checkpoint transfer. Do not repeat the
completed score scan, copy archives, start full extraction or broaden the GT
representation exception into decision-learning main work without clarification.

Delivered reports: `GT_PILOT_BASELINE_TABLE.md`, `GT_REPRESENTATION_PILOT.md`,
`GT_CONTROLLED_CORRECTION_R1.md`, and append-only `RESEARCH_LOG.md`; public
JSON/CSV contain real bounded metrics, source/config/asset hashes and resources.
Raw descriptors, GT rows, checkpoints and peer logs/AGENTS remain private.

Subsequent M1 source evidence: ten pinned official SimOWT files (169,655 B)
were individually recovered privately and hash-verified. The COCO-named mapper
can route to TAO Train; Known-filter and original-list-retaining auxiliary
generators are different paths, neither bound to the released checkpoint.
NAS current idol/entrypoint/registry are not byte-identical to upstream despite
the same HEAD; training YAML is identical. See M1_SIMOWT_TRAINING_CHAIN.md and
simowt_official_training_source.json. Latest regression 131 tests; primary
remains unqualified. No new model/data/inference or correction round.

Subsequent alternative preflight: 15 pinned MASA/AED source/config/docs files
(96,708 B) recovered privately with Git blob/SHA256 verification. A conditional
MASA-SAM-B native anonymous proposal route is now specified, not executed or
qualified. Default public-Detic config and future-dependent demo filtering
cannot be reused as the clean causal frontend. Two optional weights have a
metadata-only 933,925,642-B ceiling; no weights or environment installed.
Minimal binary-wheel lock and actual release tensor coverage come first;
then at most eight Train images under an 8-GiB incremental stop ceiling.
See M1_MASA_AED_PREFLIGHT.md. Latest regression: 137 pass. No training, Test,
R2, full Val/cache or claim that AED's frequency-base filter proves legality.
