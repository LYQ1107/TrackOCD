# M0 — TrackOCD core asset and protocol audit

Date: 2026-10-09 (Asia/Shanghai). Engineering audit: **PASS**. Recovery of
historical research assets: **BLOCKED**. Algorithm hypotheses: **NOT TESTED**.
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
was changed. The missing NAS connection fields are HostName/User/Port and,
if present, the jump/proxy route from the user's `Host nas57` configuration.

On this follow-up, the worktree was clean before documentation edits and the
same 15 tests passed again. About 91 GiB remains free. Training, large feature
extraction and scientific evaluation have still not started; M0 remote
delivery and current NAS asset verification remain required before advancing.
