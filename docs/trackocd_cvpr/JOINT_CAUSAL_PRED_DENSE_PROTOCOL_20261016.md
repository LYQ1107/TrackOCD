# Joint causal Pred+dense protocol — 2026-10-16

The protocol boundary is now frozen in `configs/trackocd_cvpr/joint_causal_pred_dense_protocol_v1.json`. It binds the already approved 407 Train video roles, the only allowed training-label roles (`fit_known` and `policy_train_pseudo`), Train-development and independent-final separation, strict online chronology, three mechanism families (R/C/M), three seeds, fixed causal-order and invariance probes, and the 600 GPU-hour / four-GPU / 25%-RAM safety limits.

The legal model-facing frontend is an anonymous COVTrack-native `train_predicted` stream. Its public LVIS vocabulary is an implementation input to the frozen frontend only; no TAO labels, Novel vocabulary, category/text fields, or GT join may enter the physical TrackOCD stream. The two-video dense smoke is complete and proves the input/output contract, but it is not strong frontend qualification and produces no OCD/MOT score.

The full predicted dense cache is registered as four fixed contiguous shards of the sorted actual approved video-ID list (ordinal sizes `102/102/102/101`; endpoint IDs `0–632`, `633–1447`, `1453–2190`, `2194–2987`). The ID-order hash and each shard hash are audited against the approved metadata plan; numeric video IDs are not assumed contiguous. This registration does not auto-start the long cache. Each video is atomic, completed videos must not be re-extracted, raw native/cache artifacts remain local, and only compact receipts are pushed.

At the time of this protocol freeze, formal policy/development and Goal A/B/C
acceptance remained closed until the full predicted input, source/provenance
receipts, separately trained/self-correcting M controls, fair
multi-seed/order/coverage evaluation, and lawful evaluator-side matching were
complete.

## State reconciliation — 2026-10-26

The four fixed anonymous dense shards have since completed and were reconciled:
`407` videos, `13` bounded subshards, `1,245,952` physical tracks,
`13,558,940` observations, and zero invalid boxes. The authoritative receipt is
`outputs/trackocd_cvpr/cvpr2/covtrack_train_dense_shards_v1/full_cache_reconciliation.json`
with SHA256
`532e53c40a3b04561438150d1d8d308a9ee8f2d642ad6502f494c0eb6e9a36d8`.
This closes cache coverage/provenance only. The foreground-mass transfer and
causal birth-confirmation diagnostics still do not qualify the frontend, so
formal predicted-track R/C/M training, OCD/MOT evaluation, and Goal A/B/C
acceptance remain closed.
