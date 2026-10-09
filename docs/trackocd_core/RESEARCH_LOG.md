# TrackOCD core research log

## 2026-10-10 — Exact v2 recovery (M0)

- Question: did the new server restore the actual later v2, including atomic
  formal shards / cross-track DINO batching / prefix-level resume, or only the
  older published source?
- Evidence: targeted fetch of NAS recovery `604d7eff`; all 81 NAS blob hashes
  equal Git tree and A100 bytes (642,256 bytes). NAS worktree HEAD `24907fe1`
  and index/source before-after digests unchanged. 50 files added, nine
  updated, 22 identical; main, original v2 and recovery stashes preserved.
- Migration failures: first synthetic run 43 passed / two failed on old-root
  assumptions. Keep recovered source bytes unchanged; isolate only the test
  environment in a separate conftest. Regression: 48 passed.
- Findings: actual batching/prefix resume and fixed GT denominator are back.
  Legacy Known Hungarian remapping and contaminated token scoring remain;
  M3 must implement a separate evaluator, not rewrite past results.
- Decision: retain exact source recovery. No algorithm hypothesis tested.
  Next: M1 lawful frontend quality/causality and source-metadata qualification.
  No new training, feature extraction, TAO Test access or foreign process action.

## 2026-10-10 — M1 frozen-reference quality and coverage

- Question: can a present frozen physical stream support Novel discovery?
- No model/inference/sweep: sequential read of all 988 existing BT-FG-0 NPZ
  track-ID columns. Counts match frozen runtime; canonical prediction SHA
  matches export. Median length 10, single-observation 10.19%, <=4 29.38%.
- Fixed category-free temporal-IoU >=.5 / per-video Hungarian coverage:
  Known 1442/4413 (32.68%), Novel 30/819 (3.66%). Full GT denominators, not
  matched-only accuracy. Counts agree with canonical export; complete NPZ
  box-byte equality was not established. Labels never enter method inputs.
- Source continuation: five Val-only NAS audit JSON files recovered exactly
  (260,702 bytes); historical run log loads [296,512] DetPro text features.
  Current selected vocabulary includes 203/209 TrackOCD Novel classes by
  synset mapping. Association toggle cannot remove detector dependence.
- Decision: retain PANDAS and COVTrack as references; no primary freeze.
  Native clean declarations are contradicted: FAIL_NO_NOVEL_VOCABULARY /
  INCOMPARABLE as core input. Cache byte validity is distinct from legality.
  Exact old config/prompt hashes and complete supervision remain unverified.
  M1 is BLOCKED_FRONTEND_QUALITY, not proof that every physical route fails.
- Next: small SimOWT source/config/fragmentation follow-up; meanwhile the
  expressly allowed GT-feasibility route can progress, not a predicted claim.
