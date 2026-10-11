# Scientific conclusion — real negative result

1. **Why previous Known collapse?** Old R0/R1 had tiny four-class/12-track
   fitting, poorer prototype-score AUROC and heavy wrong-Known assignment.
   This implicates learned open-set geometry as well as thresholds; it does
   not uniquely prove a loss/capacity cause. Old results are unchanged and
   numerically incomparable with this independent enlarged protocol.
2. **Did more legal Train improve open-set geometry?** Overall p16 Rank1 and
   Known/pseudo AUROC improve, but primarily on fitting Known. Four sparse
   heldout pseudo categories / 16 queries all have cross-video positives;
   pseudo macroRank1 raw .3500 vs A1/A2 .3083 at p16, and learned top1 is lower
   at all five prefixes. Partial Recall@K/AUROC gains do not make a stable
   unseen-geometry PASS. Effective rank falls roughly54.18→8.59.
3. **Is Adapter better than frozen DINO?** Known recognition and Standard H
   improve in controlled experiments, not universal unseen/persistent
   discovery. See actual tables, role analysis and paired errors. A1 fixes
   many Known errors but pseudo-Novel net correct count decreases1.167 per
   p16case. Do not turn Seen-Known probe improvement into Novel success.
4. **Independent temporal contribution?** Not supported: A2 p16 H .5136 vs
   same-capacity static .5413 and A1 .5276, CT .0794 vs static .0899 / A1
   .1693; A1→A2 adds26.083 errors while fixing5.583, whereas static introduces
   fewer errors. These are mean counts, not independent categories.
5. **Does risk-aware decision beat MLP?** Not established: every1800controlled
   learned-policy case has zero pureCT. D2 lower wrongKnown can accompany
   contaminated merging; matched coverage is mandatory. Full Val fixed
   pair sign counts below are separate observations, not a rescue fit.
6. **WAIT safer or merely rejection?** Train noWAIT FM .3333 vs FULL .3003
   while reuse coverage1→.9669 and all-track commitment1→.9152. This alone is
   not a matched-risk benefit. Full Val reports both actual coverage gaps;
   no favorable interpolation or infinite WAIT. NoMemory/reset remove reuse,
   so their low mergerate is not safe persistent discovery by itself.
7. **Cross-individual/video anonymous reuse?** Audited raw baselines can;
   learned policy has zero pureCT on controlled holdout. Offline Standard
   Hungarian can still count an impure token as New-correct, so that score
   is not live correct memory reuse. All840fullVal seals/2040logical reports
   are completed, including unknown/unmatched memory members.
8. **GT vs predicted gap?** Compare the completed tables below. Train207/16
   targets vs full Val4413/819, category support, physical noise/shortness,
   15→48prototype deployment shift and304561predicted IDs differ. Their
   numeric differences are descriptive, not an isolated frontend causal
   experiment. Selected Train mean available observations is
   11.3703 vs Native
   4.2491; match-conditioned
   lengths and Known prototype/physical coverage are in the actual posthoc
   descriptive diagnostic, not causal interventions. No GT repair or
   counterfactual retraining was performed.
9. **Physical restriction?** Known reliable1400/4413,Novel189/819. Main fixed
   reuse527 has118reliable current opportunities,409missed; only87have an
   earlier reliable same-category match. Other orders are in the support
   table. Strict both>=16support averages13.17%, **not** an absolute ceiling
   for true clipped short-track replay; any-observation support is larger.
   Unknown membership and semantic errors can reduce purity further.
10. **Enough for a new-method TrackOCD paper?** No supported core novelty
   claim currently: unseen geometry, independent temporal benefit and safe
   persistence are not established in controlled tests; Native/foundation
   provenance/coverage remains incomplete. This is a reproducible research
   negative result, not a publication-success claim or permission for rescue
   tuning. Any future loss, data or frontend intervention needs a new explicit
   hypothesis/authorization; no automatic extra experiment is launched.

## Full Val fixed p16 comparison sign counts

```json
[
  {
    "left": "A1_SELECTED",
    "right": "A2_CAPACITY_CONTROL",
    "cases": 12,
    "H_positive": 5,
    "CT_positive": 0,
    "matched_coverage_cases": 12,
    "matched_CT_gain_or_FM_reduction_without_more_wrongKnown": 4
  },
  {
    "left": "A1_SELECTED",
    "right": "A2_EVIDENCE",
    "cases": 12,
    "H_positive": 4,
    "CT_positive": 0,
    "matched_coverage_cases": 12,
    "matched_CT_gain_or_FM_reduction_without_more_wrongKnown": 7
  },
  {
    "left": "B1_track_nearest",
    "right": "FULL",
    "cases": 12,
    "H_positive": 0,
    "CT_positive": 0,
    "matched_coverage_cases": 12,
    "matched_CT_gain_or_FM_reduction_without_more_wrongKnown": 0
  },
  {
    "left": "B2_track_dpmeans",
    "right": "FULL",
    "cases": 12,
    "H_positive": 0,
    "CT_positive": 0,
    "matched_coverage_cases": 12,
    "matched_CT_gain_or_FM_reduction_without_more_wrongKnown": 0
  },
  {
    "left": "D1_SIMPLE_MLP",
    "right": "D2_RISK_AWARE",
    "cases": 12,
    "H_positive": 0,
    "CT_positive": 0,
    "matched_coverage_cases": 12,
    "matched_CT_gain_or_FM_reduction_without_more_wrongKnown": 0
  },
  {
    "left": "WITHOUT_RISK",
    "right": "FULL",
    "cases": 12,
    "H_positive": 0,
    "CT_positive": 0,
    "matched_coverage_cases": 12,
    "matched_CT_gain_or_FM_reduction_without_more_wrongKnown": 0
  },
  {
    "left": "WITHOUT_TEMPORAL",
    "right": "FULL",
    "cases": 12,
    "H_positive": 0,
    "CT_positive": 0,
    "matched_coverage_cases": 12,
    "matched_CT_gain_or_FM_reduction_without_more_wrongKnown": 0
  }
]
```

Three training seeds/four orders are not independent heldout categories.
Conditional500video bootstrap fixes global mapping and memory stream; it is
not a rerun/causal CI. Mean H averages actual H, never H of mean Old/New.
Unmatched does not mean proven background/pollution; unknown cannot certify
pure Novel. PHE INCOMPARABLE and historical M1 blocked are retained.
