"""Write Phase 4M open-source audit CSVs and docs.

Every repository below was cloned and pinned at the recorded commit in
third_party/research_refs_phase4m during this phase.  License strings
were read from the cloned LICENSE file or README; "none detected" means
no license file/metadata was present in the clone.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs" / "iclr27_phase4m" / "open_source"
DOC = ROOT / "docs" / "iclr27_phase4m"

INVENTORY = [
    {
        "method": "ml_edm",
        "paper": "Early Classification of Time Series: A Survey and Benchmark",
        "year": 2025, "venue": "TMLR",
        "repo": "ML-EDM/ml_edm",
        "commit": "93ffd9380dc80b30f258e833ad847aa2bdb0559b",
        "license": "BSD-3-Clause (OpenReview); LICENSE file placeholder TODO",
        "strict_online": "yes (inference online, training offline)",
        "future_access": "no",
        "abstention": "yes (trigger defers the class decision)",
        "temporal_evidence": "yes (chronological classifiers over prefixes)",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "no",
        "relevant_files": "src/ml_edm/early_classifier.py;src/ml_edm/trigger/_stopping_rule.py;src/ml_edm/trigger/_calimera.py",
        "relevant_functions": "EarlyClassifier;StoppingRule;Calimera halters",
        "relevance": "separates decision time from prediction; accuracy-latency stopping rules",
        "used": "principle only",
        "why": "static-class early decision; no online class birth or open-world tracking",
    },
    {
        "method": "FIRMBOUND",
        "paper": "FIRMBOUND: Optimal Early Classification of Sequential Data under Finite-Horizon Constraints",
        "year": 2025, "venue": "ICLR",
        "repo": "Akinori-F-Ebihara/FIRMBOUND",
        "commit": "58e87f9138674155c1ff34ba961c910d64a8977c",
        "license": "MIT",
        "strict_online": "yes (sequential inference)",
        "future_access": "no",
        "abstention": "yes (delays decision until a learned stopping boundary)",
        "temporal_evidence": "yes (density-ratio / SPRT evidence over prefixes)",
        "delayed_class_assignment": "yes (decision postponed to deadline)",
        "dynamic_class_birth": "no",
        "relevant_files": "backward_induction_cfl_train.py;backward_induction_gp_train.py;density_ratio_estimation_main.py",
        "relevant_functions": "backward induction; SPRT-style dynamic thresholds",
        "relevance": "finite-horizon optimal stopping for sequential identity",
        "used": "principle only",
        "why": "single static class with deadline; no physical tracking / open-set birth",
    },
    {
        "method": "SPEED",
        "paper": "SPEED: Selective Prediction for Early Exit Deep Neural Networks",
        "year": 2024, "venue": "arXiv (no venue in repo)",
        "repo": "Div290/SPEED",
        "commit": "5c48dff57ffd9b1d44e81e02e15d6d84ed53ee96",
        "license": "none detected",
        "strict_online": "yes (inference-time early exit)",
        "future_access": "no",
        "abstention": "yes (deferral labels for hard samples)",
        "temporal_evidence": "partial (early exits per token/time step)",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "no",
        "relevant_files": "train.py;model.py;param.py",
        "relevant_functions": "deferral label generation; exit-selection cost",
        "relevance": "accuracy-latency deferral via early exits",
        "used": "principle only",
        "why": "static classifier exits; no identity resolution or memory",
    },
    {
        "method": "scikit-fallback",
        "paper": "scikit-fallback: selective prediction and reject option library",
        "year": "2024-2026", "venue": "library",
        "repo": "sanjaradylov/scikit-fallback",
        "commit": "d7910c256107ea7065f70c6a037a55d0419e2c93",
        "license": "BSD-3-Clause",
        "strict_online": "partial (scikit-learn fit/predict)",
        "future_access": "no",
        "abstention": "yes (fallback/reject label, multi-threshold)",
        "temporal_evidence": "no",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "no",
        "relevant_files": "skfb/estimators/_rule.py;skfb/ensemble/_routing.py;skfb/experimental/enable_multi_threshold_fallback_classifier_cv.py",
        "relevant_functions": "FallbackRuleClassifier; threshold cascade routing",
        "relevance": "coverage/risk API for reject-option policies",
        "used": "principle only",
        "why": "static per-sample abstention; no causal retry on the same object",
    },
    {
        "method": "boxmot",
        "paper": "BoxMOT: library of online MOT trackers",
        "year": "2022-2026", "venue": "library",
        "repo": "mikel-brostrom/boxmot",
        "commit": "b23bf5f453d57c3fa3243e6648af6ea6738575b4",
        "license": "AGPL-3.0",
        "strict_online": "yes (online MOT)",
        "future_access": "no",
        "abstention": "no",
        "temporal_evidence": "no",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "no",
        "relevant_files": "boxmot/trackers/bbox/bytetrack.py;boxmot/trackers/bbox/occluboost.py",
        "relevant_functions": "unconfirmed tracks; tentative_max_age",
        "relevance": "physical tentative/confirmed state only",
        "used": "principle only",
        "why": "semantic deferral is not physical track confirmation; AGPL not copied",
    },
    {
        "method": "l2d_pop",
        "paper": "Learning to Defer to a Population with Limited Expert Demonstrations",
        "year": 2025, "venue": "IEEE (document 11302495)",
        "repo": "nil123532/learning-to-defer-to-a-population-with-limited-demonstrations",
        "commit": "8613eba0ae8ca1e1f545324f67ad708769daeeb9",
        "license": "none detected",
        "strict_online": "no (offline training)",
        "future_access": "no",
        "abstention": "yes (learned deferral to population)",
        "temporal_evidence": "no",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "no",
        "relevant_files": "l2d-pop/;train_generated_experts_CHOSEN_DATASET.py",
        "relevant_functions": "pop_attn / pop / single deferral losses",
        "relevance": "deferral to experts; not online identity resolution",
        "used": "not used",
        "why": "human-expert deferral, static classes, offline",
    },
    {
        "method": "sc_likelihood_ratios",
        "paper": "Know When to Abstain: Optimal Selective Classification with Likelihood Ratios",
        "year": 2025, "venue": "ICLR",
        "repo": "clear-nus/sc-likelihood-ratios",
        "commit": "89aa91b711798674751ebdafde1394bab6ee0e1e",
        "license": "MIT",
        "strict_online": "yes (inference-time selector)",
        "future_access": "no",
        "abstention": "yes (likelihood-ratio selector)",
        "temporal_evidence": "no",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "no",
        "relevant_files": "calculate_selector_scores.py;calculate_risk_coverage_curve.py",
        "relevant_functions": "selector score; risk-coverage curve",
        "relevance": "optimal abstention scoring and coverage-risk curves",
        "used": "principle only",
        "why": "single-sample abstention; no retry on the same object",
    },
    {
        "method": "sc_gap",
        "paper": "What Does It Take to Build a Performant Selective Classifier?",
        "year": 2025, "venue": "arXiv 2510.20242",
        "repo": "cleverhans-lab/sc-gap",
        "commit": "9417d3245a6241823a3b13bad148e75a2d9b1041",
        "license": "none detected",
        "strict_online": "yes (scoring)",
        "future_access": "no",
        "abstention": "yes (coverage-risk analysis)",
        "temporal_evidence": "no",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "no",
        "relevant_files": "eval_arch.py;eval_shift.py;eval_cifar_c.py",
        "relevant_functions": "gap decomposition; ranking vs calibration",
        "relevance": "coverage-uniform gap; monotone calibration cannot fix ranking",
        "used": "principle only",
        "why": "theory of selective gap; no temporal identity evidence",
    },
    {
        "method": "uq_for_deferral",
        "paper": "Is UQ a Viable Alternative to Learned Deferral?",
        "year": 2026, "venue": "Springer (2026)",
        "repo": "annawundram/UQforDeferral",
        "commit": "6a86a0d770c832a3bd76b1f140af3b29357799e5",
        "license": "none detected",
        "strict_online": "yes (inference-time UQ)",
        "future_access": "no",
        "abstention": "yes (UQ-based deferral vs learned deferral)",
        "temporal_evidence": "no",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "no",
        "relevant_files": "models/;plots/evaluation.ipynb",
        "relevant_functions": "UQ scores; deferral comparison",
        "relevance": "uncertainty score as deferral trigger",
        "used": "principle only",
        "why": "medical-image deferral; no tracking loop",
    },
    {
        "method": "StopAndHop",
        "paper": "Stop&Hop: Early Classification of Irregular Time Series",
        "year": 2022, "venue": "CIKM",
        "repo": "thartvigsen/StopAndHop",
        "commit": "b9c3e01152f5189810556c6c34976854a8287a96",
        "license": "none detected",
        "strict_online": "yes (halting at inference)",
        "future_access": "no (training may use full series)",
        "abstention": "yes (halting policy delays decision)",
        "temporal_evidence": "yes (halting point over irregular time series)",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "no",
        "relevant_files": "src/modules.py;src/model.py",
        "relevant_functions": "stopPolicy; hopPolicy; getReward",
        "relevance": "learned halting for early decisions",
        "used": "principle only",
        "why": "no semantic identity, no open-set birth, no MOT",
    },
    {
        "method": "LTC",
        "paper": "Learning through Creation: A Hash-Free Framework for On-the-Fly Category Discovery",
        "year": 2026, "venue": "CVPR Findings",
        "repo": "brandinzhang/LTC",
        "commit": "44584bfddae5e6b82bbc182f68588a05e45365bb",
        "license": "none detected",
        "strict_online": "yes (prototype creation at inference)",
        "future_access": "no",
        "abstention": "no",
        "temporal_evidence": "no",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "yes (PrototypeLayer.add_prototype)",
        "relevant_files": "LTCmodel.py;config.py;mkee.py",
        "relevant_functions": "PrototypeLayer.add_prototype; adaptive threshold; MKEE pseudo-unknowns",
        "relevance": "dynamic class birth at inference",
        "used": "principle only",
        "why": "image-level immediate birth; no track-local deferral, no MOT",
    },
    {
        "method": "TALON",
        "paper": "TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery",
        "year": 2026, "venue": "CVPR",
        "repo": "ynanwu/TALON",
        "commit": "4091c2df89100974d316ce05659bc25654fcff1e",
        "license": "none detected",
        "strict_online": "yes (test-time stream adaptation)",
        "future_access": "no",
        "abstention": "no",
        "temporal_evidence": "no",
        "delayed_class_assignment": "no",
        "dynamic_class_birth": "yes (NCM prototypes + dynamic tau)",
        "relevant_files": "methods/talon/trainer.py;methods/talon/utils.py;methods/talon/model.py",
        "relevant_functions": "_estimate_dynamic_tau; model_tta; build_ncm_prototypes",
        "relevance": "adaptive threshold over a test-time stream",
        "used": "principle only",
        "why": "no physical tracking, no track-local unresolved state, no identity deferral",
    },
]

MATRIX = [
    ("ml_edm", "yes", "yes", "no", "no", "yes", "no",
     "trigger/stopping probability", "principle only",
     "decision-time / prediction separation; accuracy-latency stopping"),
    ("FIRMBOUND", "yes", "yes", "yes", "no", "yes", "no",
     "learned SPRT boundary", "principle only",
     "finite-horizon optimal stopping; delayed class assignment"),
    ("SPEED", "yes", "partial", "no", "no", "yes", "no",
     "exit deferral label", "principle only",
     "early-exit deferral cost"),
    ("scikit-fallback", "yes", "no", "no", "no", "partial", "no",
     "fallback/reject label", "principle only",
     "coverage-risk policy API"),
    ("boxmot", "no", "no", "no", "no", "yes", "yes",
     "tentative/confirmed track state", "principle only",
     "physical confirmation is orthogonal to semantic deferral"),
    ("l2d_pop", "yes", "no", "no", "no", "no", "no",
     "learned deferral loss", "not used",
     "human-expert deferral; static classes"),
    ("sc_likelihood_ratios", "yes", "no", "no", "no", "yes", "no",
     "likelihood-ratio selector", "principle only",
     "optimal single-sample abstention and risk-coverage"),
    ("sc_gap", "yes", "no", "no", "no", "yes", "no",
     "selector score ordering", "principle only",
     "coverage-uniform gap; monotone calibration limits"),
    ("uq_for_deferral", "yes", "no", "no", "no", "yes", "no",
     "uncertainty scores", "principle only",
     "UQ as deferral trigger"),
    ("StopAndHop", "yes", "yes", "no", "no", "yes", "no",
     "halting policy", "principle only",
     "learned halting; early decision"),
    ("LTC", "no", "no", "no", "yes", "yes", "no",
     "PrototypeLayer.add_prototype", "principle only",
     "dynamic class birth at inference; no deferral"),
    ("TALON", "no", "no", "no", "yes", "yes", "no",
     "NCM prototypes + dynamic tau", "principle only",
     "adaptive class threshold; no track semantics"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)
    fields = list(INVENTORY[0].keys())
    with open(OUT / "repository_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(INVENTORY)
    with open(OUT / "mechanism_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo", "abstention", "temporal_evidence",
                    "delayed_assignment", "dynamic_birth", "strict_online",
                    "tracking_loop", "selector_signal", "verdict",
                    "phase4m_relevance"])
        w.writerows(MATRIX)
    _write_docs()
    print("OPEN_SOURCE_AUDIT_DONE", len(INVENTORY))


def _write_docs():
    (DOC / "OPEN_SOURCE_REPOSITORY_AUDIT.md").write_text(
        """# Phase 4M Open-Source Repository Audit

Scope: selective classification / learning-to-defer, temporal deferral
and early classification, sequential optimal stopping, online novel
class discovery with delayed assignment, open-world tracking, and MOT
tentative-track state.  Priority was given to 2025-2026 work.

All repositories were cloned into
`third_party/research_refs_phase4m/` and pinned; commit hashes, license
strings, and relevant files were verified from the clones (no
paper-abstract-only entries).  Phase 4L already audited PROB, mepu-owod,
OV-DQUO, FusionSORT, ARPL, osr_closed_set_all_you_need and several
unreleased placeholders; those conclusions are reused.

## Verified 2025-2026 repositories (this phase)

| Repo | Year/Venue | License | Mechanism relevant to Phase 4M |
|---|---|---|---|
| ML-EDM/ml_edm | 2025 TMLR | BSD-3 (declared; LICENSE placeholder TODO) | separates decision time from prediction; accuracy-latency stopping rules |
| Akinori-F-Ebihara/FIRMBOUND | 2025 ICLR | MIT | finite-horizon optimal stopping; SPRT-style dynamic boundaries |
| Div290/SPEED | 2024 arXiv | none detected | early-exit deferral / hard-sample deferral labels |
| sanjaradylov/scikit-fallback | 2024-2026 | BSD-3 | reject-option / fallback API; coverage-risk curves |
| mikel-brostrom/boxmot | 2022-2026 | AGPL-3.0 | physical track tentative/confirmed states |
| nil123532/learning-to-defer-... | 2025 IEEE | none detected | learned deferral to a population of experts |
| clear-nus/sc-likelihood-ratios | 2025 ICLR | MIT | likelihood-ratio selector; optimal abstention |
| cleverhans-lab/sc-gap | 2025 arXiv | none detected | coverage-uniform selective-classification gap |
| annawundram/UQforDeferral | 2026 Springer | none detected | uncertainty quantification vs learned deferral |
| thartvigsen/StopAndHop | 2022 CIKM | none detected | learned halting policy for early classification |
| brandinzhang/LTC | 2026 CVPR Findings | none detected | on-the-fly dynamic class birth (PrototypeLayer.add_prototype) |
| ynanwu/TALON | 2026 CVPR | none detected | test-time stream adaptation; NCM prototypes + dynamic tau |

## What is genuinely absent

No 2025-2026 repository implements **frame-online open-world
multi-object tracking with causal semantic deferral**: abstaining from
an irreversible EXISTING_NOVEL / NEW_NOVEL identity resolution while the
same physical object continues to be tracked and its soft semantic
evidence continues to affect association.  The closest pieces are
FIRMBOUND (delayed class decision with a deadline, no tracking/memory)
and ml_edm (decision-vs-prediction separation, static classes).
LTC/TALON confirm that *dynamic class birth* exists at inference, but
their birth is immediate and image-level: there is no track-local
unresolved state, no delayed assignment, and no MOT loop.

Conclusion: `NO_DIRECTLY_COMPATIBLE_EXTERNAL_METHOD`.  Phase 4M borrows
only principles: (i) decision time may differ from observation time
(ml_edm / FIRMBOUND), (ii) abstention must be scored and its
coverage-risk trade-off measured (sc-likelihood-ratios / scikit-fallback
/ sc-gap), and (iii) physical confirmation is a different axis from
semantic resolution (boxmot).
""")
    (DOC / "OPEN_SOURCE_IMPLEMENTATION_NOTES.md").write_text(
        """# Phase 4M Open-Source Implementation Notes

## ml_edm (TMLR 2025)

`EarlyClassifier` trains chronological classifiers for each prefix length
and a separate `trigger` (e.g. StoppingRule / Calimera halters) that
decides *when* to stop and emit a class decision.  Phase 4M adopts the
decision/prediction separation at the *identity* level: semantic
observation and association are immediate; identity resolution is
triggered only when evidence is decisive.

## FIRMBOUND (ICLR 2025)

Learns time-dependent SPRT boundaries via backward induction with CFL/GP
and density-ratio estimation.  Its delayed-decision-under-deadline idea
matches TrackOCD's `UNRESOLVED_NOVEL` retry loop, but FIRMBOUND has no
physical tracking, no open-set birth, and no memory pollution; we use
only the principle that waiting can be optimal and must be evaluated by
an accuracy-latency trade-off.

## SPEED / scikit-fallback / sc-likelihood-ratios / uq_for_deferral

These confirm the standard coverage-risk machinery: abstain on a
per-sample score, measure coverage vs error, and prefer scores with good
ranking (sc-gap shows monotone calibration cannot fix ranking).  Phase
4M's deferral score is *not* a trained black-box abstainer; it is a
causal geometry rule (margin / entropy / known-relative evidence)
re-evaluated frame by frame on the same physical track.

## boxmot

ByteTrack's `unconfirmed` tracks and OCCUBOOST's `tentative_max_age`
confirm that physical track confirmation is a separate axis.  Phase 4M
does not reuse their code (AGPL-3.0) and does not add physical
tentative/quarantine states; `UNRESOLVED_NOVEL` is purely semantic.

## No code copied

All repositories were used for mechanism review only.  No source code,
losses, or architectures were copied into TrackOCD.
""")


if __name__ == "__main__":
    main()
