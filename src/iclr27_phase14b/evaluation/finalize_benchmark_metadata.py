"""Attach the preregistered public-checkpoint execution ledger."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
path = ROOT / "outputs/iclr27_phase14b/eval/foundation_feature_benchmark.json"
d = json.loads(path.read_text())
d["checkpoint_execution"] = {
    "object_centric_entity": {
        "status": "no_compatible_local_checkpoint",
        "candidates": {
            "TrackVerse": {"status": "not_executed", "reason": "official release is a large tracked-object dataset/recipe; no compatible frozen encoder checkpoint was available locally; downloading data/checkpoint was not preregistered under the storage budget"},
            "VESSA": {"status": "not_executed", "reason": "official code is an adaptation/training pipeline; no compatible local checkpoint and no verified one-shot crop encoder interface"},
            "SRL": {"status": "not_executed", "reason": "official MOVi/YTVIS slot checkpoints are not local and require the released object-centric training/data pipeline"},
        },
    },
    "streaming_video_semantic": {
        "status": "no_local_compatible_checkpoint",
        "candidates": {
            "InternVideo2.5/3": {"status": "not_executed", "reason": "no local checkpoint; InternVideo3 public option is an 8B instruction model and no local transformers/runtime was available; no large download under storage budget"},
            "StreamFormer": {"status": "not_executed", "reason": "official README lists a Hugging Face checkpoint but the repository retains checkpoint/data-release TODOs; no local checkpoint/runtime was available, and no download was used"},
        },
    },
    "tracking_aware_correspondence": {
        "status": "representative_checkpoint_loaded",
        "candidates": {
            "OVTR": {
                "status": "checkpoint_loaded_runtime_forward_blocked",
                "repository": "https://github.com/jinyanglii/OVTR",
                "checkpoint": "data/iclr27_phase14b/checkpoints/ovtr_5_frame.pth",
                "details": "Official model built and 239 MB checkpoint loaded on one idle GPU using the ovtr environment. The one-frame forward then exposed a repository runtime/config contract (model.ious_thresh is assigned by the full evaluator, not the model constructor); no source or evaluator was modified.",
            },
            "COVTrack": {
                "status": "checkpoint_not_executed_environment_blocked",
                "repository": "https://github.com/zekunqian/COVTrack",
                "checkpoint": "data/iclr27_phase14b/checkpoints/covtrack_ctao_public.pth",
                "details": "The project ovtrack environment has MMCV 2.1 without mmcv.parallel; the compatible MMCV 1.7.1 environment lacks the required OpenAI clip package. A single-frame smoke therefore stopped before model construction.",
            },
            "SimOWT": {"status": "not_executed", "reason": "public checkpoint is a detector/tracker pipeline, not a head-agnostic semantic encoder; running it would require a separate proposal-stream audit"},
            "MoSiC": {"status": "not_executed", "reason": "official checkpoint links are external and no local compatible checkpoint was present; its dense motion objective is retained as a prior, not silently substituted"},
            "TRACT/TraCLIP": {"status": "not_executed", "reason": "official repository has no verified compatible local checkpoint/interface for this GT-box causal benchmark"},
        },
    },
    "q1_used": False,
    "devplus_labels_used_for_candidate_selection": False,
    "selection_rule": "all executable local candidates plus the preregistered official-family ledger; no Q1 replay",
}
d["not_executed_yet"] = {
    "object_centric_entity": ["TrackVerse (no compatible encoder checkpoint)", "VESSA (no local checkpoint/interface)", "SRL (no local checkpoint/interface)"],
    "streaming_video_semantic": ["InternVideo2.5/3 (no local checkpoint/runtime; 8B option not feasible)", "StreamFormer (no local checkpoint/runtime)"],
    "tracking_aware_correspondence": ["COVTrack (MMCV/clip environment blocker)", "SimOWT (proposal pipeline, not encoder)", "MoSiC/TRACT (no local compatible checkpoint)"],
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(d, indent=2))
tmp.replace(path)
print("updated", path)
