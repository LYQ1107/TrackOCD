# Phase76R preregistration — Phase75 contract errata and Pareto audit

This audit is TRAIN/validation-only and does not change any Phase75 artifact,
candidate universe, denominator, seed or evaluator. It reads every 500-step
Phase75E formal checkpoint (four folds, 120 files) and evaluates exact p16
R-global and R-legal scores with the frozen Phase75D protocol. No model
selection, held outcome, controller, StateMemory, DEV+, Q1, public-new or
sealed input is used.

The historical teacher-authorizer is rechecked with its intended global
fold-R@1 guard. Phase75E's actual optimizer and per-fold seeds are recorded as
errata. A Pareto window is diagnostic only: global/ legal unsafe=0,
global ΔR@1≥−0.005, global ΔmAP≥−0.002, legal ΔR@1>0, legal ΔmAP>0 and mean
adapted/raw cosine≥0.98. Absence or presence of this window does not authorize
the old adapter or a controller.

