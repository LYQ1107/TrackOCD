"""Independent Phase20 proposal-aware correspondence audit namespace.

The package deliberately does not import or mutate Phase19R controller code.
The Stage0/Stage1 command-line audits live in ``scripts/iclr27_phase20`` and
read frozen TRAIN-derived proposal/features by reference.
"""

PHASE = "iclr27_phase20"
PREFIXES = (1, 2, 4, 8, 16)
RELIABLE_IOU = 0.5

