"""Phase74 read-only asset/replay contract audit.

The package deliberately contains no model training or semantic/controller
code.  It operates on frozen Q0 and Phase19R artifacts and keeps evaluator
metadata out of model-facing records.
"""

__all__ = ["io", "manifest_reader", "prefix_contract", "asset_identity",
           "mapping", "failure_taxonomy", "q0_lineage_exporter",
           "tracklet_alignment", "q0_dependency_audit", "gates"]
