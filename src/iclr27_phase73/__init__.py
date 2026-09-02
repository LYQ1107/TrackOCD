"""Phase73 Q0 lineage/exporter contract.

The package is intentionally isolated from the historical Phase19R/Phase71
runtime.  It contains only read-only audit and plumbing helpers; it does not
train a model or alter the frozen Q0 physical stream.
"""

__all__ = ["contracts", "io", "alignment", "export"]
