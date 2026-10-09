"""TrackOCD v2: a clean benchmark and model namespace.

The v2 namespace deliberately does not import the historical Phase19--90
controllers.  Adapters may depend on them later, but the benchmark contract
and evaluator remain independent.
"""

__version__ = "0.1.0"
