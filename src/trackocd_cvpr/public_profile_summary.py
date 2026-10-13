"""Export only preregistered project function timing, never profiler callers.

No raw .prof/pickle, locals, arguments, route IDs or private trace export.
Cumulative time overlaps between caller/callee; do not sum it as wall time.
"""
import math
from pathlib import Path
import re


def public_profile_summary(stats, root, allowed_sources, *, max_rows=50):
    root = Path(root).resolve()
    allowed = {str((root / path).resolve()): path for path in allowed_sources}
    if type(max_rows) is not int or not 1 <= max_rows <= 50:
        raise ValueError('Small fixed public function timing limit')
    rows, total_self, total_calls = [], 0., 0
    for key, value in stats.items():
        if type(key) is not tuple or len(key) != 3 or len(value) != 5:
            raise ValueError('Exact cProfile function-stat framing')
        filename, line, name = key
        primitive, calls, self_seconds, cumulative_seconds, _ = value
        if (any(type(x) is not int or x < 0 for x in (primitive, calls))
                or any(not math.isfinite(x) or x < 0 for x in (self_seconds, cumulative_seconds))):
            raise ValueError('Finite nonnegative profile counters')
        total_self += self_seconds; total_calls += calls
        # Never emit unknown filename/function strings, callers or values.
        if (filename in allowed and type(line) is int and line > 0
                and type(name) is str and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name)):
            rows.append({'source': allowed[filename], 'line': line, 'function': name,
                'primitive_calls': primitive, 'total_calls': calls,
                'self_seconds': self_seconds, 'cumulative_seconds': cumulative_seconds})
    rows.sort(key=lambda row: (-row['cumulative_seconds'], row['source'], row['line'], row['function']))
    return {'top_registered_functions_by_overlapping_cumulative_time': rows[:max_rows],
        'all_profiled_functions_self_seconds': total_self, 'all_profiled_calls': total_calls,
        'registered_function_rows_before_display_bound': len(rows),
        'cumulative_times_overlap_not_additive_wall_time': True,
        'raw_profiler_callers_arguments_locals_or_private_strings_exported': False,
        'instrumented_policy_timing_not_full_dev_bottleneck_proof_or_isolated_speedup': True}
