"""Resource accounting for the continuous Phase88 supervisor.

The manager distinguishes file-backed memmap RSS from anonymous memory and only
requests a pause after a persistent RAM-floor breach.  It never kills a PID;
the caller owns any task-process lifecycle decisions.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


RAM_FLOOR_GIB = 31.25
RAM_BREACH_SAMPLES = 3
RAM_BREACH_INTERVAL_SECONDS = 30
MAX_GPUS = 4
DEFAULT_WORKER_GPU_MIB = 4096


def _meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if parts:
                result[key] = int(parts[0])
    except (OSError, ValueError):
        return result
    return result


def process_memory_detail(pid: int | None = None) -> dict[str, int]:
    """Return smaps_rollup values in KiB for a process.

    ``Pss`` and ``Anonymous`` are the values used for worker budgeting; RSS is
    retained for compatibility with earlier Phase88 traces.
    """
    target = int(pid if pid is not None else os.getpid())
    result = {
        "Rss": 0, "Pss": 0, "Private_Clean": 0, "Private_Dirty": 0,
        "Shared_Clean": 0, "Shared_Dirty": 0, "Anonymous": 0, "Swap": 0,
    }
    try:
        text = Path(f"/proc/{target}/smaps_rollup").read_text()
    except OSError:
        return result
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key in result:
            try:
                result[key] = int(value.strip().split()[0])
            except (IndexError, ValueError):
                pass
    return result


def _gpu_rows() -> list[dict[str, str]]:
    command = [
        "nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        raw = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    rows: list[dict[str, str]] = []
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        if len(row) >= 4:
            rows.append({"index": row[0].strip(), "memory_used": row[1].strip(),
                         "memory_free": row[2].strip(), "utilization": row[3].strip()})
    return rows


def _compute_gpu_processes() -> dict[int, list[dict[str, str]]]:
    command = [
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader",
    ]
    try:
        raw = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {}
    # UUID->index mapping is queried separately because nvidia-smi's compute
    # table intentionally reports UUID rather than the ordinal index.
    try:
        mapping_raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    uuid_to_index = {}
    for row in csv.reader(mapping_raw.splitlines(), skipinitialspace=True):
        if len(row) >= 2:
            uuid_to_index[row[1].strip()] = int(row[0].strip())
    used: dict[int, list[dict[str, str]]] = {}
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        if row and row[0].strip() in uuid_to_index:
            idx = uuid_to_index[row[0].strip()]
            values = [x.strip() for x in row]
            used.setdefault(idx, []).append({
                "uuid": values[0], "pid": values[1] if len(values) > 1 else "",
                "process_name": values[2] if len(values) > 2 else "",
                "used_memory": values[3] if len(values) > 3 else "",
            })
    return used


def discover_usable_gpus(
    own_pids: Iterable[int] = (),
    *,
    allow_shared: bool = True,
    worker_gpu_mib: int = DEFAULT_WORKER_GPU_MIB,
) -> list[int]:
    """Return idle GPUs and, when authorized, GPUs with measured free headroom.

    Sharing is conservative: every occupied GPU must still expose at least
    ``worker_gpu_mib`` free memory.  No external process is inspected beyond
    read-only ``nvidia-smi`` output and no process is ever terminated.
    """
    own = {int(x) for x in own_pids}
    occupied = _compute_gpu_processes()
    idle_result: list[int] = []
    shared_result: list[int] = []
    for row in _gpu_rows():
        idx = int(row["index"])
        try:
            free_mib = int(row["memory_free"])
        except ValueError:
            continue
        processes = occupied.get(idx, [])
        external = [p for p in processes if p.get("pid", "").isdigit() and int(p["pid"]) not in own]
        if not external:
            if free_mib > 1024:
                idle_result.append(idx)
        elif allow_shared and free_mib >= int(worker_gpu_mib):
            shared_result.append(idx)
    return (idle_result + shared_result)[:MAX_GPUS]


def safe_worker_count(
    mem_available_kib: int,
    worker_rss_kib: int,
    free_gpu_count: int,
    max_workers: int = MAX_GPUS,
    floor_gib: float = RAM_FLOOR_GIB,
) -> int:
    floor_kib = int(floor_gib * 1024 * 1024)
    if worker_rss_kib <= 0 or mem_available_kib <= floor_kib:
        return 0
    computed = (mem_available_kib - floor_kib) // max(1, int(1.25 * worker_rss_kib))
    return max(0, min(int(max_workers), int(computed), int(free_gpu_count)))


@dataclass
class ResourceSnapshot:
    timestamp_utc: str
    mem_available_kib: int
    mem_total_kib: int
    worker_rss_kib: int
    worker_anon_kib: int
    worker_pss_kib: int
    free_gpu_indices: list[int]
    safe_workers: int
    shared_gpu_indices: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class ResourceManager:
    def __init__(self, worker_rss_kib: int, max_workers: int = MAX_GPUS):
        self.worker_rss_kib = max(0, int(worker_rss_kib))
        self.max_workers = int(max_workers)
        self.breach_count = 0

    def snapshot(self, worker_pid: int | None = None, own_pids: Iterable[int] = ()) -> ResourceSnapshot:
        info = _meminfo()
        detail = process_memory_detail(worker_pid)
        free = discover_usable_gpus(own_pids, allow_shared=True,
                                    worker_gpu_mib=DEFAULT_WORKER_GPU_MIB)
        idle = discover_usable_gpus(own_pids, allow_shared=False,
                                    worker_gpu_mib=DEFAULT_WORKER_GPU_MIB)
        safe = safe_worker_count(info.get("MemAvailable", 0), self.worker_rss_kib,
                                 len(free), self.max_workers)
        return ResourceSnapshot(
            timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            mem_available_kib=int(info.get("MemAvailable", 0)),
            mem_total_kib=int(info.get("MemTotal", 0)),
            worker_rss_kib=int(detail.get("Rss", self.worker_rss_kib)),
            worker_anon_kib=int(detail.get("Anonymous", 0)),
            worker_pss_kib=int(detail.get("Pss", 0)),
            free_gpu_indices=free,
            safe_workers=safe,
            shared_gpu_indices=[int(x) for x in free if x not in idle],
        )

    def register_breach(self, snapshot: ResourceSnapshot) -> bool:
        floor_kib = int(RAM_FLOOR_GIB * 1024 * 1024)
        if snapshot.mem_available_kib < floor_kib:
            self.breach_count += 1
        else:
            self.breach_count = 0
        return self.breach_count >= RAM_BREACH_SAMPLES
