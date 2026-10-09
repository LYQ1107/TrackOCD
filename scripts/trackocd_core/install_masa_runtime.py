#!/usr/bin/env python3
"""Install only the measured hash-locked wheels into a task-owned isolated env."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json
from scripts.trackocd_core.inventory_masa_wheels import ALLOWED_HOSTS

BASE_PYTHON = Path("/data3/liuyeqiang/.venvs/trackocd-a100/bin/python")


def allocated_bytes(roots: list[Path]) -> int:
    """Deduplicate hardlinked inodes; never follow links outside owned roots."""
    seen, total, pending = set(), 0, list(roots)
    while pending:
        path = pending.pop()
        try:
            stat = path.lstat()
        except FileNotFoundError:
            continue
        identity = stat.st_dev, stat.st_ino
        if identity in seen:
            continue
        seen.add(identity)
        total += stat.st_blocks * 512
        if path.is_dir() and not path.is_symlink():
            with os.scandir(path) as entries:
                pending.extend(Path(entry.path) for entry in entries)
    return total


def locked_requirement(record: dict) -> str:
    url = urllib.parse.urlparse(record["url"])
    digest = record["expected_sha256_from_lock"]
    if url.scheme != "https" or url.hostname not in ALLOWED_HOSTS or len(digest) != 64:
        raise ValueError("Unexpected wheel source/hash")
    if any(c not in "0123456789abcdef" for c in digest) or not url.path.endswith(".whl"):
        raise ValueError("Expected a pinned binary wheel")
    return record["name"] + " @ " + record["url"] + " --hash=sha256:" + digest + "\n"


def distribution_receipt(python: Path) -> dict:
    command = "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name'].lower().replace('_','-'): {'version':d.version,'direct_url':json.loads(d.read_text('direct_url.json') or '{}')} for d in m.distributions()}))"
    return json.loads(subprocess.check_output([str(python), "-c", command], text=True, timeout=30))


def matches_install_provenance(actual: dict, record: dict, operation: dict | None) -> bool:
    """PEP 610 archive hashes are optional in uv; never trust an empty one alone."""
    direct = actual.get("direct_url", {})
    if actual.get("version") != record["version"] or direct.get("url") != record["url"]:
        return False
    digest = record["expected_sha256_from_lock"]
    saved = direct.get("archive_info", {}).get("hashes", {}).get("sha256")
    if saved is not None:
        return saved == digest
    return bool(operation and operation.get("returncode") == 0
                and operation.get("url") == record["url"] and operation.get("sha256") == digest
                and operation.get("require_hashes") is True)


def verify_installed_record(python: Path, name: str) -> dict:
    """Stream installed RECORD digests, not a second archive download or import."""
    code = """
import base64, csv, hashlib, importlib.metadata as m, io, json, pathlib, sys
d=m.distribution(sys.argv[1]); root=pathlib.Path(sys.prefix).resolve()
count=total=0; errors=[]
for relative, encoded, expected_size in csv.reader(io.StringIO(d.read_text('RECORD') or '')):
    if not encoded:
        continue
    p=pathlib.Path(d.locate_file(relative))
    if not p.resolve().is_relative_to(root) or not p.is_file():
        errors.append(relative); continue
    algorithm, expected=encoded.split('=',1)
    if algorithm != 'sha256':
        errors.append(relative); continue
    h=hashlib.sha256(); size=0
    with p.open('rb') as reader:
        for block in iter(lambda: reader.read(1024**2), b''):
            h.update(block); size+=len(block)
    actual=base64.urlsafe_b64encode(h.digest()).rstrip(b'=').decode()
    if actual != expected or (expected_size and size != int(expected_size)):
        errors.append(relative)
    count+=1; total+=size
print(json.dumps({'hashed_files':count,'hashed_bytes':total,'mismatch_count':len(errors),'mismatches':errors[:8]}))
"""
    result = json.loads(subprocess.check_output([str(python), "-c", code, name], text=True, timeout=120))
    if not result["hashed_files"] or result["mismatch_count"]:
        raise ValueError("Installed RECORD integrity failure: " + name)
    return result


def recover_initial_torch_operation(record: dict, previous: dict, original_owner: dict) -> dict | None:
    """Narrow recovery of this installer's first post-success metadata failure."""
    if (record["name"] != "torch" or previous.get("error") != "Installed version/hash provenance mismatch: torch"
            or previous.get("status") != "BLOCKED_OWNED_INSTALL_PARTIAL_PRESERVED"
            or previous.get("current_operation") != "wheel_torch"
            or previous.get("installer_pid") != original_owner.get("installer_pid")):
        return None
    relative = Path(previous["task_temporary_directory"])
    if relative.is_absolute() or ".." in relative.parts or relative.parent != Path("outputs/trackocd_core/audit"):
        return None
    directory = ROOT / relative
    if (directory / "torch.requirements.txt").read_text() != locked_requirement(record):
        return None
    log = (directory / "wheel_torch.log").read_text()
    marker = "+ torch==" + record["version"] + " (from " + record["url"] + ")"
    if "Installed 1 package" not in log or marker not in log:
        return None
    # This error is raised only after run() has checked the child's zero exit.
    return {"url": record["url"], "sha256": record["expected_sha256_from_lock"], "returncode": 0,
            "require_hashes": True, "proof": "owned_installer_control_flow_and_saved_uv_success_log",
            "original_operation_log": str((relative / "wheel_torch.log")),
            "not_a_hash_read_from_direct_url_metadata": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-owned", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Explicit --execute required; no environment changed")
    started = time.monotonic()
    plan = json.loads((ROOT / "configs/trackocd_core/masa_sam_candidate_smoke.json").read_text())
    inventory = json.loads((ROOT / "outputs/trackocd_core/audit/masa_runtime_wheel_inventory.json").read_text())
    lock_path = ROOT / "outputs/trackocd_core/audit/masa_runtime_resolution/pylock.toml"
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if lock_sha != inventory["dependency_lock_sha256"] or not inventory["fits_candidate_ceiling"]:
        raise ValueError("Complete measured dependency lock required")
    target = Path(plan["runtime"]["isolated_destination"])
    if target != Path("/data3/liuyeqiang/.venvs/trackocd-masa-smoke") or target.is_symlink():
        raise ValueError("Unexpected isolated destination")
    owner_path = target / ".trackocd_masa_owner.json"
    original_owner = {}
    if target.exists():
        if not args.resume_owned or not owner_path.is_file():
            raise ValueError("Preserve existing destination; only verified task-owned resumption allowed")
        owner = json.loads(owner_path.read_text())
        original_owner = dict(owner)
        if owner["dependency_lock_sha256"] != lock_sha:
            raise ValueError("Owned environment belongs to a different lock")
        prior_cmdline = Path(f"/proc/{owner['installer_pid']}/cmdline")
        if prior_cmdline.exists() and b"install_masa_runtime.py" in prior_cmdline.read_bytes():
            raise ValueError("A matching installer is live; do not launch a second")
    else:
        owner = {"schema_version": "trackocd.core.masa_env_owner.v1", "dependency_lock_sha256": lock_sha}
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("Existing uv not available")
    task_tmp = Path(tempfile.mkdtemp(prefix="masa-install-", dir=ROOT / "outputs/trackocd_core/audit"))
    task_env = os.environ.copy()
    task_env.update(HTTPS_PROXY="http://127.0.0.1:17890", HTTP_PROXY="http://127.0.0.1:17890",
                    TMPDIR=str(task_tmp), UV_HTTP_TIMEOUT="60", UV_HTTP_RETRIES="2",
                    UV_CONCURRENT_DOWNLOADS="1", UV_CONCURRENT_INSTALLS="1", UV_NO_CONFIG="1")
    destination = ROOT / "outputs/trackocd_core/audit/masa_runtime_install.json"
    previous = json.loads(destination.read_text()) if args.resume_owned and destination.exists() else {}
    if previous and (previous.get("dependency_lock_sha256") != lock_sha or previous.get("environment") != str(target)):
        raise ValueError("Previous receipt does not belong to the owned environment/lock")
    if previous:
        atomic_json(task_tmp / "previous_install_receipt.json", previous)
    result = {"schema_version": "trackocd.core.masa_runtime_install.v1", "status": "RUNNING",
              "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "dependency_lock_sha256": lock_sha, "environment": str(target), "task_temporary_directory": str(task_tmp.relative_to(ROOT)),
              "base_environment": str(BASE_PYTHON.parent.parent), "base_distributions_before": distribution_receipt(BASE_PYTHON),
              "installer_pid": os.getpid(), "completed_packages": [],
              "peak_owned_allocated_bytes": previous.get("peak_owned_allocated_bytes", 0),
              "verified_install_operations": dict(previous.get("verified_install_operations", {})),
              "candidate_ceiling_bytes": plan["execution_limits"]["max_new_environment_source_weights_and_output_bytes"],
              "hash_required": True, "binary_only": True, "no_dependencies_or_indexes": True,
              "boundary": {"base_environment_mutated": False, "source_build": False,
                           "model_or_data_execution": False, "test_access": False, "foreign_process_interference": False}}
    roots = [target, task_tmp, ROOT / "checkpoints/masa_sam_candidate",
             ROOT / "outputs/trackocd_core/audit/frontend_alternatives_source",
             ROOT / "outputs/trackocd_core/audit/masa_candidate_source",
             ROOT / "outputs/trackocd_core/audit/masa_runtime_resolution"]
    retained = previous.get("retained_temporary_directories", [])
    if previous.get("task_temporary_directory"):
        retained = [*retained, previous["task_temporary_directory"]]
    result["retained_temporary_directories"] = sorted(set(retained))
    for relative in result["retained_temporary_directories"]:
        path = Path(relative)
        if path.is_absolute() or path.parent != Path("outputs/trackocd_core/audit") or not path.name.startswith("masa-install-"):
            raise ValueError("Unexpected retained task directory")
        roots.append(ROOT / path)

    def guard() -> None:
        used = allocated_bytes(roots)
        result["peak_owned_allocated_bytes"] = max(result["peak_owned_allocated_bytes"], used)
        if used > result["candidate_ceiling_bytes"] or shutil.disk_usage(ROOT).free < 4 * 1024**3:
            raise RuntimeError("Owned transient budget or disk headroom stop")
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        if int(mem["MemAvailable"].split()[0]) / int(mem["MemTotal"].split()[0]) < .25:
            raise RuntimeError("System RAM headroom stop")
        if time.monotonic() - started > 1800:
            raise RuntimeError("Bounded setup elapsed-time stop")

    def run(command: list[str], label: str, seconds: int = 900) -> None:
        guard()
        log = task_tmp / (label + ".log")
        result["current_operation"] = label
        atomic_json(destination, result)
        print(json.dumps({"operation": label, "status": "START"}), flush=True)
        with log.open("wb") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=task_env, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            operation_started = time.monotonic()
            try:
                while process.poll() is None:
                    guard()
                    if time.monotonic() - operation_started > seconds:
                        raise RuntimeError("Single package/setup operation time stop")
                    atomic_json(destination, result)
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError(label + " failed, exit " + str(process.returncode))
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)  # Only the session just created here.
                    process.wait(timeout=30)
                raise
        guard()

    try:
        if not target.exists():
            run([uv, "venv", "--no-project", "--no-python-downloads", "--no-cache", "--python", str(BASE_PYTHON), str(target)], "create_venv", 60)
        owner.update(installer_pid=os.getpid(), updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(owner_path, owner)
        probe = task_tmp / "same_fs_hardlink_probe"
        probe.write_bytes(b"trackocd-masa-owned")
        linked = target / (".same_fs_hardlink_probe_" + str(os.getpid()))
        os.link(probe, linked)
        if probe.stat().st_ino != linked.stat().st_ino or probe.stat().st_dev != linked.stat().st_dev:
            raise RuntimeError("Same-filesystem hardlink requirement failed")
        records = sorted(inventory["wheels"], key=lambda r: (r["name"] != "torch", -r["compressed_archive_bytes"], r["name"]))
        installed = distribution_receipt(target / "bin/python")
        for record in records:
            name = canonicalize_name(record["name"])
            existing = installed.get(name)
            if existing:
                operation = result["verified_install_operations"].get(name)
                if operation is None:
                    operation = recover_initial_torch_operation(record, previous, original_owner)
                    if operation:
                        result["verified_install_operations"][name] = operation
                if not matches_install_provenance(existing, record, operation):
                    raise ValueError("Preserve unexpected existing owned package: " + name)
                integrity = verify_installed_record(target / "bin/python", name)
                result["completed_packages"].append({"name": name, "version": existing["version"],
                    "sha256": record["expected_sha256_from_lock"], "installed_record_integrity": integrity,
                    "resumed_and_verified": True})
                atomic_json(destination, result)
                continue
            requirement = task_tmp / (name + ".requirements.txt")
            requirement.write_text(locked_requirement(record))
            run([uv, "pip", "install", "--python", str(target / "bin/python"), "--no-index", "--no-deps", "--require-hashes",
                 "--only-binary", ":all:", "--no-cache", "--link-mode", "hardlink", "-r", str(requirement)], "wheel_" + name)
            operation = {"url": record["url"], "sha256": record["expected_sha256_from_lock"], "returncode": 0,
                         "require_hashes": True, "proof": "owned_uv_hash_required_operation_completed",
                         "operation_log": str((task_tmp / ("wheel_" + name + ".log")).relative_to(ROOT))}
            result["verified_install_operations"][name] = operation
            atomic_json(destination, result)
            actual = distribution_receipt(target / "bin/python").get(name)
            if not actual or not matches_install_provenance(actual, record, operation):
                raise RuntimeError("Installed version/hash provenance mismatch: " + name)
            integrity = verify_installed_record(target / "bin/python", name)
            result["completed_packages"].append({"name": name, "version": actual["version"],
                "sha256": record["expected_sha256_from_lock"], "installed_record_integrity": integrity,
                "resumed_and_verified": False})
            atomic_json(destination, result)
        result["base_distributions_after"] = distribution_receipt(BASE_PYTHON)
        if result["base_distributions_before"] != result["base_distributions_after"]:
            raise RuntimeError("Base distribution snapshot changed")
        result["status"] = "PASS_HASH_LOCKED_ISOLATED_INSTALL_NOT_MODEL_COMPATIBILITY"
    except Exception as exc:
        result["status"] = "BLOCKED_OWNED_INSTALL_PARTIAL_PRESERVED"
        result["error"] = str(exc)
    result.update(elapsed_seconds=time.monotonic() - started, final_owned_allocated_bytes=allocated_bytes(roots),
                  completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(destination, result)
    print(json.dumps({k: result[k] for k in ("status", "peak_owned_allocated_bytes", "final_owned_allocated_bytes", "elapsed_seconds")}, indent=2), flush=True)
    if result["status"].startswith("BLOCKED"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
