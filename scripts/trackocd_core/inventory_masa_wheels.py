#!/usr/bin/env python3
"""Inspect selected wheel ZIP directories via bounded HTTP ranges, not install."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import resource
import shutil
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

import tomli
from packaging.markers import Marker
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.trackocd_v2.io import atomic_json

ALLOWED_HOSTS = {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org", "download.openmmlab.com"}


def select_wheel(wheels: list[dict], ranks: dict) -> dict:
    choices = []
    for wheel in wheels:
        url = urllib.parse.urlparse(wheel["url"])
        if url.scheme != "https" or url.hostname not in ALLOWED_HOSTS:
            raise ValueError("Unexpected wheel source")
        filename = urllib.parse.unquote(url.path.rsplit("/", 1)[-1])
        _, _, _, tags = parse_wheel_filename(filename)
        compatible = [ranks[t] for t in tags if t in ranks]
        if compatible:
            choices.append((min(compatible), wheel["url"], wheel))
    if not choices:
        raise ValueError("No compatible binary wheel")
    return min(choices, key=lambda x: x[:2])[2]


def directory_totals(directory: bytes) -> dict:
    offset, count, logical, allocated = 0, 0, 0, 0
    while offset < len(directory):
        if directory[offset:offset + 4] != b"PK\x01\x02" or len(directory) - offset < 46:
            raise ValueError("Invalid ZIP central directory")
        compressed, unpacked = struct.unpack_from("<II", directory, offset + 20)
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", directory, offset + 28)
        next_offset = offset + 46 + name_len + extra_len + comment_len
        if next_offset > len(directory):
            raise ValueError("Truncated ZIP central entry")
        if unpacked == 0xFFFFFFFF or compressed == 0xFFFFFFFF:
            extra = directory[offset + 46 + name_len:offset + 46 + name_len + extra_len]
            pos, found = 0, False
            while pos + 4 <= len(extra):
                kind, size = struct.unpack_from("<HH", extra, pos)
                data = extra[pos + 4:pos + 4 + size]
                if kind == 1:
                    if unpacked == 0xFFFFFFFF:
                        unpacked, = struct.unpack_from("<Q", data)
                    found = True
                    break
                pos += 4 + size
            if not found:
                raise ValueError("Missing ZIP64 entry sizes")
        count += 1
        logical += unpacked
        allocated += ((unpacked + 4095) // 4096) * 4096
        offset = next_offset
    return {"archive_entries": count, "unpacked_logical_bytes": logical,
            "unpacked_4k_allocation_bytes": allocated}


def curl_metadata_request(url: str, byte_range: str | None, cap: int) -> tuple[int, dict, bytes]:
    """Use the observed working official curl transport, with strict body limits."""
    command = ["curl", "--fail", "--silent", "--show-error", "--max-time", "20",
               "--proto", "=https", "--suppress-connect-headers",
               "--proxy", "http://127.0.0.1:17890"]
    if byte_range is None:
        command += ["--head", url]
    else:
        command += ["--max-filesize", str(cap), "--range", byte_range.removeprefix("bytes="), "--dump-header", "-", url]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw = process.stdout.read(cap + 16385)
    if len(raw) > cap + 16384:
        process.terminate()  # This process was created here; never an external worker.
        process.communicate(timeout=5)
        raise ValueError("Curl response exceeded bounded metadata envelope")
    remainder, _ = process.communicate(timeout=25)
    if remainder or process.returncode:
        raise ValueError("Bounded official curl metadata request failed")
    header, separator, body = raw.partition(b"\r\n\r\n")
    if not separator:
        raise ValueError("Unexpected HTTP header layout")
    lines = header.decode("iso-8859-1").splitlines()
    status = int(lines[0].split()[1])
    headers = {key.title(): value.strip() for line in lines[1:] if ":" in line
               for key, value in [line.split(":", 1)]}
    if len(body) > cap or byte_range is None and body:
        raise ValueError("Unexpected HTTP metadata body")
    return status, headers, body


def range_directory_inventory(url: str, opener, known_size: int | None = None,
                              use_curl: bool = False) -> dict:
    transferred = 0

    def fetch(header: str, cap: int):
        nonlocal transferred
        request = urllib.request.Request(url, headers={"Range": header, "Accept-Encoding": "identity"})
        if use_curl:
            status, headers, data = curl_metadata_request(url, header, cap)
        else:
            with opener.open(request, timeout=20) as response:
                status, headers = response.status, response.headers
                if status != 206 and int(headers.get("Content-Length", cap + 1)) > cap:
                    raise ValueError("Server ignored bounded range for a large/unknown archive")
                data = response.read(cap + 1)
        bounds = headers.get("Content-Range", "")
        if status != 206:
            size = int(headers.get("Content-Length", cap + 1))
            if status != 200 or size > cap:
                raise ValueError("Server ignored bounded range for a large/unknown archive")
            bounds = f"bytes 0-{size - 1}/{size}"
        if len(data) > cap:
            raise ValueError("Range exceeded memory budget")
        transferred += len(data)
        return data, bounds

    try:
        if use_curl:
            _, headers, _ = curl_metadata_request(url, None, 65536)
            archive_size = int(headers["Content-Length"])
            tail, bounds = fetch(f"bytes={max(0, archive_size - 65536)}-{archive_size - 1}", 65536)
        else:
            tail, bounds = fetch(f"bytes=-{min(65536, known_size or 65536)}", 65536)
    except urllib.error.HTTPError as exc:
        if exc.code != 416:
            raise
        with opener.open(urllib.request.Request(url, method="HEAD"), timeout=20) as response:
            size = int(response.headers["Content-Length"])
        if size > 65536:
            raise ValueError("Unexpected range rejection for large archive") from exc
        tail, bounds = fetch(f"bytes=0-{size - 1}", 65536)
    archive_size = int(bounds.rsplit("/", 1)[1])
    if known_size is not None and archive_size != known_size:
        raise ValueError("Archive size differs from dependency lock")
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0 or len(tail) - eocd < 22:
        raise ValueError("ZIP end record not found")
    count, directory_bytes, directory_offset = struct.unpack_from("<HII", tail, eocd + 10)
    if count == 65535 or directory_bytes == 0xFFFFFFFF or directory_offset == 0xFFFFFFFF:
        raise ValueError("ZIP64 archive end record needs explicit support; no full-download fallback")
    if directory_bytes > 8 * 1024**2 or directory_offset + directory_bytes > archive_size:
        raise ValueError("Unexpected central directory bounds")
    start_of_tail = archive_size - len(tail)
    if directory_offset >= start_of_tail:
        directory = tail[directory_offset - start_of_tail:directory_offset - start_of_tail + directory_bytes]
    else:
        directory, _ = fetch(f"bytes={directory_offset}-{directory_offset + directory_bytes - 1}", directory_bytes)
    totals = directory_totals(directory)
    if totals["archive_entries"] != count:
        raise ValueError("ZIP entry count mismatch")
    return {"compressed_archive_bytes": archive_size, "range_bytes_received": transferred,
            "bounded_small_archive_body_received": archive_size <= 65536, **totals}


def main() -> None:
    started = time.perf_counter()
    lock_path = ROOT / "outputs/trackocd_core/audit/masa_runtime_resolution/pylock.toml"
    lock_payload = lock_path.read_bytes()
    lock = tomli.loads(lock_payload.decode())
    ranks = {t: i for i, t in enumerate(sys_tags())}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https": "http://127.0.0.1:17890"}))
    records = []
    for package in lock["packages"]:
        if package.get("marker") and not Marker(package["marker"]).evaluate():
            continue
        candidates = package.get("wheels") or [package["archive"]]
        wheel = select_wheel(candidates, ranks)
        record = {"name": package["name"], "version": package.get("version"),
                  "url": wheel["url"], "expected_sha256_from_lock": wheel["hashes"]["sha256"],
                  "wheel_payload_downloaded_or_hash_verified": False}
        try:
            record.update(range_directory_inventory(wheel["url"], opener, wheel.get("size"),
                          use_curl=urllib.parse.urlparse(wheel["url"]).hostname in
                          {"download.pytorch.org", "download-r2.pytorch.org"}))
            record["range_metadata_inspected"] = True
        except Exception as exc:
            record.update(range_metadata_inspected=False, error_type=type(exc).__name__)
        records.append(record)
        print(json.dumps({"package": record["name"], "range_metadata_inspected": record["range_metadata_inspected"],
                          "unpacked_4k_allocation_bytes": record.get("unpacked_4k_allocation_bytes")}), flush=True)
    complete = all(r["range_metadata_inspected"] for r in records)
    unpacked = sum(r.get("unpacked_4k_allocation_bytes", 0) for r in records)
    reserve = 256 * 1024**2 + (unpacked + 4) // 5
    plan = json.loads((ROOT / "configs/trackocd_core/masa_sam_candidate_smoke.json").read_text())
    ceiling = plan["execution_limits"]["max_new_environment_source_weights_and_output_bytes"]
    conservative = unpacked + reserve + plan["conditional_weight_ceiling_bytes"] + 16 * 1024**2
    result = {
        "schema_version": "trackocd.core.masa_wheel_inventory.v1",
        "inspected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dependency_lock_sha256": hashlib.sha256(lock_payload).hexdigest(),
        "compatible_wheel_packages": len(records), "wheels": records,
        "all_range_metadata_inspected": complete,
        "installed_file_4k_allocation_bytes": unpacked,
        "reserve_for_pyc_and_environment_bytes": reserve,
        "optional_two_weight_ceiling_bytes": plan["conditional_weight_ceiling_bytes"],
        "conservative_environment_source_weights_output_estimate_bytes": conservative,
        "candidate_stop_ceiling_bytes": ceiling,
        "fits_candidate_ceiling": complete and conservative <= ceiling,
        "status": "PASS_METADATA_BUDGET_NOT_INSTALL_OR_RUNTIME_PROOF" if complete and conservative <= ceiling else "BLOCKED_INCOMPLETE_OR_EXCESS_WHEEL_BUDGET",
        "disk_available_bytes": shutil.disk_usage(ROOT).free,
        "resources": {"workers": 1, "elapsed_seconds": time.perf_counter() - started,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "boundary": {"large_wheel_full_download": False, "archive_payload_saved": False,
                     "bounded_small_archives_may_be_read_in_memory": True, "archive_installed": False,
                     "wheel_payload_hash_verified": False, "base_environment_changed": False,
                     "model_or_data_execution": False, "test_access": False},
        "next": "Install only after complete hash-locked binary-wheel budget and actual usage monitoring; no compile, unpinned dependency or silent budget expansion"}
    destination = ROOT / "outputs/trackocd_core/audit/masa_runtime_wheel_inventory.json"
    atomic_json(destination, result)
    print(json.dumps({"output": str(destination), "status": result["status"],
                      "conservative_bytes": conservative, "ceiling_bytes": ceiling,
                      "resources": result["resources"]}, indent=2))


if __name__ == "__main__":
    main()
