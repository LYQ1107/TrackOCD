"""Safe checkpoint structure and bounded wheel metadata checks; no inference."""
import io
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.tags import Tag

from scripts.trackocd_core.inspect_masa_checkpoint import pickle_global_names, prefix_summary
from scripts.trackocd_core.inventory_masa_wheels import (
    curl_metadata_request, directory_totals, range_directory_inventory, select_wheel,
)

ROOT = Path(__file__).resolve().parents[2]


def test_tensor_prefix_inventory_is_not_a_runtime_model_certificate():
    tensor = SimpleNamespace(numel=lambda: 12, element_size=lambda: 4)
    result = prefix_summary({"detector.backbone.layer.weight": tensor, "rpn_head.weight": tensor})
    assert result["detector.backbone"] == {"tensors": 1, "elements": 12, "tensor_bytes": 48}
    assert result["rpn_head"]["tensor_bytes"] == 48


def test_non_tensor_state_value_is_rejected():
    with pytest.raises(ValueError):
        prefix_summary({"rpn_head.weight": "not a tensor"})


def test_pickle_opcode_inventory_does_not_execute_global(tmp_path):
    archive = tmp_path / "metadata.pth"
    with zipfile.ZipFile(archive, "w") as writer:
        writer.writestr("archive/data.pkl", b"cbuiltins\neval\n.")
    assert pickle_global_names(archive) == ["builtins.eval"]


def test_checkpoint_receipt_is_exact_safe_and_never_frontend_qualification():
    receipt = json.loads((ROOT / "outputs/trackocd_core/audit/masa_checkpoint_recovery.json").read_text())
    assert receipt["size_and_sha256_match"]
    assert receipt["bytes"] == 558882875
    assert receipt["sha256"] == "441c05bf9519632fead1afd5200bb6a4f13b4a41c58f4428024490b4c2bd777c"
    assert receipt["safe_load"]["weights_only"] and receipt["safe_load"]["mmap"]
    assert not receipt["safe_load"]["extra_globals_allowlisted"]
    assert not receipt["safe_load"]["unsafe_fallback"]
    assert receipt["tensor_structure"]["total_tensors"] == 419
    assert receipt["tensor_structure"]["all_required_component_prefixes_present"]
    shapes = receipt["tensor_structure"]["selected_tensor_shapes"]
    assert shapes["roi_head.bbox_head.fc_cls.weight"] == [2, 1024]
    assert shapes["roi_head.bbox_head.fc_reg.weight"] == [4, 1024]
    assert not receipt["exact_runtime_model_key_and_shape_match_verified"]
    assert receipt["released_training_metadata"]["metadata_key_count"] == 0
    assert not receipt["released_training_metadata"]["saved_training_config_present"]
    assert not receipt["released_training_metadata"]["complete_release_stage_training_binding_verified"]
    assert not any(receipt["boundary"].values())


def test_complete_wheel_inventory_is_budget_metadata_not_install_proof():
    root = ROOT / "outputs/trackocd_core/audit"
    receipt = json.loads((root / "masa_runtime_wheel_inventory.json").read_text())
    lock = (root / "masa_runtime_resolution/pylock.toml").read_bytes()
    assert receipt["dependency_lock_sha256"] == hashlib.sha256(lock).hexdigest()
    assert receipt["compatible_wheel_packages"] == 48
    assert receipt["all_range_metadata_inspected"]
    assert all(w["range_metadata_inspected"] for w in receipt["wheels"])
    assert all(not w["wheel_payload_downloaded_or_hash_verified"] for w in receipt["wheels"])
    assert sum(w["unpacked_4k_allocation_bytes"] for w in receipt["wheels"]) == 5378584576
    assert receipt["conservative_environment_source_weights_output_estimate_bytes"] == 7673439806
    assert receipt["fits_candidate_ceiling"]
    assert not receipt["boundary"]["archive_installed"]
    assert not receipt["boundary"]["large_wheel_full_download"]
    assert not receipt["boundary"]["archive_payload_saved"]


def small_archive():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as writer:
        writer.writestr("one.py", "x" * 16)
        writer.writestr("two.bin", b"y" * 4097)
    return output.getvalue()


def test_central_directory_cost_includes_filesystem_allocation():
    payload = small_archive()
    end = payload.rfind(b"PK\x05\x06")
    length, offset = struct.unpack_from("<II", payload, end + 12)
    assert directory_totals(payload[offset:offset + length]) == {
        "archive_entries": 2, "unpacked_logical_bytes": 4113,
        "unpacked_4k_allocation_bytes": 12288}


def test_truncated_directory_is_rejected():
    with pytest.raises(ValueError):
        directory_totals(b"PK\x01\x02")


def test_small_archive_range_does_not_ask_beyond_lock_size():
    payload = small_archive()
    class Response:
        status = 206
        headers = {"Content-Range": f"bytes 0-{len(payload)-1}/{len(payload)}"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, cap): return payload[:cap]
    requests = []
    class Opener:
        def open(self, request, timeout):
            requests.append(request.get_header("Range"))
            return Response()
    result = range_directory_inventory("https://files.pythonhosted.org/a.whl", Opener(), len(payload))
    assert requests == [f"bytes=-{len(payload)}"]
    assert result["bounded_small_archive_body_received"]
    assert result["unpacked_logical_bytes"] == 4113


def test_ignored_large_range_aborts_before_body_read():
    class Response:
        status = 200
        headers = {"Content-Length": "999999999"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, cap): raise AssertionError("Must not read large full body")
    class Opener:
        def open(self, request, timeout): return Response()
    with pytest.raises(ValueError, match="large/unknown"):
        range_directory_inventory("https://download-r2.pytorch.org/a.whl", Opener())


def test_wheel_selection_rejects_an_unapproved_host():
    ranks = {Tag("py3", "none", "any"): 0}
    with pytest.raises(ValueError, match="source"):
        select_wheel([{"url": "https://example.org/demo-1.0-py3-none-any.whl"}], ranks)


def test_wheel_selection_preserves_python_and_platform_compatibility():
    ranks = {Tag("cp310", "cp310", "linux_x86_64"): 0, Tag("py3", "none", "any"): 1}
    urls = ["https://download-r2.pytorch.org/demo-1.0-cp311-cp311-linux_x86_64.whl",
            "https://files.pythonhosted.org/demo-1.0-py3-none-any.whl"]
    result = select_wheel([{"url": u} for u in urls], ranks)
    assert result["url"] == urls[1]


def test_curl_head_does_not_apply_body_size_limit_to_archive_length(monkeypatch):
    calls = []
    class Process:
        returncode = 0
        stdout = io.BytesIO(b"HTTP/2 200\r\ncontent-length: 2325908864\r\n\r\n")
        def communicate(self, timeout): return b"", b""
    def launch(command, **kwargs):
        calls.append(command)
        return Process()
    monkeypatch.setattr("scripts.trackocd_core.inventory_masa_wheels.subprocess.Popen", launch)
    status, headers, body = curl_metadata_request("https://download-r2.pytorch.org/a.whl", None, 65536)
    assert status == 200 and headers["Content-Length"] == "2325908864" and not body
    assert "--head" in calls[0] and "--max-filesize" not in calls[0]


def test_curl_range_has_explicit_size_limit_and_no_shell(monkeypatch):
    calls = []
    class Process:
        returncode = 0
        stdout = io.BytesIO(b"HTTP/2 206\r\ncontent-range: bytes 0-2/3\r\n\r\nabc")
        def communicate(self, timeout): return b"", b""
    def launch(command, **kwargs):
        assert "shell" not in kwargs
        calls.append(command)
        return Process()
    monkeypatch.setattr("scripts.trackocd_core.inventory_masa_wheels.subprocess.Popen", launch)
    status, _, body = curl_metadata_request("https://download-r2.pytorch.org/a.whl", "bytes=0-2", 3)
    assert status == 206 and body == b"abc"
    assert calls[0][calls[0].index("--max-filesize") + 1] == "3"
    assert calls[0][calls[0].index("--range") + 1] == "0-2"
