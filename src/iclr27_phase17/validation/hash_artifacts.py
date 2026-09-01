"""Write a deterministic SHA-256 manifest for Phase 17 artifacts."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

def main() -> None:
    roots = [ROOT / "docs/iclr27_phase17", ROOT / "src/iclr27_phase17", ROOT / "configs/iclr27_phase17", ROOT / "outputs/iclr27_phase17", ROOT / "data/iclr27_phase17"]
    items = {}
    for base in roots:
        if not base.exists(): continue
        for p in sorted(base.rglob("*")):
            if p.is_symlink() or not p.is_file() or p.name.endswith(".tmp") or p.name == "artifact_hashes.json": continue
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
            items[str(p.relative_to(ROOT))] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
    out = ROOT / "outputs/iclr27_phase17/manifests/artifact_hashes.json"; out.parent.mkdir(parents=True, exist_ok=True)
    value = {"protocol": "trackocd_iclr27_phase17_artifact_hashes", "files": items, "symlinks_excluded": True}
    tmp = out.with_suffix(out.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True)); os.replace(tmp, out)
    print(json.dumps({"files": len(items), "out": str(out)}, indent=2))

if __name__ == "__main__": main()
