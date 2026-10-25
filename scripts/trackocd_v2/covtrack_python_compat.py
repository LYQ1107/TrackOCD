#!/usr/bin/env python3
"""Launch the recovered COVTrack code with its compatible package overlay.

The new server retained COVTrack's Python 3.10 site-packages and the matching
PyTorch 2.0.0/cu117 packages, but the original venv interpreter symlink points
to a removed environment.  Keep the overlay explicit and isolated: this
launcher never changes a venv and never exposes the donor environments' full
site-packages to unrelated commands.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _first_existing(values: list[Path]) -> Path:
    for value in values:
        if value.is_file():
            return value
    raise FileNotFoundError("no compatible COVTrack base interpreter: " + ", ".join(map(str, values)))


def main() -> None:
    base_python = _first_existing([
        Path(os.environ.get("TRACKOCD_COVTRACK_BASE_PYTHON", "")),
        Path("/data3/liuyeqiang/.local/share/uv/python/cpython-3.10.21-linux-x86_64-gnu/bin/python3.10"),
        Path("/data3/liuyeqiang/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10"),
    ])
    site_packages = [
        Path(os.environ.get(
            "TRACKOCD_COVTRACK_SITE_PACKAGES",
            "/data3/liuyeqiang/.venvs/covtrack-public/lib/python3.10/site-packages",
        )),
        Path(os.environ.get(
            "TRACKOCD_COVTRACK_TORCH_SITE_PACKAGES",
            "/data3/liuyeqiang/.venvs/cfp-fishmot-cu117/lib/python3.10/site-packages",
        )),
        Path(os.environ.get(
            "TRACKOCD_COVTRACK_SUPPORT_SITE_PACKAGES",
            "/data3/liuyeqiang/.venvs/trackocd-masa-smoke/lib/python3.10/site-packages",
        )),
        Path(os.environ.get(
            "TRACKOCD_COVTRACK_PANDAS_SITE_PACKAGES",
            "/data3/liuyeqiang/.venvs/trackocd-a100/lib/python3.10/site-packages",
        )),
    ]
    package_paths = [str(path) for path in site_packages if path.is_dir()]
    # Keep the source overlay selected by TRACKOCD_COVTRACK_ROOT ahead of the
    # recovered default package.  This matters for bounded source-contract
    # probes: a temporary, explicitly selected shadow tree must not silently
    # import the tracked vendor tree just because the compatibility launcher
    # prepended its default path.
    covtrack_root = Path(os.environ.get(
        "TRACKOCD_COVTRACK_ROOT",
        str(ROOT / "vendor/trackocd_external/covtrack_public"),
    ))
    package_paths.extend([
        str(covtrack_root.resolve()),
        str(ROOT.resolve()),
    ])
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    if inherited_pythonpath:
        package_paths.append(inherited_pythonpath)

    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(package_paths)
    torch_site = Path(os.environ.get(
        "TRACKOCD_COVTRACK_TORCH_SITE_PACKAGES",
        "/data3/liuyeqiang/.venvs/cfp-fishmot-cu117/lib/python3.10/site-packages",
    ))
    torch_env = torch_site.parents[2] if torch_site.is_dir() else Path(
        "/data3/liuyeqiang/.venvs/cfp-fishmot-cu117"
    )
    library_paths = [
        str(path / "lib")
        for path in (torch_env, Path("/data3/liuyeqiang/.venvs/covtrack-public"))
        if (path / "lib").is_dir()
    ]
    inherited_ld = environment.get("LD_LIBRARY_PATH", "")
    if inherited_ld:
        library_paths.append(inherited_ld)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    os.execvpe(str(base_python), [str(base_python), *sys.argv[1:]], environment)


if __name__ == "__main__":
    main()
