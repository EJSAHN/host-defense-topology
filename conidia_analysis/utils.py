from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pkg_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not_installed"


def write_manifest(path: Path, inputs: Dict[str, Path], outputs: Dict[str, Path]) -> None:
    data: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "statsmodels": _pkg_version("statsmodels"),
            "openpyxl": _pkg_version("openpyxl"),
        },
        "inputs": {},
        "outputs": {},
    }

    for name, p in inputs.items():
        data["inputs"][name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    for name, p in outputs.items():
        if not p.exists():
            continue
        data["outputs"][name] = {
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
