"""Record what produced a result: code versions, environment, hardware."""
from __future__ import annotations

import os
import platform
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TERRADIT_ROOT = os.path.join(REPO_ROOT, "third_party", "TerraDiT")


def _git(args, cwd):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              check=True).stdout.strip()
    except Exception:
        return None


def git_state(path):
    commit = _git(["rev-parse", "HEAD"], path)
    dirty = _git(["status", "--porcelain", "--untracked-files=no"], path)
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def environment():
    import torch
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if torch.cuda.is_available() else None,
    }
    for mod in ("diffusers", "transformers", "torchmetrics", "huggingface_hub", "pyarrow"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env


def provenance():
    return {"eo_tts": git_state(REPO_ROOT), "terradit": git_state(TERRADIT_ROOT),
            "env": environment()}


def ensure_terradit_importable():
    """Make ``import terradit`` resolve to the pinned submodule."""
    if TERRADIT_ROOT not in sys.path:
        sys.path.insert(0, TERRADIT_ROOT)
    import terradit  # noqa: F401
    if not os.path.abspath(terradit.__path__[0]).startswith(os.path.abspath(TERRADIT_ROOT)):
        print(f"[warn] terradit imported from {terradit.__path__[0]}, not the pinned submodule")
