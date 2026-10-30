"""Check that this machine can run the pipeline: GPU, packages, pinned TerraDiT, disk, Hub login.

    python scripts/00_check_env.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'ok' if cond else 'FAIL'}] {name}{': ' + detail if detail else ''}")


try:
    import torch
    check("torch", True, torch.__version__)
    check("CUDA available", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "install the CUDA build of torch")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        check("GPU memory >= 10 GB", p.total_memory / 2**30 >= 10, f"{p.total_memory / 2**30:.1f} GB")
        check("bf16 supported", torch.cuda.is_bf16_supported())
except ImportError:
    check("torch", False, "not installed")

for mod in ("diffusers", "transformers", "torchmetrics", "torch_fidelity", "pyarrow", "huggingface_hub",
            "safetensors", "timm", "einops", "scipy", "open_clip"):
    try:
        m = __import__(mod)
        check(mod, True, getattr(m, "__version__", ""))
    except ImportError:
        check(mod, False, "not installed")

from eotts.provenance import TERRADIT_ROOT, git_state, REPO_ROOT
st = git_state(TERRADIT_ROOT)
check("TerraDiT submodule", st["commit"] is not None,
      f"{st['commit']} dirty={st['dirty']}" if st["commit"] else "run: git submodule update --init")
check("TerraDiT at pinned commit", st["commit"] == "8d90f0ddb8994f87794cbd431f0ebb531cd0dd64")
try:
    from eotts.provenance import ensure_terradit_importable
    ensure_terradit_importable()
    import terradit.generate  # noqa: F401
    check("import terradit.generate", True)
except Exception as e:
    check("import terradit.generate", False, f"{type(e).__name__}: {e}")

free = shutil.disk_usage(REPO_ROOT).free / 2**30
check("free disk >= 50 GB", free >= 50, f"{free:.0f} GB free at {REPO_ROOT}")

try:
    from huggingface_hub import whoami
    check("Hugging Face login", True, whoami()["name"])
except Exception:
    print("[warn] not logged in to Hugging Face: downloads work but are rate-limited; "
          "run `huggingface-cli login` before building the random subset")

print("\nall good" if ok else "\nfix the FAIL lines above")
sys.exit(0 if ok else 1)
