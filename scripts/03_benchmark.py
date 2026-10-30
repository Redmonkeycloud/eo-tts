"""Measure TerraDiT sampling speed and memory on this GPU, fp32 vs bf16 autocast.

    python scripts/03_benchmark.py                 # omega_xl, spatial split tiles
    python scripts/03_benchmark.py --batch-sizes 4 8 16 --fidelity-batch 8

Two parts:
1. Throughput: a few Euler steps per batch size, in fp32 (the repo's default) and in bf16
   autocast; seconds per image for the full 100-step protocol are extrapolated.
2. Fidelity: the same tiles and the same noise sampled for the full 100 steps in fp32 and
   in bf16; reports how far apart the decoded images are and saves them side by side.

Needs: scripts/01_download_assets.py and scripts/02_build_git10m_subset.py --split spatial.
Writes runs/benchmark/benchmark.json and runs/benchmark/fp32_vs_bf16.png.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eotts.provenance import ensure_terradit_importable, provenance
from eotts.data.git10m_subset import Git10MSubset

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="omega_xl")
ap.add_argument("--split", default="spatial")
ap.add_argument("--data-root", default="data/terradit")
ap.add_argument("--subset-root", default="data/git10m_subset")
ap.add_argument("--weights-root", default="data/weights")
ap.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 16, 32])
ap.add_argument("--timing-steps", type=int, default=5)
ap.add_argument("--fidelity-batch", type=int, default=8)
ap.add_argument("--num-steps", type=int, default=100)
ap.add_argument("--out-dir", default="runs/benchmark")
args = ap.parse_args()

ensure_terradit_importable()
import torch
from torch.utils.data._utils.collate import default_collate
from terradit.generation import build_inference_model, load_text_vae, decode_latents
from terradit.data.dataset import build_dataset
from terradit.sampling.samplers import euler_sampler
from terradit.training.train_terradit import build_model_kwargs
from terradit.hf import resolve_checkpoint
from terradit.generate import split_paths

assert torch.cuda.is_available(), "benchmark needs a CUDA GPU"
dev = "cuda"
FAMILY = {"alpha_xl": "alpha", "sigma_xl": "sigma", "omega_xl": "omega", "omega_base": "omega"}[args.model]
os.makedirs(args.out_dir, exist_ok=True)

ckpt = os.path.dirname(resolve_checkpoint(args.model, checkpoints_root=args.weights_root))
model = build_inference_model(FAMILY, None, ckpt, dev)
_, clip, vae = load_text_vae(dev)
weights_gb = torch.cuda.memory_allocated() / 2**30

sp = split_paths(args.data_root, args.split)
subset = Git10MSubset(os.path.join(args.subset_root, args.split))
extra = dict(point_source="instances") if FAMILY == "sigma" else {}
ds = build_dataset(FAMILY, args.data_root, hf_dataset=subset, split="test", vae_on_the_fly=True,
                   metadata_path=sp["metadata_path"], inst_meta_path=sp["inst_meta_path"],
                   loc_embed_path=sp["loc_embed_path"], **extra)


def conditioning(b):
    samples = [ds[i] for i in range(b)]
    batch = default_collate([[x for x in s if torch.is_tensor(x)] for s in samples])
    _, _, kw = build_model_kwargs(FAMILY, batch, clip, dev, 0, 1)
    if FAMILY == "omega":
        kw["dropout_probs"] = None      # condition_type omega: all primitives
    return kw


def sample(kw, xT, steps, bf16):
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
    with torch.no_grad(), ctx:
        return euler_sampler(model, xT, num_steps=steps, cfg_scale=0.0, path_type="linear", **kw).float()


def timed(b, bf16):
    kw = conditioning(b)
    g = torch.Generator(device=dev).manual_seed(0)
    xT = torch.randn(b, model.in_channels, 32, 32, device=dev, generator=g)
    sample(kw, xT, 1, bf16)                              # warm-up
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    sample(kw, xT, args.timing_steps, bf16)
    torch.cuda.synchronize()
    per_step = (time.time() - t0) / args.timing_steps
    return {"batch": b, "dtype": "bf16" if bf16 else "fp32", "s_per_step_batch": round(per_step, 4),
            "s_per_image_100steps": round(per_step * args.num_steps / b, 3),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


timing = []
for bf16 in (False, True):
    for b in args.batch_sizes:
        try:
            r = timed(b, bf16)
            timing.append(r)
            print(f"[time] {r}")
        except torch.cuda.OutOfMemoryError:
            print(f"[time] batch {b} {'bf16' if bf16 else 'fp32'}: out of memory")
            torch.cuda.empty_cache()
            break


def best(dtype):
    rs = [r for r in timing if r["dtype"] == dtype]
    return min(rs, key=lambda r: r["s_per_image_100steps"]) if rs else None


# fidelity: same tiles, same noise, full protocol, fp32 vs bf16
b = args.fidelity_batch
kw = conditioning(b)
g = torch.Generator(device=dev).manual_seed(1234)
xT = torch.randn(b, model.in_channels, 32, 32, device=dev, generator=g)
img32 = decode_latents(vae, sample(kw, xT, args.num_steps, False), dev).float().clamp(0, 1)
img16 = decode_latents(vae, sample(kw, xT, args.num_steps, True), dev).float().clamp(0, 1)
mse = ((img32 - img16) ** 2).flatten(1).mean(1)
psnr = (10 * torch.log10(1.0 / mse.clamp_min(1e-12))).cpu().tolist()
fidelity = {"tiles": b, "mean_abs_diff": float((img32 - img16).abs().mean()),
            "psnr_db_per_tile": [round(x, 2) for x in psnr], "psnr_db_mean": round(sum(psnr) / b, 2)}
import torchvision
torchvision.utils.save_image(torch.cat([img32, img16]), os.path.join(args.out_dir, "fp32_vs_bf16.png"), nrow=b)
print(f"[fidelity] {fidelity}")

est = {}
for dtype in ("fp32", "bf16"):
    r = best(dtype)
    if r is None:
        continue
    s = r["s_per_image_100steps"]
    est[dtype] = {"best_batch": r["batch"], "s_per_image": s,
                  "full_random_split_h": round(14932 * s / 3600, 1),
                  "full_spatial_split_h": round(14387 * s / 3600, 1),
                  "bon16_h": {n: round(n * 16 * s / 3600, 1) for n in (250, 500, 1000, 2000)}}

out = {"model": args.model, "weights_gb_on_gpu": round(weights_gb, 2), "timing": timing,
       "fidelity_fp32_vs_bf16": fidelity, "estimates": est, "provenance": provenance()}
json.dump(out, open(os.path.join(args.out_dir, "benchmark.json"), "w", encoding="utf-8"), indent=2)
print(json.dumps(est, indent=2))
print(f"[benchmark] -> {args.out_dir}/benchmark.json, fp32_vs_bf16.png (top row fp32, bottom bf16)")
