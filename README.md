# EO-TTS: Earth Observation Test-Time Scaling

![TerraDiT-Ω-XL samples on Git-Spatial tiles: fp32 (top) vs bf16 autocast (bottom), same noise](docs/assets/fp32_vs_bf16.png)

*TerraDiT-Ω-XL, 100 Euler steps, same tiles and noise: fp32 (top) vs bf16 autocast (bottom). Mean PSNR 54.2 dB; bf16 is 1.7× faster on an RTX 3060 (6.3 vs 10.8 s per image).*

**EO-TTS** studies test-time scaling for Earth observation diffusion transformers: inference-only search
(Best-of-N, then particle-based search with Rollover Budget Forcing) over pretrained
satellite image generators, guided by remote-sensing verifiers.

## Base model

All experiments build on **TerraDiT-Ω-XL** by Wei, Sastry, Cher, Xing and Jacobs
(Multimodal Vision Research Lab, Washington University in St. Louis). Their code is pinned
here as a git submodule at commit `8d90f0d` and used unmodified.

| | Link |
| --- | --- |
| Paper (TerraDiT-Ω, ECCV 2026) | [arXiv:2606.31029](https://arxiv.org/abs/2606.31029) · [project page](https://brian-j-wei.github.io/terradit-omega/index.html) |
| Paper (TerraDiT-α/Σ) | [arXiv:2603.02172](https://arxiv.org/abs/2603.02172) · [project page](https://vishu26.github.io/terradit/index.html) |
| Code | [github.com/mvrl/TerraDiT](https://github.com/mvrl/TerraDiT) |
| Weights | [huggingface.co/MVRL/TerraDiT](https://huggingface.co/MVRL/TerraDiT) |
| Derived data and test splits | [huggingface.co/datasets/MVRL/TerraDiT-data](https://huggingface.co/datasets/MVRL/TerraDiT-data) |
| Imagery (Git-10M) | [huggingface.co/datasets/lcybuaa/Git-10M](https://huggingface.co/datasets/lcybuaa/Git-10M) |

If you use this repository, please cite their work:

```bibtex
@inproceedings{wei2026terraditomega,
  title     = {TerraDiT-{\Omega}: Unified Spatial Control for Satellite Image Synthesis with Any Geospatial Primitive},
  author    = {Wei, Brian and Sastry, Srikumar and Cher, Daniel and Xing, Eric and Jacobs, Nathan},
  booktitle = {European Conference on Computer Vision},
  year      = {2026}
}

@article{sastry2026terradit,
  title   = {TerraDiT: Point-Conditioned Diffusion Transformer for Satellite Image Synthesis},
  author  = {Sastry, Srikumar and Cher, Daniel and Wei, Brian and Dhakal, Aayush and Khanal, Subash and Gupta, Dev and Jacobs, Nathan},
  journal = {arXiv:2603.02172},
  year    = {2026}
}
```

## Status

| Stage | State |
| --- | --- |
| Paper-protocol baseline, TerraDiT-Ω-XL, Git-Rand | done: FID 9.81 vs 9.25 in the paper ([details](docs/baseline_reproduction.md)) |
| Experiment protocol for Best-of-N, fixed before any run | done ([protocol](docs/experiment_protocol.md)) |
| Best-of-N on 2,000 Git-Rand tiles, N ∈ {1, 2, 4, 8, 16}, 2 verifiers | next |
| Best-of-N on 500 Git-Spatial tiles (held-out region), with fp32 anchor | next |
| Full Git-Spatial baseline | not run; Spatial is used as a held-out test set for search ([why](docs/experiment_protocol.md#anchors)) |
| Best verifier on the full Git-Rand split, N = 4 | after the subset results |
| Rollover Budget Forcing with custom allocation rules | later |

## Baseline results

TerraDiT-Ω-XL re-run with the authors' code and weights on the full Git-Rand split
(14,932 tiles, 100 Euler steps, no guidance, batch 32, seed 42, fp32):

| Metric | Ours | Paper | Ours − paper |
| --- | --- | --- | --- |
| FID ↓ | 9.81 | 9.25 | +0.56 |
| LPIPS ↓ | 0.3401 | 0.3438 | −0.0037 |
| SSIM ↑ | 0.2669 | 0.2665 | +0.0004 |
| CLIP score (cosine) ↑ | 0.2882 | 0.2812 | +0.0070 |
| KID ×10³ ↓ | 3.98 ± 0.41 | not reported | – |

The per-image metrics match the paper; FID is 6 % higher. What each metric measures, what
the paper leaves unstated, and the likely reasons for the gap (fp16 release weights, seed,
software and hardware) are in [docs/baseline_reproduction.md](docs/baseline_reproduction.md).
Raw output: [results/baseline/omega_xl_random_omega.json](results/baseline/omega_xl_random_omega.json).
All later experiments are compared against this re-run.

## Reproducing the paper baseline

The baseline calls TerraDiT's own `generate.py` and `eval/metrics.py` in-process. The only
substitution is where imagery comes from: instead of downloading all of Git-10M (394 GB,
787 shards), `Git10MSubset` keeps exactly the rows a test split uses, checked against the
split metadata by `img_name`, so pixels and captions are byte-identical.

| Split | Tiles | Git-10M shards needed | Download |
| --- | --- | --- | --- |
| random | 14,932 | 151 of 787 | 29.3 GB |
| spatial | 14,387 | 2 of 787 | 0.3 GB |

Shards are fetched one at a time and deleted after extraction; the stored subset is ~0.2 GB per split.

Paper targets (TerraDiT-Ω-XL, condition Ω, from Table 1 and Table S3 of [arXiv:2606.31029](https://arxiv.org/abs/2606.31029)) are in
`configs/paper_targets.json`; every result file stores them next to our numbers.

| Split | FID ↓ | LPIPS ↓ | SSIM ↑ | CLIP (cos) ↑ |
| --- | --- | --- | --- | --- |
| random | 9.25 | 0.3438 | 0.2665 | 0.2812 |
| spatial | 17.54 | 0.3614 | 0.3421 | 0.2761 |

Differences between the paper's stated protocol and the released code, recorded in each result:

- **Batch size:** the paper says 32, the repo defaults to 16. Noise is drawn per batch, so
  this changes individual samples, not the distribution. Use `--batch-size 32` if it fits.
- **Seed:** not reported in the paper; we use the repo default, 42.
- **CLIP:** the repo's `metrics.py` reports the CLIP logit (cosine × 100); the paper reports
  cosine. Results carry both `clip_score` and `clip_cos`.
- **sFID:** in the paper, not computed by the repo; not reported here.
- **KID:** not in the paper; added because it is unbiased on small subsets (Best-of-N).

## Setup (Windows 11, NVIDIA GPU)

```powershell
git clone --recursive https://github.com/Redmonkeycloud/eo-tts.git
cd eo-tts
conda create -n eotts python=3.11 -y
conda activate eotts
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e "third_party/TerraDiT[eval]"
pip install -e ".[dev]"
huggingface-cli login            # optional, avoids Hub rate limits on the 29 GB download
python scripts/00_check_env.py
```

If you cloned without `--recursive`: `git submodule update --init`.

## Run order

```powershell
python scripts/01_download_assets.py                        # omega_xl weights + test split files (~2.5 GB)
python scripts/02_build_git10m_subset.py --split spatial    # ~0.3 GB, about a minute
python scripts/03_benchmark.py                              # speed, VRAM, fp32 vs bf16 (~10 min)
python scripts/04_run_baseline.py --split spatial --limit 64   # smoke test
python scripts/02_build_git10m_subset.py --split random     # ~29 GB download, resumable
python scripts/04_run_baseline.py --split random            # the paper number (long run, resumable)
python scripts/04_run_baseline.py --split spatial
```

Generation resumes from the images already on disk, so an interrupted run is restarted with
the same command. Results land in `runs/baseline/<run>/results.json`.

## Layout

```
configs/paper_targets.json      paper numbers we compare against
docs/baseline_reproduction.md   baseline results, metric definitions, gap analysis
docs/experiment_protocol.md     Best-of-N questions, tiles, verifiers, endpoints, statistics (pre-registered)
configs/subsets/                fixed evaluation tile lists and their representativeness statistics
eotts/data/subsets.py           how the subsets are drawn and checked
results/                        result files of finished runs (images stay local)
eotts/data/git10m_subset.py     exact local subset of Git-10M (shard math, verify, read)
eotts/baselines/                paper-protocol baseline runner
eotts/eval/kid.py               KID (extra metric for small subsets)
eotts/provenance.py             code versions, environment, GPU, written into every result
scripts/00..04                  the steps above, in order
third_party/TerraDiT            upstream code, pinned, unmodified
tests/                          subset builder tests (no GPU, no network)
```

## Licences

Code licence for this repo: to be decided. TerraDiT code is Apache-2.0; its weights are CC-BY-NC-4.0 and Git-10M imagery
is CC-BY-NC-ND-4.0, so this work is non-commercial research only. Git-10M data is never
redistributed by this repo; it is downloaded from the original repository.
