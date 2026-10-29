# EO-TTS: Earth Observation Test-Time Scaling

![TerraDiT-Ω-XL samples on Git-Spatial tiles: fp32 (top) vs bf16 autocast (bottom), same noise](docs/assets/fp32_vs_bf16.png)

*TerraDiT-Ω-XL, 100 Euler steps, same tiles and noise: fp32 (top) vs bf16 autocast (bottom). Mean PSNR 54.2 dB; bf16 is 1.7× faster on an RTX 3060 (6.3 vs 10.8 s per image).*

**EO-TTS** studies test-time scaling for Earth observation diffusion transformers: inference-only search
(Best-of-N, then particle-based search with Rollover Budget Forcing) over pretrained
satellite image generators, guided by remote-sensing verifiers.

Base model: **TerraDiT-Ω-XL** ([code](https://github.com/mvrl/TerraDiT),
[weights](https://huggingface.co/MVRL/TerraDiT), ECCV 2026), pinned as a git submodule at
commit `8d90f0d` and used unmodified.

## Status

| Stage | State |
| --- | --- |
| Paper-protocol baseline (TerraDiT-Ω-XL, random + spatial) | code ready |
| Best-of-N, N ∈ {1, 4, 8, 16}, 2–3 verifiers | next |
| Rollover Budget Forcing with custom allocation rules | later |

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

Paper targets (TerraDiT-Ω-XL, condition Ω, from Table 1 and Table S3) are in
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
