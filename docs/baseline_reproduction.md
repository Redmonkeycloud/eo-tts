# Baseline reproduction: TerraDiT-Ω-XL on Git-Rand-15k

We re-ran the evaluation of TerraDiT-Ω-XL ([arXiv:2606.31029](https://arxiv.org/abs/2606.31029))
with the authors' released code and weights, on the full random test split, and obtained
FID 9.81 against the paper's 9.25. LPIPS, SSIM and CLIP score agree with the paper to
within 0.007. All Best-of-N and search results in this repository are compared against
this re-run, not against the paper's numbers.

Raw result file: [`results/baseline/omega_xl_random_omega.json`](../results/baseline/omega_xl_random_omega.json)

## Results

| Metric | Ours | Paper | Ours − paper | Better is |
| --- | --- | --- | --- | --- |
| FID | 9.81 | 9.25 | +0.56 (+6.1 %) | lower |
| LPIPS | 0.3401 | 0.3438 | −0.0037 | lower |
| SSIM | 0.2669 | 0.2665 | +0.0004 | higher |
| CLIP score (cosine) | 0.2882 | 0.2812 | +0.0070 | higher |
| KID (×10³) | 3.98 ± 0.41 | not reported | – | lower |
| sFID | not computed | 4.38 | – | lower |

Paper numbers: Table 1 (FID, sFID, LPIPS) and Table S3 (SSIM, CLIP), row TerraDiT-Ω-XL,
condition T+Ω+L, column Git-Rand.

## What was run

| Setting | Value | Source of the value |
| --- | --- | --- |
| Model | `omega_xl` (SiT-XL/2, 1.18 B parameters) | [MVRL/TerraDiT](https://huggingface.co/MVRL/TerraDiT), released EMA weights |
| Conditioning | text + geolocation (RANGE+) + all geometry primitives (Ω) | `--condition-type omega` |
| Test split | Git-Rand, 14,932 tiles | [MVRL/TerraDiT-data](https://huggingface.co/datasets/MVRL/TerraDiT-data) `splits/random` |
| Sampler | Euler, linear interpolant, 100 steps | paper §Sampling; repo default |
| Classifier-free guidance | none (scale 0.0) | paper; repo default |
| Batch size | 32 | paper (the repo defaults to 16) |
| Seed | 42 | repo default; the paper does not report one |
| Precision | fp32 | repo default |
| Code | [mvrl/TerraDiT](https://github.com/mvrl/TerraDiT) at `8d90f0d`, unmodified | git submodule |
| Hardware | 1× NVIDIA RTX 3060 12 GB, Windows 11 | |
| Software | Python 3.14.0, PyTorch 2.9.1 (CUDA 12.6), torchmetrics 1.9.0, transformers 5.21.0, diffusers 0.42.1 | recorded in the result file |
| Wall-clock | 48.3 h generation (11.6 s per image), 57 min metrics | |

The only change to the upstream pipeline is where images and captions are read from.
Instead of downloading all of Git-10M (394 GB), we kept the 14,932 rows the split refers to,
from the same pinned dataset revision (`29f192b8`), and checked every row's file name
against the split metadata before use (see `eotts/data/git10m_subset.py`). Pixels and
captions are therefore byte-identical to what the upstream code would have read.

## How each metric is computed

All metrics come from the upstream `terradit/eval/metrics.py`, except KID, which we added.
The reference set for every metric is the ground-truth tile of the same split row.

**FID (Fréchet Inception Distance).** Distance between Gaussians fitted to Inception-v3
pool features (2,048-d) of the 14,932 generated tiles and the 14,932 ground-truth tiles,
using `torchmetrics.image.fid.FrechetInceptionDistance`. Generated tiles are resized to the
ground-truth size if they differ (both are 256×256 here, so no resizing happens). Measures
how close the distribution of generated images is to the real one; it does not compare
images pairwise. FID is biased upward at small sample sizes, so it is only comparable to
the paper on the full split.

**LPIPS (Learned Perceptual Image Patch Similarity).** Perceptual distance between each
generated tile and its own ground-truth tile, SqueezeNet backbone, inputs scaled to [−1, 1],
averaged over tiles (`torchmetrics`). Measures how close each sample is to the one real image
that its conditioning came from. Because the model is stochastic and many images fit one
caption and layout, this is a measure of conditioning fidelity, not of image quality, and it
has a floor well above 0.

**SSIM (Structural Similarity).** Pairwise structural similarity between each generated
tile and its ground truth, Gaussian window, data range 1, averaged over tiles
(`torchmetrics`). Same caveat as LPIPS: it rewards matching the specific real image, not
realism.

**CLIP score.** Cosine similarity between the CLIP ViT-L/14 image embedding of each
generated tile and the text embedding of its caption (`openai/clip-vit-large-patch14`,
Hugging Face `transformers`), averaged over tiles. The upstream code reports the CLIP logit,
which is the cosine multiplied by CLIP's learned temperature (exactly 100); the result file
stores both, as `clip_score` (logit) and `clip_cos` (cosine). The paper reports the cosine.

**KID (Kernel Inception Distance)**, added by us. Squared maximum mean discrepancy with a
polynomial kernel on the same Inception-v3 features as FID, averaged over 100 random
subsets of 1,000 tiles (`torchmetrics.image.kid`); the reported ± is the standard deviation
over subsets. KID is an unbiased estimator, so unlike FID it can be compared across sample
sizes. We use it for the Best-of-N experiments, which run on a 500-tile subset where FID
would not be meaningful.

## What the paper reports that we do not

- **sFID** (spatial FID on Inception `Mixed_6e` features): described in the paper but not
  implemented in the released code.
- **Classification Accuracy Score (CAS)**: needs a classifier trained on generated crops;
  out of scope here.
- **Git-Spatial and Git-Dense splits**: not yet re-run.
- **Box-only and point-only conditioning**: not yet re-run.

## What the paper does not report

These are the details we had to choose ourselves. Each could move FID by some amount.

- **Random seed** for the initial noise.
- **Software versions** (PyTorch, torchmetrics, the Inception weights' source, PIL).
- **Inference hardware and numerical precision.** Training used 4× H100; the GPU and
  precision used to generate the evaluation samples are not stated.
- **Which weights were evaluated.** The paper does not say whether the reported numbers come
  from the fp32 training checkpoint or from the released fp16 export.
- **CLIP-score details**: how captions longer than CLIP's 77-token limit were handled, and
  the exact implementation.
- **Number of evaluation runs**: whether the numbers are from one run or an average.

## Why FID differs by 0.56

We cannot isolate the cause without more runs, but these are the plausible contributors,
most likely first.

1. **Released weights are fp16.** The model card states that the published checkpoint holds
   EMA weights cast to fp16, with the REPA heads removed. If the paper evaluated the original
   fp32 training checkpoint, the cast perturbs every weight slightly. Over 100 sampling
   steps that changes individual samples, and FID is sensitive to small, systematic shifts in
   the feature distribution. This is the only known difference in the model itself.
2. **Different initial noise.** The paper's seed is unknown and ours is 42. Noise is also drawn
   per batch, so the same seed gives different samples at different batch sizes. Seed-to-seed
   variation of FID at about 15,000 samples is usually a few tenths of a point, so this alone
   would likely explain part of the gap, but probably not all of it.
3. **Software and numerical differences.** Different PyTorch and torchmetrics versions,
   different CUDA kernels on a different GPU (RTX 3060 vs a data-centre GPU), and possible
   TF32 or mixed precision on their side change floating-point results. In a 100-step
   sampler, these differences compound into different samples.
4. **Image decoding and storage.** Generated tiles are written as 8-bit PNG and real tiles are
   decoded from the dataset's JPEGs. A different decoder or library version in their pipeline
   would shift the Inception features of the real set slightly.

The other three metrics argue against a systematic error in our pipeline. LPIPS and SSIM are
per-image comparisons against the same ground truth and match the paper to the third
decimal. A wrong conditioning input, split or checkpoint would move them clearly. A gap in
FID alone, with the per-image metrics intact, is what one expects from sampling and
numerical differences.

## Why CLIP score differs by 0.007

Our CLIP cosine is slightly higher. The likely reason is implementation detail, not image
quality: the treatment of long captions (Git-10M captions often exceed 77 tokens and are
truncated here), image preprocessing (resize and centre crop by the CLIP processor), or a
different CLIP implementation in the paper's script. LPIPS is likewise 0.004 better here,
which is within what different samples of the same model produce.

## Provenance note

The result file records the `eo-tts` commit as `e458719`. Provenance is written when the
run finishes, and two commits that change only `README.md` were added while it was running.
The code that generated and scored the images is commit `5e434c6`; no file under `eotts/`,
`scripts/` or `configs/` changed between the two. Later runs record the commit at start.

## Reproducing

```powershell
python scripts/01_download_assets.py
python scripts/02_build_git10m_subset.py --split random
python scripts/04_run_baseline.py --split random --batch-size 32
```
