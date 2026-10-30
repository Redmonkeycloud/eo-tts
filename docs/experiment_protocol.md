# Experiment protocol: Best-of-N for TerraDiT-Ω-XL

Version 1, written and committed before any Best-of-N sample was generated. The git history
of this file is the record of what was decided in advance. Any later change is listed under
[Deviations](#deviations) with its date and reason, and the original text stays visible in
the history.

## Questions

**Q1. Does Best-of-N selection improve TerraDiT-Ω-XL samples, and how does the gain grow with N?**
N ∈ {1, 2, 4, 8, 16}, compared with N = 1 (no selection) on the same tiles.

**Q2. Does the gain depend on the verifier, and does it carry over to metrics the verifier
never saw?** A gain that appears only on metrics close to the verifier is reward hacking,
not better images.

**Q3. Is the gain different on geographically held-out tiles (Git-Spatial)?** On Spatial the
base model is weaker (paper FID 17.54 vs 9.25 on Git-Rand). We test whether search recovers
more there.

## Design

Every comparison is **paired**: for each tile, all N values select from the same 16 samples,
which are generated once. BoN-N chooses among samples 0 … N−1, so BoN-1 is sample 0, a plain
draw from the model, and the sets are nested. The only thing that differs between N values
is how many candidates the verifier sees.

### Tiles

| Set | Split | Tiles | Role |
| --- | --- | --- | --- |
| R2000 | Git-Rand | 2,000 | Q1, Q2 |
| S500 | Git-Spatial | 500 | Q3 |

Each set is the first n entries of `numpy.random.default_rng(0).permutation(len(split))`
over the split's row order. The lists are committed with this protocol in
`configs/subsets/{random_2000,spatial_500}.json` (image name and Git-10M row index per tile).
The splits are sorted by location, so a prefix of the split would be biased. A smoke test on
the first 64 Spatial tiles gave SSIM 0.43 against 0.34 for the paper's full split.

**Representativeness**, computed before any generation
(`configs/subsets/*_representativeness.json`):

| Statistic | Git-Rand full | R2000 | Git-Spatial full | S500 |
| --- | --- | --- | --- | --- |
| Tiles | 14,932 | 2,000 | 14,387 | 500 |
| Instances per tile, mean | 10.84 | 11.00 | 2.36 | 2.27 |
| Instances per tile, median / 90th pct. | 3 / 41 | 3 / 41 | 1 / 2 | 1 / 2 |
| Tiles without instances | 20.0 % | 19.4 % | 28.6 % | 30.0 % |
| Tiles with ≥ 15 instances | 22.9 % | 23.3 % | 3.0 % | 3.0 % |
| Latitude, mean ± sd (°) | 32.0 ± 19.4 | 32.2 ± 19.7 | 29.9 ± 5.1 | 29.2 ± 5.2 |
| Longitude, mean ± sd (°) | 37.2 ± 71.8 | 35.2 ± 71.8 | 92.5 ± 1.3 | 92.6 ± 1.3 |
| Top-20 OSM tags shared with the full split | – | 18 of 20 | – | 17 of 20 |
| Total variation distance, top-20 tag shares | – | 0.046 | – | 0.038 |

Sizes are set by compute (see [Budget](#budget)) and are **fixed**: we will not enlarge a set
because a result is or is not significant. A larger set would be a new, separately
reported experiment.

### Generation

| Setting | Value |
| --- | --- |
| Model | `omega_xl`, released weights, TerraDiT code at `8d90f0d` |
| Conditioning | text + geolocation + all primitives (`condition_type = omega`) |
| Sampler | upstream `euler_sampler`, 100 steps, CFG 0 (as in the paper) |
| Samples per tile | 16 |
| Precision | bf16 autocast (see [Anchors](#anchors)) |
| Batch size | 32 |
| Initial noise | one generator per sample, seeded with the first 8 bytes of `sha256("{split}:{hf_idx}:{k}")` for sample k |

Per-sample seeds make every sample reproducible on its own, independent of batch size,
batch composition and run order. Generation is resumable per sample.

### Verifiers (used to select)

| Id | Model | Score |
| --- | --- | --- |
| V1 | RemoteCLIP ViT-B/32 (`chendelong/RemoteCLIP`, file `RemoteCLIP-ViT-B-32.pt`, revision `bf1d8a3c`) | cosine(image, caption) |
| V2 | GeoRSCLIP ViT-L/14 (`Zilun/GeoRSCLIP`, file `ckpt/RS5M_ViT-L-14.pt`, revision `4920188e`) | cosine(image, caption) |
| V1+V2 | rank average | mean of the sample's rank under V1 and under V2 among the N candidates; ties broken by V2 |

Captions longer than the text encoder's 77-token context are truncated, the same way for
both verifiers. A structural verifier for the geometry primitives (V3) is not part of
version 1. It will be added by amending this protocol **before** it is run.

### Evaluators (never used to select)

| Id | Metric | Kind | Relation to the verifiers |
| --- | --- | --- | --- |
| E1 | CLIP score, OpenAI ViT-L/14, cosine (the paper's metric, upstream code) | text alignment | **Related.** V1 and V2 are continued training of general-purpose CLIP models, and V2 has the same architecture as E1. |
| E2 | LPIPS to the tile's ground truth (upstream code) | conditioning fidelity | independent |
| E3 | SSIM to the tile's ground truth (upstream code) | conditioning fidelity | independent |
| E4 | KID, Inception-v3 features | distribution realism | independent |
| E5 | KID, DINOv2 ViT-L/14 features (`facebook/dinov2-large`, revision `47b73eef`) | distribution realism, different feature space | independent |

FID is **not** reported on these sets: at 2,000 or 500 images it is strongly biased and not
comparable to anything. KID is unbiased at any size. E5 follows the finding that Inception
features under-represent the qualities modern generators differ in (Stein et al., NeurIPS
2023). It guards against a gain that exists only in Inception space.

Because E1 is related to the verifiers, we report the Spearman correlation between each
verifier and each evaluator over all 16 × 2,000 R2000 samples. A gain in E1 alone is not
evidence of better images.

### Endpoints

Decided now, so that no metric is singled out after the fact.

- **Primary, Q1/Q2:** change in E1 (CLIP score) from N = 1 to N = 16 on R2000, for V1, V2 and
  V1+V2.
- **Primary guard:** E5 (KID-DINOv2) at N = 16 is non-inferior to N = 1: the upper bound of
  the 95 % CI of KID(N = 16) − KID(N = 1) must be below 10 % of KID(N = 1). A text-alignment
  gain bought with less realistic images counts as a failure of the method, not a success.
- **Primary, Q3:** the same E1 change on S500, compared with R2000.
- **Secondary:** all evaluators at every N, for every verifier, on both sets; the scaling curve
  over N; the oracle for E1–E3 (the best of the N samples by that evaluator) as an upper
  bound on what selection can reach.

### Statistics

- Per-image metrics (E1–E3): mean paired difference against N = 1, 95 % percentile bootstrap
  confidence interval over tiles, 10,000 resamples.
- KID (E4, E5): difference against N = 1, bootstrap over tiles, 1,000 resamples. Features
  are computed once per image and reused.
- Multiple comparisons: Holm correction within the primary family (3 verifiers × 2 sets).
  Secondary results carry uncorrected intervals and are labelled as such.
- All configurations are reported, including null and negative results.

### Anchors

These check that a subset and bf16 do not change what is measured.

1. **R2000 vs the full baseline.** The fp32 baseline images
   ([results/baseline/omega_xl_random_omega.json](../results/baseline/omega_xl_random_omega.json))
   cover every R2000 tile. On those tiles we report E1–E3 means with CIs, against the
   full-split means. This checks that the subset is representative in outcome, not only in
   metadata.
2. **bf16 vs fp32 on R2000.** E1–E3 and E4 of BoN-1 (bf16, per-sample seeds) against the fp32
   baseline images on the same tiles. The noise differs, so this is a distributional
   comparison.
3. **bf16 vs fp32 on S500, same noise.** Sample 0 of every S500 tile is also generated in fp32
   with the identical seed. We report per-image PSNR between the two and E1–E4 for both. The
   benchmark already measured 54.2 dB mean PSNR on 8 tiles.
4. **S500 vs the paper.** E1–E3 of the fp32 sample 0 on S500 next to the paper's Git-Spatial
   values, with CIs. This is a plausibility check, **not** a reproduction claim: the full
   Git-Spatial baseline is not run.

If an anchor shows a difference larger than the CI of the corresponding comparison, we stop
and report it before running anything further.

### Final full-split run

After the R2000 results, the best verifier by the primary endpoint is run on the full
Git-Rand split at N = 4 (about 105 GPU-hours), to give a FID directly comparable to our
baseline (9.81) and the paper (9.25). The verifier is chosen on R2000, which is part of the
full split. This selection effect is small (2,000 of 14,932 tiles) and is stated with the
result.

## Budget

On one RTX 3060 (measured: bf16 6.33 s, fp32 11.6 s per image at batch 32).

| Run | Images | GPU-hours |
| --- | --- | --- |
| R2000, 16 samples, bf16 | 32,000 | 56 |
| S500, 16 samples, bf16 | 8,000 | 14 |
| S500 sample 0, fp32 anchor | 500 | 1.6 |
| Verifier and evaluator scoring | 40,500 | about 3 (estimate) |
| **Subtotal, Q1–Q3** | | **≈ 75** |
| Final run, full Git-Rand, N = 4, bf16 | 59,728 | 105 |

## Limitations known in advance

- One base model. Results may not hold for other EO generators.
- Two text-alignment verifiers. Neither scores the geometry conditioning (V3 is deferred).
- The verifiers' training data (RS5M and RemoteCLIP's sources) may overlap with Git-10M imagery
  or captions. We have not checked this.
- S500 is one held-out region of narrow longitude (sd 1.3°), not "unseen geography" in general.
- The Git-Spatial full-split baseline is not run. The Spatial comparison is paired and
  internal (anchors 3 and 4).

## Deviations

None yet.
