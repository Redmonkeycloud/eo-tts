# Experiment protocol, Study 2: Best-of-N for TerraDiT-Ω-B on all of Git-Rand

Version 2, written and committed on 2026-11-12, before any sample of this study was generated.
As for version 1, the git history of this file is the record of what was decided in advance.
Later changes go under [Deviations](#deviations) with date and reason.

Study 1 ([experiment_protocol.md](experiment_protocol.md), results in
[bon_results.md](bon_results.md)) ran Best-of-N with TerraDiT-Ω-XL on 2,000 Git-Rand tiles
(R2000) and 500 Git-Spatial tiles (S500). Its results stand as they are and are reported as the
**pilot study**. Study 2 is the main experiment. It changes three things, decided at the
supervision meeting of 2026-11-12:

1. **Base model:** TerraDiT-Ω-B instead of Ω-XL.
2. **Tiles:** the R2000 experiment on the **whole** Git-Rand test split (14,932 tiles), not a
   subset. Git-Spatial is not part of Study 2.
3. **FID in every experiment**, next to the other metrics, and as a primary endpoint.

Everything else is carried over from Study 1 unchanged: sampler, seeds, verifiers, evaluators,
selection rules, bootstrap, guard. Where this file says nothing, version 1 with its amendments
applies.

## Questions

**Q1. Does Best-of-N selection improve TerraDiT-Ω-B samples on Git-Rand, and how does the gain
grow with N?** N ∈ {1, 2, 4, 8, 16}, against N = 1 on the same tiles.

**Q2. Does the gain depend on the verifier and carry over to metrics the verifier never saw**,
in particular to FID, the paper's headline metric, computed exactly as in the paper on the full
split?

## Design

### Tiles

| Set | Split | Tiles | Order |
| --- | --- | --- | --- |
| RFULL | Git-Rand (test) | 14,932, all of them | `numpy.random.default_rng(0).permutation(14932)` |

The list is `configs/subsets/random_full.json`. It uses the same permutation as R2000, so its
first 2,000 tiles are exactly the R2000 tiles of Study 1, in the same order (checked when the
list was written). The order only matters for the R2000-prefix analysis below and for the
same-noise anchor; every endpoint uses all tiles. No tile is left out, so representativeness is
not a question.

### Generation

| Setting | Value |
| --- | --- |
| Model | `omega_base`: TerraDiT-Ω-B (SiT-B/2, depth 12, hidden size 768), released EMA weights, TerraDiT code at `8d90f0d` |
| Conditioning | text + geolocation + all primitives (`condition_type = omega`) |
| Sampler | upstream `euler_sampler`, 100 steps, CFG 0 |
| Samples per tile | 16, so 238,912 images |
| Precision | bf16 autocast |
| Batch size | 32 |
| Initial noise | as in Study 1: one CPU generator per sample, seeded with the first 8 bytes of `sha256("random:{hf_idx}:{k}")` |

The seeds depend only on the split, the Git-10M row and k, so the R2000 tiles get the same
initial noise in both studies. The two models still give different images.

N is not reduced and tiles are not dropped because of run time. The run is complete when all
238,912 samples exist.

### Verifiers and evaluators

Verifiers V1 (RemoteCLIP ViT-B/32), V2 (GeoRSCLIP ViT-L/14) and the V1+V2 rank average, and
evaluators E1–E5, exactly as in version 1 (models, revisions, inputs). One evaluator is added:

| Id | Metric | Kind | Relation to the verifiers |
| --- | --- | --- | --- |
| E6 | FID, Inception-v3 features (the same features as E4), on all 14,932 tiles | distribution realism | independent |

E6 uses the torchmetrics formula, covariances with the n − 1 normaliser, and the
torchmetrics Inception network that the upstream TerraDiT evaluation uses
(`eotts/stats/fid.py`). At N = 1 it is computed on one sample per tile of the full test split,
as the paper computes FID, so the N = 1 value is directly comparable with a paper-protocol
baseline. Version 1 excluded FID because it is strongly biased at 2,000 or 500 images. At
14,932 images per set, and with every comparison made at the same n on the same tiles, that
objection no longer applies.

### Endpoints

Decided now, so that no metric is singled out after the fact.

- **Primary (two metrics, three rules):** the change from N = 1 to N = 16 on RFULL of
  - E1, CLIP score (higher is better), and
  - E6, FID (lower is better),

  for V1, V2 and V1+V2. These six tests form one family, Holm-corrected.
- **Primary guard:** unchanged from version 1. E5 (KID-DINOv2) at N = 16 is non-inferior to
  N = 1: the upper bound of the 95 % CI of KID(N = 16) − KID(N = 1) is below 10 % of
  KID(N = 1), or undefined if KID(N = 1) ≤ 0 (amendment 2).
- **Reading the primary results.** For each rule we report three verdicts side by side: CLIP
  (significant gain or not), FID (significant decrease, significant increase, or no significant
  change) and the guard (pass, fail, undefined). A rule is called an **improvement** only if the
  CLIP gain is significant, FID does not increase significantly, and the guard passes. A
  significant FID increase is reported as a loss of realism, whatever CLIP does.
- **Secondary:** every evaluator (E1–E6) at every N for every rule; the scaling curves; the
  oracle for E1–E3; verifier–evaluator correlations; verifier validity on the real tiles; the
  selected-index histogram (position bias).
- **Secondary, R2000 prefix:** the same analysis restricted to the first 2,000 tiles of RFULL,
  which are the R2000 tiles of Study 1. This sets Ω-B next to the Ω-XL pilot on identical tiles
  and seeds. FID is biased at that size, so its prefix values are compared only with each other,
  never with the full-split values.

### Statistics

As version 1 (amendment 1), plus FID:

- E1–E3: mean paired difference vs N = 1, percentile bootstrap over tiles, 10,000 resamples,
  seed 12345.
- E4, E5 (KID): full-sample unbiased MMD² with the KID kernel; tile bootstrap, 1,000 resamples,
  seed 54321; within-set pairs of the same tile excluded.
- **E6 (FID):** the same 1,000 tile resamples as KID (seed 54321), so the FID and KID intervals
  are paired. For a resample with tile counts w, FID is computed from the weighted mean and the
  weighted covariance with normaliser Σw − 1. This equals FID of the resampled set with each
  tile's row repeated w times (checked in `tests/test_bon_stats.py` to 1e-9). The real-image
  statistics are resampled with the same weights. tr((S_g S_r)^½) is computed as
  Σ √eigvalsh(A S_g A) with A = S_r^½, which has the same eigenvalues as the torchmetrics form.
- p-values: 2 · min(P(Δ* ≤ 0), P(Δ* ≥ 0)), floored at 1/B; with B = 1,000 for FID the floor is
  0.001, so the smallest Holm-adjusted FID p-value is 0.006.
- **KID computation at this size.** On 14,932 tiles the float64 CPU kernel computation is
  too slow (about 70 TFLOP per analysis). KID is computed in float32 on the GPU with TF32
  disabled and the reductions in float64 (`eotts/stats/kid.py`). The two paths agree to about
  1e-5 relative on MMD² differences (test on 3,000 × 2,048 features). As a check on real data,
  before the Study 2 analysis we re-run the Study 1 R2000 analysis with the GPU path and compare
  it with the committed `results/bon/R2000.json`. If any KID estimate or CI bound differs by more
  than 1 % of its CI width, the float64 path is used instead, however long it takes.
- All configurations are reported, including null and negative results.

### Anchors

1. **Paper-protocol baseline for Ω-B on all of Git-Rand.** `scripts/04_run_baseline.py --model
   omega_base --split random --batch-size 32`: upstream generation (fp32, upstream seed 42),
   upstream metrics (FID, sFID, CLIP, LPIPS, SSIM) and KID, like the Ω-XL baseline. The paper
   reports Ω-B only in its ablation (Table 3: FID 21.97, sFID 6.25, SSIM 0.2632, CLIP 0.2759,
   400k training steps). That table does not name the split, gives no LPIPS, and it is not
   stated whether the released `omega_base` weights are that checkpoint. We put our values next
   to these as a plausibility check, **not** as a reproduction claim.
2. **bf16 BoN-1 vs the fp32 baseline, all tiles, different noise.** E1–E3 with CIs, KID (E4,
   E5) and FID (E6) with a paired bootstrap CI.
3. **bf16 vs fp32, same noise, first 500 tiles of RFULL.** Sample 0 of these tiles is also
   generated in fp32 with the identical seed (`05_generate_bon.py --anchor-fp32 --anchor-tiles
   500`). Per-image PSNR, and E1–E4 for both.

**Stop rule** (as amendment 1.4, extended to FID): an anchor triggers it if its absolute
difference on a metric exceeds the half-width of the narrowest N = 16 vs N = 1 CI for that
metric on RFULL. This is checked for E1–E3 on anchors 2 and 3 and for FID on anchor 2. If it
triggers, we stop and report before interpreting the endpoints.

## Budget

Measured with `scripts/03_benchmark.py --model omega_base` on the RTX 3060 before this protocol
was committed: 2.34 s per image in bf16 and 3.29 s in fp32 at batch 32; fp32 vs bf16 on the
same noise, 56.2 dB mean PSNR over 8 tiles. In Study 1 the real run was 17 % slower than the
benchmark (VAE decoding, PNG writing, data loading), and the hours below include that.

| Run | Images | GPU-hours, about |
| --- | --- | --- |
| Anchor 1: paper-protocol baseline, fp32, all of Git-Rand | 14,932 | 16 |
| RFULL, 16 samples, bf16 | 238,912 | 180 |
| Anchor 3: first 500 tiles, sample 0, fp32 | 500 | 0.5 |
| Scoring (verifiers, E1–E5) | 254,344 | 12 (estimate) |
| **Total** | | **≈ 210 (about 9 days)** |

The budget does not change any decision above: N, tiles and metrics are fixed.

## Order of work

1. Anchor 1 (baseline), then RFULL generation, then anchor 3.
3. Scoring (`06_score_bon.py`), the GPU-KID check on Study 1 R2000, the analysis
   (`10_analyze_v2.py`).

## Limitations known in advance

- One base model per study. Ω-B and Ω-XL are compared only descriptively (R2000 prefix).
- One split. Study 2 says nothing about held-out geography; the Study 1 S500 result (realism
  loss with V2, seen in DINOv2 features only) is not re-tested here.
- Two text-alignment verifiers, no structural verifier.
- No clean paper reference for Ω-B on Git-Rand (anchor 1).
- Verifier training data may overlap with Git-10M; not checked.

## Deviations

None yet.
