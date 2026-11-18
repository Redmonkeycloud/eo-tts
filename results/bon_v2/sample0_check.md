# POST-HOC: is sample 0 different from samples 1-15?

z = (sample 0 − mean of samples 1–15) / SD of samples 1–15, over per-sample set means; rank 1 = highest of 16.

## R2000_omega_base

| Quantity | sample 0 | mean 1–15 | SD 1–15 | z | rank of sample 0 |
| --- | --- | --- | --- | --- | --- |
| evaluators.clip_cos | 0.28330 | 0.28336 | 0.00047 | -0.13 | 8 |
| evaluators.lpips | 0.33784 | 0.33835 | 0.00081 | -0.64 | 11 |
| evaluators.ssim | 0.26127 | 0.26022 | 0.00052 | +2.03 | 1 |
| verifiers.v1_remoteclip_b32 | 0.26869 | 0.26916 | 0.00068 | -0.69 | 13 |
| verifiers.v2_georsclip_l14 | 0.22788 | 0.22789 | 0.00021 | -0.04 | 9 |
| image.file_kb | 107.27678 | 107.05904 | 0.44315 | +0.49 | 6 |
| image.brightness | 0.36239 | 0.36104 | 0.00319 | +0.42 | 6 |
| image.contrast | 0.10607 | 0.10785 | 0.00137 | -1.30 | 14 |
| image.saturation | 0.11396 | 0.11244 | 0.00113 | +1.34 | 3 |
| image.sharpness_lapvar | 8.76472 | 8.85986 | 0.20689 | -0.46 | 11 |

generation.json: `{"model": "omega_base", "precision": "bf16", "batch_size": 32, "sample_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "num_steps": 100, "runs": [{"started": "2026-11-16T18:00:35", "finished": null, "pending_at_start": 32000, "s_per_image": null}, {"started": "2026-11-17T18:39:26", "finished": "2026-11-18T00:10:05", "pending_at_start": 6880, "s_per_image": 2.882}]}`

## S500_omega_base

| Quantity | sample 0 | mean 1–15 | SD 1–15 | z | rank of sample 0 |
| --- | --- | --- | --- | --- | --- |
| evaluators.clip_cos | 0.27585 | 0.27588 | 0.00100 | -0.04 | 9 |
| evaluators.lpips | 0.34792 | 0.34794 | 0.00161 | -0.01 | 9 |
| evaluators.ssim | 0.33386 | 0.33132 | 0.00205 | +1.24 | 2 |
| verifiers.v1_remoteclip_b32 | 0.24438 | 0.24357 | 0.00126 | +0.64 | 5 |
| verifiers.v2_georsclip_l14 | 0.22190 | 0.22192 | 0.00061 | -0.04 | 9 |
| image.file_kb | 93.34015 | 93.52447 | 0.39244 | -0.47 | 11 |
| image.brightness | 0.37180 | 0.37236 | 0.00286 | -0.20 | 11 |
| image.contrast | 0.07940 | 0.08004 | 0.00185 | -0.35 | 9 |
| image.saturation | 0.12442 | 0.12531 | 0.00093 | -0.95 | 13 |
| image.sharpness_lapvar | 2.23627 | 2.31345 | 0.14927 | -0.52 | 11 |

generation.json: `{"model": "omega_base", "precision": "bf16", "batch_size": 32, "sample_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "num_steps": 100, "runs": [{"started": "2026-11-16T11:01:32", "finished": "2026-11-16T17:22:29", "pending_at_start": 8000, "s_per_image": 2.856}]}`

## R2000

| Quantity | sample 0 | mean 1–15 | SD 1–15 | z | rank of sample 0 |
| --- | --- | --- | --- | --- | --- |
| evaluators.clip_cos | 0.28882 | 0.28853 | 0.00042 | +0.70 | 4 |
| evaluators.lpips | 0.33826 | 0.33921 | 0.00078 | -1.22 | 14 |
| evaluators.ssim | 0.26511 | 0.26384 | 0.00051 | +2.47 | 1 |
| verifiers.v1_remoteclip_b32 | 0.26442 | 0.26391 | 0.00050 | +1.02 | 4 |
| verifiers.v2_georsclip_l14 | 0.23021 | 0.23029 | 0.00023 | -0.35 | 10 |
| image.file_kb | 106.56553 | 106.40538 | 0.39936 | +0.40 | 6 |
| image.brightness | 0.36426 | 0.36241 | 0.00323 | +0.57 | 4 |
| image.contrast | 0.10763 | 0.10986 | 0.00142 | -1.58 | 16 |
| image.saturation | 0.11568 | 0.11407 | 0.00101 | +1.60 | 2 |
| image.sharpness_lapvar | 8.10314 | 8.24705 | 0.26365 | -0.55 | 10 |

generation.json: `{"model": "omega_xl", "precision": "bf16", "batch_size": 32, "sample_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "num_steps": 100, "runs": [{"started": "2026-10-30T17:51:28", "finished": "2026-11-02T12:01:50", "pending_at_start": 32000, "s_per_image": 7.444}]}`

## S500

| Quantity | sample 0 | mean 1–15 | SD 1–15 | z | rank of sample 0 |
| --- | --- | --- | --- | --- | --- |
| evaluators.clip_cos | 0.28157 | 0.27956 | 0.00111 | +1.81 | 1 |
| evaluators.lpips | 0.35889 | 0.35873 | 0.00164 | +0.10 | 9 |
| evaluators.ssim | 0.33984 | 0.33828 | 0.00193 | +0.81 | 4 |
| verifiers.v1_remoteclip_b32 | 0.23650 | 0.23572 | 0.00162 | +0.48 | 7 |
| verifiers.v2_georsclip_l14 | 0.22365 | 0.22297 | 0.00052 | +1.31 | 2 |
| image.file_kb | 91.68025 | 91.90585 | 0.37434 | -0.60 | 11 |
| image.brightness | 0.38614 | 0.38762 | 0.00320 | -0.46 | 11 |
| image.contrast | 0.07804 | 0.07901 | 0.00187 | -0.52 | 11 |
| image.saturation | 0.12898 | 0.12923 | 0.00110 | -0.23 | 9 |
| image.sharpness_lapvar | 1.98093 | 2.09061 | 0.14738 | -0.74 | 13 |

generation.json: `{"model": "omega_xl", "precision": "bf16", "batch_size": 32, "sample_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "num_steps": 100, "runs": [{"started": "2026-11-02T12:02:14", "finished": "2026-11-03T04:36:02", "pending_at_start": 8000, "s_per_image": 7.451}]}`

