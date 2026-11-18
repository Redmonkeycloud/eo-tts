# POST-HOC: FID and KID across independent draws of the same model

Not pre-registered. Written after the stop rule triggered on R2000 (protocol v2, Reports). Draw k = sample k of every tile: 16 independent draws of the set from the same model, no selection.

| Run | Tiles | FID mean ± SD over draws | FID min – max | abs. difference between two draws: median / p95 / max | KID-Inception ×10³ mean ± SD | KID-DINOv2 ×10³ mean ± SD |
| --- | --- | --- | --- | --- | --- | --- |
| R2000_omega_base | 2000 | 39.70 ± 0.34 | 39.17 – 40.47 | 0.32 / 0.92 / 1.30 | 11.845 ± 0.310 | 1188.2 ± 13.7 |
| S500_omega_base | 500 | 70.96 ± 0.95 | 69.75 – 73.17 | 0.80 / 2.79 / 3.41 | 12.999 ± 0.995 | 1312.1 ± 46.7 |
| R2000 | 2000 | 29.32 ± 0.28 | 28.75 – 29.69 | 0.28 / 0.74 / 0.94 | 3.970 ± 0.145 | 471.1 ± 8.6 |
| S500 | 500 | 68.94 ± 0.87 | 67.72 – 70.80 | 0.82 / 2.40 / 3.08 | 10.747 ± 0.561 | 544.3 ± 17.2 |
