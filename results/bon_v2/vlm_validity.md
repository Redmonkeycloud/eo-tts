# V3 candidates on real tiles (R2000, first 500 tiles, 20 distractor captions)

Share of distractor captions scoring above the tile's own caption (0 = own caption first, 0.5 = chance) and how often the own caption is first; 95 % bootstrap CIs over tiles.

| Verifier | Kind | Share above own [95 % CI] | Own caption first [95 % CI] | P(yes)+P(no) | s/pair |
| --- | --- | --- | --- | --- | --- |
| v1_remoteclip_b32 | clip | 0.086 [0.074, 0.097] | 0.456 [0.412, 0.502] | nan | nan |
| v2_georsclip_l14 | clip | 0.033 [0.026, 0.041] | 0.730 [0.692, 0.766] | nan | nan |
| v3_qwen25vl_3b | vlm | 0.042 [0.035, 0.051] | 0.622 [0.578, 0.666] | 0.995 | 0.194 |
| v3_internvl3_2b | vlm | 0.022 [0.017, 0.028] | 0.768 [0.732, 0.804] | 1.000 | 0.135 |

**V3 = v3_internvl3_2b** (lowest mean share; rule fixed in amendment 2).
