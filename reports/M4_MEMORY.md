# M4 linear-memory diagnostic (S3, 33-tap FIRs)

| Run | Model | Device | Seed | Taps | Params | Test ESR | 100-300 Hz | Train-window ESR | Transfer | Residual ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m4_memory_fulltone_s3t33_seed0_v1 | S3 | fulltone | 0 | 33 | 1276 | 0.1404 | 0.0545 | 0.1041 | +0.0363 | 6.12e-05 |
| m4_memory_fulltone_s3t33_seed1_v1 | S3 | fulltone | 1 | 33 | 1276 | 0.1445 | 0.0643 | 0.1146 | +0.0299 | 1.38e-06 |
| m4_memory_fulltone_s3t33_seed2_v1 | S3 | fulltone | 2 | 33 | 1276 | 0.1333 | 0.0522 | 0.1127 | +0.0205 | 4.95e-07 |
| m4_fulltone_s3_seed0_v1 | S3 | fulltone | 0 | 17 | 1244 | 0.2319 | 0.1340 | 0.1431 | +0.0887 | 3.42e-06 |
| m4_fulltone_s3_seed1_v1 | S3 | fulltone | 1 | 17 | 1244 | 0.2044 | 0.1115 | 0.1552 | +0.0492 | 3.05e-07 |
| m4_fulltone_s3_seed2_v1 | S3 | fulltone | 2 | 17 | 1244 | 0.2112 | 0.1182 | 0.1430 | +0.0681 | 4.50e-06 |
| m4_fulltone_s4_seed0_v1 | S4 | fulltone | 0 | 17 | 1244 | 0.2318 | 0.1339 | 0.1437 | +0.0881 | 2.90e-06 |
| m4_fulltone_s4_seed1_v1 | S4 | fulltone | 1 | 17 | 1244 | 0.2158 | 0.1191 | 0.1505 | +0.0653 | 1.19e-05 |
| m4_fulltone_s4_seed2_v1 | S4 | fulltone | 2 | 17 | 1244 | 0.2116 | 0.1184 | 0.1428 | +0.0689 | 3.78e-06 |
| m4_fulltone_b0_seed0_v1 | B0 | fulltone | 0 | - | 12145 | 0.0668 | 0.0272 | 0.0387 | +0.0280 | 0.00e+00 |
| m4_fulltone_b0_seed1_v1 | B0 | fulltone | 1 | - | 12145 | 0.0639 | 0.0231 | 0.0320 | +0.0319 | 0.00e+00 |
| m4_fulltone_b0_seed2_v1 | B0 | fulltone | 2 | - | 12145 | 0.0409 | 0.0147 | 0.0166 | +0.0243 | 0.00e+00 |
| m4_memory_bigmuff_s3t33_seed0_v1 | S3 | bigmuff | 0 | 33 | 1276 | 0.9444 | 0.0650 | 0.9544 | -0.0101 | 7.30e-03 |
| m4_memory_bigmuff_s3t33_seed1_v1 | S3 | bigmuff | 1 | 33 | 1276 | 0.9443 | 0.0629 | 0.9511 | -0.0068 | 1.16e-03 |
| m4_memory_bigmuff_s3t33_seed2_v1 | S3 | bigmuff | 2 | 33 | 1276 | 0.9450 | 0.0642 | 0.9493 | -0.0043 | 3.83e-03 |
| m4_bigmuff_s3_seed0_v1 | S3 | bigmuff | 0 | 17 | 1244 | 0.9670 | 0.0685 | 0.9561 | +0.0109 | 8.69e-03 |
| m4_bigmuff_s3_seed1_v1 | S3 | bigmuff | 1 | 17 | 1244 | 0.9493 | 0.0649 | 0.9495 | -0.0002 | 2.27e-04 |
| m4_bigmuff_s3_seed2_v1 | S3 | bigmuff | 2 | 17 | 1244 | 0.9961 | 0.0601 | 0.9900 | +0.0060 | 3.19e-04 |
| m4_bigmuff_s4_seed0_v1 | S4 | bigmuff | 0 | 17 | 1244 | 0.9925 | 0.0876 | 0.9710 | +0.0215 | 2.21e-03 |
| m4_bigmuff_s4_seed1_v1 | S4 | bigmuff | 1 | 17 | 1244 | 0.9487 | 0.0648 | 0.9477 | +0.0010 | 1.97e-05 |
| m4_bigmuff_s4_seed2_v1 | S4 | bigmuff | 2 | 17 | 1244 | 0.9595 | 0.0668 | 0.9654 | -0.0060 | 1.98e-03 |
| m4_bigmuff_b0_seed0_v1 | B0 | bigmuff | 0 | - | 12145 | 0.5481 | 0.0133 | 0.5443 | +0.0038 | 0.00e+00 |
| m4_bigmuff_b0_seed1_v1 | B0 | bigmuff | 1 | - | 12145 | 0.6535 | 0.0141 | 0.5192 | +0.1343 | 0.00e+00 |
| m4_bigmuff_b0_seed2_v1 | B0 | bigmuff | 2 | - | 12145 | 0.5969 | 0.0109 | 0.5164 | +0.0806 | 0.00e+00 |

```json
{
  "primary_model": "S3",
  "primary_device": "fulltone_full_drive_2",
  "seeds_present": [
    0,
    1,
    2
  ],
  "medians": {
    "test_esr": 0.14040445733663934,
    "band_error": 0.05447250086190326,
    "train_window_esr": 0.11271604155960621,
    "transfer_loss": 0.029897072411001993
  },
  "seed0": {
    "run_id": "m4_memory_fulltone_s3t33_seed0_v1",
    "phase": "M4_MEMORY",
    "model": "S3",
    "device": "fulltone_full_drive_2",
    "seed": 0,
    "taps": 33,
    "parameters": 1276,
    "best_step": 160,
    "test_esr": 0.14040445733663934,
    "test_esr_recomputed": 0.14040445732304221,
    "gain_error": -0.1942441007116935,
    "residual_energy_ratio": 6.122784423165876e-05,
    "band_error": 0.05447250086190326,
    "train_window_esr": 0.10407557007652472,
    "transfer_loss": 0.036328887260114615
  },
  "verdict": "partial",
  "discriminant": "mixed",
  "gap_closure_fraction_vs_a2_median": 0.4805133791916161,
  "regression_control": {
    "device": "electro_harmonix_big_muff",
    "median_test_esr_new": 0.944370542995317,
    "median_test_esr_reference": 0.9669655454152576,
    "median_increase": -0.022595002419940613,
    "passes": true
  }
}
```
