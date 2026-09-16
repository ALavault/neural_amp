# M4 spline-grid diagnostic (S3, 33-tap FIRs, knots over +-0.4)

| Run | Model | Device | Seed | Taps | Grid | Params | Test ESR | 100-300 Hz | Train-window ESR | Transfer | Residual ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m4_grid_fulltone_s3t33_seed0_v1 | S3 | fulltone | 0 | 33 | +-0.4 | 1276 | 0.0628 | 0.0222 | 0.0464 | +0.0164 | 3.01e-07 |
| m4_grid_fulltone_s3t33_seed1_v1 | S3 | fulltone | 1 | 33 | +-0.4 | 1276 | 0.0791 | 0.0324 | 0.0543 | +0.0248 | 3.47e-06 |
| m4_grid_fulltone_s3t33_seed2_v1 | S3 | fulltone | 2 | 33 | +-0.4 | 1276 | 0.0857 | 0.0378 | 0.0529 | +0.0328 | 1.50e-06 |
| m4_memory_fulltone_s3t33_seed0_v1 | S3 | fulltone | 0 | 33 | +-2 | 1276 | 0.1404 | 0.0545 | 0.1041 | +0.0363 | 6.12e-05 |
| m4_memory_fulltone_s3t33_seed1_v1 | S3 | fulltone | 1 | 33 | +-2 | 1276 | 0.1445 | 0.0643 | 0.1146 | +0.0299 | 1.38e-06 |
| m4_memory_fulltone_s3t33_seed2_v1 | S3 | fulltone | 2 | 33 | +-2 | 1276 | 0.1333 | 0.0522 | 0.1127 | +0.0205 | 4.95e-07 |
| m4_fulltone_b0_seed0_v1 | B0 | fulltone | 0 | - | - | 12145 | 0.0668 | 0.0272 | 0.0387 | +0.0280 | 0.00e+00 |
| m4_fulltone_b0_seed1_v1 | B0 | fulltone | 1 | - | - | 12145 | 0.0639 | 0.0231 | 0.0320 | +0.0319 | 0.00e+00 |
| m4_fulltone_b0_seed2_v1 | B0 | fulltone | 2 | - | - | 12145 | 0.0409 | 0.0147 | 0.0166 | +0.0243 | 0.00e+00 |
| m4_grid_bigmuff_s3t33_seed0_v1 | S3 | bigmuff | 0 | 33 | +-0.4 | 1276 | 1.0353 | 0.0418 | 0.9835 | +0.0518 | 1.44e-02 |
| m4_grid_bigmuff_s3t33_seed1_v1 | S3 | bigmuff | 1 | 33 | +-0.4 | 1276 | 1.0003 | 0.0572 | 0.9614 | +0.0389 | 9.72e-05 |
| m4_grid_bigmuff_s3t33_seed2_v1 | S3 | bigmuff | 2 | 33 | +-0.4 | 1276 | 1.0265 | 0.0484 | 0.9921 | +0.0344 | 2.17e-04 |
| m4_memory_bigmuff_s3t33_seed0_v1 | S3 | bigmuff | 0 | 33 | +-2 | 1276 | 0.9444 | 0.0650 | 0.9544 | -0.0101 | 7.30e-03 |
| m4_memory_bigmuff_s3t33_seed1_v1 | S3 | bigmuff | 1 | 33 | +-2 | 1276 | 0.9443 | 0.0629 | 0.9511 | -0.0068 | 1.16e-03 |
| m4_memory_bigmuff_s3t33_seed2_v1 | S3 | bigmuff | 2 | 33 | +-2 | 1276 | 0.9450 | 0.0642 | 0.9493 | -0.0043 | 3.83e-03 |
| m4_bigmuff_b0_seed0_v1 | B0 | bigmuff | 0 | - | - | 12145 | 0.5481 | 0.0133 | 0.5443 | +0.0038 | 0.00e+00 |
| m4_bigmuff_b0_seed1_v1 | B0 | bigmuff | 1 | - | - | 12145 | 0.6535 | 0.0141 | 0.5192 | +0.1343 | 0.00e+00 |
| m4_bigmuff_b0_seed2_v1 | B0 | bigmuff | 2 | - | - | 12145 | 0.5969 | 0.0109 | 0.5164 | +0.0806 | 0.00e+00 |

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
    "test_esr": 0.07912581839254824,
    "band_error": 0.03238324178365274,
    "train_window_esr": 0.05291304217785864,
    "transfer_loss": 0.02479234604938796
  },
  "reference_medians": {
    "test_esr": 0.14040445733663934,
    "band_error": 0.05447250086190326,
    "train_window_esr": 0.11271604155960621,
    "transfer_loss": 0.029897072411001993
  },
  "train_window_esr_decrease": 0.059802999381747574,
  "verdict": "confirmed",
  "regression_control": {
    "device": "electro_harmonix_big_muff",
    "median_test_esr_new": 1.026522417858757,
    "median_test_esr_reference": 0.944370542995317,
    "median_increase": 0.08215187486344011,
    "passes": false
  }
}
```
