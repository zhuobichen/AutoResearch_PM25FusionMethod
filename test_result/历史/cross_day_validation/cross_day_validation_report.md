# Cross-Day Validation Report - SuperStackingEnsemble Weights

## Weights Used (from SuperStackingEnsemble learned on Day 1)

| Model | Weight |
|-------|--------|
| RK-Poly | +2.934 |
| RK-Poly3 | -0.939 |
| RK-OLS | -0.996 |
| eVNA | +0.206 |
| aVNA | -0.216 |
| CMAQ | +0.100 |

## Results Comparison by Date

### Individual Method R2 Comparison

| Method | Day 1 | Day 2 | Day 3 |
|--------|-------|-------|-------|
| RK-Poly | 0.8519 | 0.8809 | 0.9061 |
| RK-Poly3 | 0.8461 | 0.8740 | 0.9024 |
| RK-OLS | 0.8494 | 0.8756 | 0.9042 |
| eVNA | 0.8100 | 0.8408 | 0.8716 |
| aVNA | 0.7941 | 0.8376 | 0.8784 |
| CMAQ | -0.0376 | 0.0015 | 0.2240 |
| SuperStackingEnsemble | 0.8179 | 0.8287 | 0.8793 |

### Detailed Metrics Comparison

#### SuperStackingEnsemble

| Date | Samples | R2 | MAE | RMSE | MB |
|------|---------|-----|-----|------|-----|
| 2020-01-01 | 1428 | 0.8179 | 8.68 | 12.25 | 4.62 |
| 2020-01-02 | 1446 | 0.8287 | 11.54 | 16.08 | 5.25 |
| 2020-01-03 | 1436 | 0.8793 | 12.43 | 17.32 | 6.09 |

### Analysis

1. **Cross-Day Generalization**: Using weights learned from Day 1, test on Day 2 and Day 3 data to observe if the weights have cross-day applicability.

2. **Weight Interpretation**:
   - RK-Poly weight is +2.934, the main contributing model
   - RK-Poly3 (-0.939) and RK-OLS (-0.996) have negative weights, serving as correction factors
   - eVNA (+0.206) and aVNA (-0.216) provide spatial correction
   - CMAQ (+0.100) serves as the base reference

3. **Comparison with Day 1**: Observe the difference in performance of each method on Day 2 and Day 3 compared to Day 1 to evaluate model stability.
