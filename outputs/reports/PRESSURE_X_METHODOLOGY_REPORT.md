# PRESSURE-X: Context-Aware Cricket Pressure Intelligence System
## Official Research Methodology Report — Rajasthan Royals SupeRR Selector Season 2

---

### Key System Metrics:
- **Total Deliveries Processed**: 200,999
- **Total Matches**: 883 (371 IPL, 482 T20I, 30 Other)
- **Legal Deliveries**: 192,445 | **Illegal Deliveries**: 8,554 (7,714 wides, 576 overstep, 264 other no-balls)
- **ML Validation**: 5-Fold Chronological Out-of-Sample Predictions (Zero In-Sample Contamination)
- **Holdout Chases WP AUC**: 0.8763 | **Combined Holdout AUC**: 0.7661
- **Monte Carlo Resamples**: 1,000 Match-Clustered Iterations

---

## 1. Executive Summary
Traditional T20 cricket player evaluation relies heavily on uncontextualized aggregate metrics: batting average, strike rate, bowling economy, and wickets. These metrics fundamentally suffer from situational blindness: scoring 20 runs off 10 balls when chasing 18 off the final 2 overs under heavy collapse stress is treated identically to scoring 20 off 10 in a low-stakes dead rubber.

PRESSURE-X introduces a statistically defensible, multi-layered cricket pressure intelligence framework developed for the Rajasthan Royals SupeRR Selector competition. The system reconstructs match-state physics across 200,999 delivery records spanning 883 matches, calculates situational Win Probability via calibrated gradient boosting, derives continuous situational pressure indices, eliminates in-sample prediction leakage via 5-fold chronological out-of-sample cross-validation, enforces legal-delivery denominator scoring, and quantifies ranking uncertainty through 1,000 match-clustered Monte Carlo simulations.

---

## 2. Problem Statement & Objectives
The objective of PRESSURE-X is to identify the Top 10 Batters and Top 10 Bowlers who disproportionately elevate match-winning probability under high pressure:
1. Mathematical Cricket Accounting: Rigorously distinguishing illegal deliveries (wides, no-balls) from legal balls.
2. Ex-Ante Pressure Modeling: Quantifying situational tension before the delivery is bowled.
3. Leakage-Free Out-of-Sample Predictions: Ensuring expected batting runs, expected bowling concessions, and win probabilities are out-of-sample.
4. Empirical Bayes Regularization: Shrinking extreme small-sample noise towards historical priors.
5. Uncertainty Quantification: Establishing bootstrap confidence intervals and Top 10 inclusion probabilities.

---

## 3. Dataset & Data Quality Audit
- **ball_by_ball**: 200,999 rows × 33 cols (0 duplicate rows).
- **matches**: 883 rows × 22 cols.
- **competitions**: 141 rows × 4 cols.
- **teams**: 353 rows × 3 cols.
- **playing_eleven**: 28,381 rows × 9 cols.
- **competition_team_player_mapping**: 6,413 rows × 6 cols.

### Legal Delivery Accounting:
- Legal Deliveries: 192,445 (189,167 no-extras + 2,748 leg-byes + 529 byes + 1 Law 41 penalty delivery).
- Illegal Deliveries: 8,554 (7,714 wides + 576 overstep no-balls + 264 non-overstep no-balls).
- Denominators: Balls faced and overs bowled strictly use legal deliveries.

---

## 4. Player and Match Data Processing
Player identity is strictly tracked via canonical integer IDs (`striker_id` and `bowler_id`).
Homonymous entities (e.g., Rashid Khan ID 4722 AFG/GT vs ID 20172 NEP; Abdul Samad ID 7446 SRH/LSG vs ID 6922 PAK) are kept completely distinct.

---

## 5. Situational Pressure Framework
```
Chase Batting Pressure (2nd Innings):
P_bat = min(1.0, [0.40 * C_t + 0.35 * RRR_stress + 0.25 * Wicket_stress] * w_phase / 1.15)

1st Innings Batting Pressure:
P_bat = min(1.0, [0.40 * C_t + 0.35 * Inn1_urgency + 0.25 * Wicket_stress] * w_phase / 1.15)

Chase Bowling Pressure (2nd Innings):
P_bowl = min(1.0, [0.45 * C_t + 0.30 * (1 - RRR_stress) * Opp_resource + 0.25 * I(Death)] * w_phase / 1.15)

1st Innings Bowling Pressure:
P_bowl = min(1.0, [0.45 * C_t + 0.35 * Opp_resource + 0.20 * I(Death)] * w_phase / 1.15)
```
- Normalization Factor: 1.15 represents max phase multiplier in death overs.
- High-Pressure Band: Standardized as P > 0.666.
- Delivery Leverage: L_t = |WP_post - WP_pre| measured ex-post.

---

## 6. Win Probability Model Architecture
Dual LightGBM classifiers calibrated via sigmoid (Platt) scaling:
- Innings 1 Features: balls_remaining, wickets_remaining, innings_runs, crr, phase_encoded.
- Innings 2 Features: runs_required, balls_remaining, wickets_remaining, rrr, crr, target, phase_encoded.
- Exact Boundaries: Target reached = 1.0; All out / balls exhausted = 0.0.

---

## 7. Expected Batting & Bowling Models (OOF ML Architecture)
- Batting Runs MAE: 1.2847 (vs Baseline MAE: 1.3046, +1.52% lift)
- Batting Runs RMSE: 1.7141 (vs Baseline RMSE: 1.7301, +0.93% lift)
- Bowling Conceded MAE: 1.2721 (vs Baseline MAE: 1.2858, +1.06% lift)
- Bowling Conceded RMSE: 1.7011 (vs Baseline RMSE: 1.7135, +0.72% lift)
- Boundary Classifier AUC: 0.6005 | Dot Ball AUC: 0.6001 | Bowler Wicket AUC: 0.5912

---

## 8. Batting and Bowling Rating Methodology
- RAA = ActualRuns - ExpectedRuns_OOF
- PRAA = RAA * (1.0 + P_bat)
- RPA = ExpectedConceded_OOF - ConcededActual
- PRPA = RPA * (1.0 + P_bowl)
- Bayesian Volume Priors: K_bat = 80, K_bowl = 90, K_hp = 30 balls.

### Complete BPI Formulas:
```
BPI = 50.0 + 1.0 * Shrunk_PRAA/100 + 0.8 * Shrunk_HP_RAA/100 + 12.0 * WPA_total + 5.0 * (Reliability - 0.5)
Bowling_BPI = 50.0 + 1.0 * Shrunk_PRPA/100 + 0.8 * Shrunk_HP_RPA/100 + 3.0 * WAA/100 + 12.0 * WPA_total + 5.0 * (Reliability - 0.5)
```

---

## 9. Eligibility Criteria & Threshold Sensitivity
- Min Legal Balls Faced / Bowled: &ge; 250 legal balls
- Min Matches: &ge; 8 matches
- Min High-Pressure Balls: &ge; 20 legal balls
- Qualified Players: 70 Batters, 103 Bowlers (Combined).

---

## 10. Chronological Validation & Calibration Integrity
- Train Matches: 671 (2022-01-22 to 2025-11-09)
- Unseen Holdout Matches: 171 (2025-11-13 to 2026-11-01)
- Disjoint Match Guarantee: True (0 overlap, max(train) < min(test))
- Innings 2 Chases AUC: **0.8763** (Brier: 0.1482, Log-Loss: 0.4476)
- Combined Holdout AUC: **0.7661** (Brier: 0.1973, Log-Loss: 0.5698)

---

## 11. Monte Carlo Uncertainty Analysis (1,000 Iterations)
Match-clustered bootstrap resampling with replacement across 867 matches.
Convergence verified across 250, 500, and 1,000 iterations: |&Delta; P(Top 10)| < 2.5%, median ranks 100% stable.

---

## 12. Top 10 Batter Results (PRESSURE-X Model Rankings)
| Rank | Player Name | Matches | Legal Balls | Runs | Exp Runs | PRAA | HP-RAA/100 | WPA | P(Top 10) | 95% CI | BPI |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | **Tim David** | 98 | 1210 | 2178 | 1862.3 | +464.9 | +32.0 | +5.64 | **96.9%** | [1 - 11] | **179.7** |
| **#2** | **Suryakumar Yadav** | 139 | 2748 | 4595 | 3930.6 | +907.7 | +39.5 | +5.28 | **97.8%** | [1 - 10] | **177.3** |
| **#3** | **Heinrich Klaasen** | 72 | 1408 | 2363 | 2136.4 | +332.3 | +20.0 | +3.43 | **66.8%** | [1 - 27] | **129.8** |
| **#4** | **Rajat Patidar** | 48 | 916 | 1541 | 1376.5 | +238.0 | +44.8 | +1.52 | **56.6%** | [1 - 45] | **128.3** |
| **#5** | **Andre Russell** | 57 | 724 | 1254 | 1071.0 | +277.7 | +32.1 | +1.37 | **64.4%** | [2 - 32] | **127.1** |
| **#6** | **Jos Buttler** | 121 | 2820 | 4396 | 4167.8 | +295.5 | +29.4 | +3.47 | **57.9%** | [2 - 29] | **125.7** |
| **#7** | **Namboori Tilak Varma** | 127 | 2500 | 3596 | 3557.5 | +109.4 | +37.9 | +2.92 | **52.4%** | [1 - 42] | **120.0** |
| **#8** | **Naman Dhir** | 46 | 588 | 992 | 850.5 | +207.2 | +23.9 | +1.34 | **48.2%** | [2 - 36] | **117.2** |
| **#9** | **Donovan Ferreira** | 30 | 341 | 562 | 483.7 | +119.3 | +35.5 | +0.36 | **39.2%** | [2 - 47] | **111.9** |
| **#10** | **Shreyas Iyer** | 98 | 1858 | 2856 | 2731.9 | +177.2 | +19.4 | +3.07 | **40.3%** | [2 - 41] | **111.9** |

---

## 13. Top 10 Bowler Results (PRESSURE-X Model Rankings)
| Rank | Player Name | Matches | Legal Balls | Runs Conc. | Exp Conc. | Wkts (WAA) | PRPA | HP-RPA/100 | WPA | P(Top 10) | 95% CI | BPI |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#1** | **Rashid Khan** | 122 | 2747 | 3403 | 3683.5 | 153 (+9.5) | +391.2 | +16.8 | +4.94 | **79.9%** | [1 - 26] | **137.6** |
| **#2** | **Jasprit Bumrah** | 75 | 1708 | 2069 | 2587.2 | 85 (-11.0) | +743.3 | +36.6 | +1.55 | **84.0%** | [1 - 21] | **137.4** |
| **#3** | **Sunil Narine** | 67 | 1533 | 1754 | 2115.5 | 66 (-10.5) | +506.2 | +29.1 | +2.56 | **80.8%** | [1 - 24] | **133.3** |
| **#4** | **Ehsan Khan** | 14 | 323 | 384 | 473.2 | 19 (-0.3) | +137.3 | +35.6 | +1.08 | **64.4%** | [1 - 31] | **124.6** |
| **#5** | **Nathan Ellis** | 43 | 916 | 1260 | 1418.0 | 58 (+3.7) | +222.1 | +18.9 | +2.64 | **56.8%** | [1 - 37] | **120.4** |
| **#6** | **Bhuvneshwar Kumar** | 101 | 2221 | 3069 | 3336.7 | 115 (-9.6) | +377.5 | +18.0 | +2.83 | **47.6%** | [1 - 44] | **113.6** |
| **#7** | **Matheesha Pathirana** | 54 | 1132 | 1670 | 1905.1 | 82 (+5.2) | +330.6 | +25.8 | +1.09 | **44.4%** | [2 - 39] | **112.2** |
| **#8** | **Kuldeep Yadav** | 89 | 1914 | 2557 | 2669.8 | 118 (+19.1) | +164.4 | +23.3 | +2.58 | **41.6%** | [1 - 44] | **110.9** |
| **#9** | **Axar Patel** | 136 | 2670 | 3512 | 3645.9 | 120 (-13.6) | +172.0 | -1.1 | +4.31 | **34.2%** | [2 - 46] | **105.8** |
| **#10** | **Mitchell Santner** | 83 | 1737 | 2155 | 2324.0 | 87 (+0.2) | +199.9 | +21.4 | +2.23 | **34.0%** | [2 - 47] | **105.0** |

---

## 14. Sensitivity and Component Ablation Analysis
- Layer 0 (Classical): Uncontextualized aggregate statistics (867 matches | 200,999 deliveries).
- Layer 1 (Contextual): LightGBM RAA & RPA (Runs MAE = 1.2847 vs Baseline MAE = 1.3046, +1.52% lift).
- Layer 2 (Pressure-Weighted): PRAA & PRPA scaled by state strain (Mean Leverage = 0.0153 | 15,114 HP deliveries).
- Layer 3 (Full Architecture): PRAA + HP-RAA + WPA + Empirical Bayes + Monte Carlo (Holdout WP AUC: 0.8763 Inn2, 0.7661 Comb).

---

## 15. Limitations and Strategic Recommendations
- Optical tracking (release point, swing, trajectory) absent from tabular feeds.
- Fielding positions modeled via venue historical distributions rather than real-time spatial telemetry.
- Recommendations:
  1. Target elite high-pressure finishers (Tim David, Heinrich Klaasen).
  2. Prioritize death bowling containment (Jasprit Bumrah, Sunil Narine, Matheesha Pathirana).
  3. Calibrate auction pricing using Monte Carlo stability bands (P(Top 10) &ge; 75%).
