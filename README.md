# PRESSURE-X: Multi-Layer Context-Aware Cricket Pressure Intelligence System
### SupeRR Selector Season 2 — Advanced Sports Analytics Research
**Official Submission for Rajasthan Royals | Round 1 Problem Statement**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-33%2F33%20passing-brightgreen.svg)]()
[![Model Calibration](https://img.shields.io/badge/Holdout%20AUC-0.8763%20(Chases)-success.svg)](outputs/charts/win_prob_calibration.png)
[![Monte Carlo Stability](https://img.shields.io/badge/P(Top%2010)-93.6%25%20(SKY)%20%7C%2098.8%25%20(Bumrah)-blueviolet.svg)]()
[![Report Format](https://img.shields.io/badge/Submission-Methodology%20Report%20(PDF)-gold.svg)](outputs/reports/PRESSURE_X_METHODOLOGY_REPORT.pdf)

---

## 1. Executive Summary & Research Hypothesis

Conventional cricket statistics (Runs, Strike Rate, Wickets, Economy) measure aggregate outcomes on the scoreboard, but systematically fail to account for situational context, match strain, and leverage. Scoring 30 runs from 20 balls chasing 120 on a benign pitch has negligible win-probability impact; scoring 30 runs from 20 balls chasing 180 under an escalating required run rate against elite death bowling in a tight finish is match-defining.

**PRESSURE-X** solves the central question posed by the Rajasthan Royals SupeRR Selector Season 2 problem statement:
> **"When the match situation becomes difficult, which players consistently perform better than an equivalent player would be expected to perform in that exact situation?"**

To answer this, PRESSURE-X implements an end-to-end, ball-by-ball machine-learning architecture evaluated across **200,999 deliveries across 883 matches (2022–2026)** covering the Indian Premier League (IPL) and International T20s (T20I). Following a rigorous 8-phase audit and reconstruction:
- **Exact Delivery Reconciliation**: Delivery accounting isolates **192,445 legal balls** (189,167 standard balls + 2,748 leg-byes + 529 byes + 1 Law 41 fielding penalty) and **8,554 illegal deliveries** (7,714 wides + 576 overstep front-foot no-balls + 264 non-overstep no-balls).
- **Correct Denominators**: Batting strike rates and per-100-ball rates strictly use legal balls faced; bowling economies and per-100-ball concessions strictly use legal balls bowled. Fielding extras (byes, leg-byes, penalty runs) are not charged to the bowler.
- **Entity Resolution**: 1,082 canonical players are disambiguated by integer player IDs, preventing catastrophic cross-contamination between homonymous entities (e.g., Afghan Rashid Khan vs. Nepali Rashid Khan; Indian Abdul Samad vs. Pakistani Abdul Samad).
- **Out-of-Fold Machine Learning**: 5-fold chronological cross-validation generates out-of-sample expected runs, expected concessions, and baseline win probabilities.
- **Strict Chronological Holdout**: Evaluated on 171 completely unseen holdout matches (39,478 deliveries) partitioned strictly by match date (`max(train_dates) < min(test_dates)`), achieving **0.8763 ROC-AUC** in 2nd innings chases.
- **Uncertainty Quantification**: 1,000 match-clustered Monte Carlo bootstrap iterations establish 95% rank credibility intervals and Top-10 inclusion probabilities.

The resulting leaderboards crown **Suryakumar Yadav** (Batting Pressure Index: **158.9**, $P(\text{Top 10}) = 93.6\%$) and **Jasprit Bumrah** (Bowling Pressure Index: **153.3**, $P(\text{Top 10}) = 98.8\%$) as the undisputed elite pressure performers in modern T20 cricket. In IPL-only competition splits, **Suryakumar Yadav** (BPI: **137.8**) and **Jasprit Bumrah** (BPI: **146.8**) lead their respective categories.

---

## 2. Submission Deliverables & Requirements

According to the official **SupeRR Selector Season 2 Problem Statement (`Round 1 Problem Statement _ SupeRR Selector Season 2.docx(3).pdf`)**, the primary evaluated submission is a **Methodology Report document upload**, not a PowerPoint presentation.

### Deliverables Matrix

| Deliverable | Status | Format | Purpose & Scope |
|:---|:---:|:---:|:---|
| **Methodology Report** | **REQUIRED** | **PDF / Document** | Official submission document containing all 15 research sections: introduction, dataset description, processing, pressure calculation, batter/bowler analysis, ML approach, top-10 rankings, supporting tables/charts, validation, and strategic recommendations. |
| **PowerPoint Presentation** | *Optional* | Slides (.pptx / PDF) | Executive overview, not a required submission per the official competition brief. |
| **Interactive Dashboard** | *Bonus* | Web Application | Broadcast-themed Rajasthan Royals Streamlit dashboard (`app.py`) with dynamic competition filtering ("Combined", "IPL Only", "T20I Only"). |

> **Official Competition Brief:** *"Prepare a professional, research-style PDF report for submission. Include graphs, statistical tables, player comparisons and evidence supporting your rankings."*

### Required 15-Section Methodology Report Structure

The generated report ([`PRESSURE_X_METHODOLOGY_REPORT.pdf`](outputs/reports/PRESSURE_X_METHODOLOGY_REPORT.pdf)) strictly satisfies all 15 research sections:
1. **Title Page & Executive Summary**: Research problem, operational definition of pressure, key findings, and quick-reference leaderboards.
2. **Introduction & Motivation**: The fatal blind spots of traditional aggregate statistics and the need for situational analytics.
3. **Dataset Description & Provenance**: Complete breakdown of 200,999 deliveries across 883 matches, entity disambiguation, and canonical player resolution.
4. **Data Cleaning, Validation & Match State Reconstruction**: Causal sequence reconstruction, exact delivery accounting (192,445 legal vs 8,554 illegal), zero temporal leakage.
5. **Contextual Pressure Index (P_t) Formulation**: Mathematical formulation of delivery leverage ($L_t$), match closeness ($C_t$), asymmetric batting and bowling pressure bands, and justification of the 1.15 normalization denominator.
6. **Win Probability Model Architecture**: Dual LightGBM models with Platt scaling calibration, strict date-based holdout evaluation, and feature importances.
7. **Situational Expectation Models**: Baseline comparison against LightGBM regressors and classifiers for expected runs, dots, boundaries, and wickets.
8. **Player Performance Metrics**: Formulations of RAA, PRAA, RPA, PRPA, WPA, and per-100-ball normalizations.
9. **Empirical Bayes Shrinkage**: Mathematical justification of partial pooling ($K_{\text{bat}}=80$, $K_{\text{bowl}}=90$, $K_{\text{hp}}=30$) to stabilize small-sample variance.
10. **Official Top-10 Batters Ranking**: Full table with metrics, confidence intervals, sample sizes, and detailed analytical profiles.
11. **Official Top-10 Bowlers Ranking**: Full table with metrics, confidence intervals, sample sizes, and detailed analytical profiles.
12. **Competition-Specific Leaderboards**: Independent splits for Indian Premier League (IPL) and International T20s (T20I).
13. **Model Validation, Ablation & Convergence**: 5-fold OOF cross-validation, dynamic 4-tier ablation study, and 1,000-iteration Monte Carlo stability tests.
14. **Strategic Recommendations for Rajasthan Royals**: Auction targets, retention valuations, tactical phase deployment, and opponent counter-strategies.
15. **Limitations & Future Work**: Ball tracking coordinates, physical pitch wear, weather data, and multi-season decay weighting.

### Official Report Artifacts Generated
- **PDF Report (Submission Upload)**: [`outputs/reports/PRESSURE_X_METHODOLOGY_REPORT.pdf`](outputs/reports/PRESSURE_X_METHODOLOGY_REPORT.pdf) *(748 KB publication-grade formatted PDF with embedded high-resolution figures and data tables)*
- **Markdown Report**: [`outputs/reports/PRESSURE_X_METHODOLOGY_REPORT.md`](outputs/reports/PRESSURE_X_METHODOLOGY_REPORT.md)
- **HTML Report**: [`outputs/reports/PRESSURE_X_METHODOLOGY_REPORT.html`](outputs/reports/PRESSURE_X_METHODOLOGY_REPORT.html)
- **Official CSV Leaderboards**:
  - Combined: [`outputs/rankings/top_10_batters.csv`](outputs/rankings/top_10_batters.csv), [`outputs/rankings/top_10_bowlers.csv`](outputs/rankings/top_10_bowlers.csv)
  - IPL Only: [`outputs/rankings/top_10_batters_ipl.csv`](outputs/rankings/top_10_batters_ipl.csv), [`outputs/rankings/top_10_bowlers_ipl.csv`](outputs/rankings/top_10_bowlers_ipl.csv)
  - T20I Only: [`outputs/rankings/top_10_batters_t20i.csv`](outputs/rankings/top_10_batters_t20i.csv), [`outputs/rankings/top_10_bowlers_t20i.csv`](outputs/rankings/top_10_bowlers_t20i.csv)
- **Interactive Streamlit App**: Run `streamlit run app.py`

---

## 3. Core Methodological Enhancements Implemented

In response to the comprehensive audit directive, eight foundational architectural enhancements were executed across the codebase:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        8 CORE METHODOLOGICAL AUDIT ENHANCEMENTS                        │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ 1. Exact Delivery Logic  │ Reconciled all 200,999 records: 192,445 legal balls and     │
│                          │ 8,554 illegal balls (7,714 wides, 576 oversteps, 264 other).│
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2. Correct Denominators  │ Strike rate uses legal balls faced; bowling economy uses    │
│                          │ legal balls bowled. Fielding extras excluded from bowler.   │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3. Canonical Entities    │ Resolved 1,082 players by canonical IDs, eliminating name   │
│                          │ string collisions across national leagues and teams.        │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4. Out-of-Fold ML Models │ 5-fold chronological cross-validation generates out-of-    │
│                          │ sample expected runs, concessions, and WP predictions.      │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 5. Strict Date Holdout   │ Chronological split (671 train, 171 test matches) with zero │
│                          │ date overlap (max(train) < min(test)) across 842 matches.   │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 6. Asymmetric Pressure   │ Separate batting and bowling pressure formulas normalized   │
│                          │ by 1.15 to bound Pt in [0, 1] without artificial clipping.  │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 7. Dynamic Ablation      │ Replaced hardcoded numbers with dynamically measured metrics│
│                          │ proving incremental lift of context, ML, and Bayesian prior.│
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 8. 1,000-Iteration MC    │ Scaled match-clustered bootstrap to 1,000 runs, proving     │
│                          │ rank stability with |Delta P(Top 10)| < 2.5% convergence.   │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 4. Official Verified Rankings: Top 10 Leaderboards

All candidates were evaluated under strict qualification thresholds:
- Minimum **250 total competitive legal balls** faced / bowled across 2022–2026
- Minimum **8 competitive matches**
- Minimum **20 high-pressure legal balls** ($P_t > 0.666$)

### Table 7: Top 10 Combined T20 Batters

| Rank | Player | Matches | Runs | PRAA | BPI |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | **Suryakumar Yadav** | 139 | 4,595 | 895.3 | **158.9** |
| **2** | **Tim David** | 98 | 2,178 | 435.1 | **155.6** |
| **3** | **Jos Buttler** | 121 | 4,396 | 288.3 | **141.1** |
| **4** | **Heinrich Klaasen** | 72 | 2,363 | 342.7 | **131.9** |
| **5** | **Andre Russell** | 57 | 1,254 | 261.2 | **124.6** |
| **6** | **Glenn Maxwell** | 75 | 1,542 | 202.6 | **114.8** |
| **7** | **Namboori Tilak Varma** | 127 | 3,596 | 87.9 | **112.4** |
| **8** | **Donovan Ferreira** | 30 | 562 | 112.8 | **111.3** |
| **9** | **Ashutosh Sharma** | 28 | 577 | 78.5 | **110.7** |
| **10** | **Naman Dhir** | 46 | 992 | 200.5 | **110.0** |

#### Detailed Batting Metrics & Monte Carlo Stability
| Rank | Player Name | Matches | Balls | Actual Runs | Exp Runs | PRAA Total | Net WPA | P(Top 10) | Median Rank | 95% Rank CI | BPI Score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Suryakumar Yadav** | 139 | 2,850 | 4,595 | 3,935.6 | +895.3 | +4.95 | **93.6%** | 2.0 | [1 - 14] | **158.9** |
| **2** | **Tim David** | 98 | 1,293 | 2,178 | 1,879.5 | +435.1 | +4.37 | **91.2%** | 3.0 | [1 - 20] | **155.6** |
| **3** | **Jos Buttler** | 121 | 2,920 | 4,396 | 4,175.1 | +288.3 | +3.28 | **74.0%** | 5.0 | [1 - 26] | **141.1** |
| **4** | **Heinrich Klaasen** | 72 | 1,480 | 2,363 | 2,128.7 | +342.7 | +3.63 | **74.8%** | 6.0 | [1 - 26] | **131.9** |
| **5** | **Andre Russell** | 57 | 785 | 1,254 | 1,081.0 | +261.2 | +1.56 | **59.6%** | 8.0 | [1 - 32] | **124.6** |
| **6** | **Glenn Maxwell** | 75 | 1,004 | 1,542 | 1,391.8 | +202.9 | +2.38 | **46.4%** | 12.0 | [2 - 42] | **114.8** |
| **7** | **Namboori Tilak Varma** | 127 | 2,631 | 3,596 | 3,570.8 | +87.9 | +2.92 | **40.0%** | 14.0 | [1 - 45] | **112.4** |
| **8** | **Donovan Ferreira** | 30 | 354 | 562 | 488.6 | +112.8 | -0.18 | **40.0%** | 15.5 | [2 - 47] | **111.3** |
| **9** | **Ashutosh Sharma** | 28 | 369 | 577 | 529.2 | +78.5 | +1.35 | **43.2%** | 12.0 | [1 - 48] | **110.7** |
| **10** | **Naman Dhir** | 46 | 603 | 992 | 857.4 | +200.5 | +0.53 | **36.0%** | 14.0 | [2 - 42] | **110.0** |

---

### Top 10 Combined T20 Bowlers

| Rank | Player | Matches | Wickets | PRPA | BPI |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | **Jasprit Bumrah** | 75 | 85 | 747.1 | **153.3** |
| **2** | **Rashid Khan** | 122 | 153 | 378.2 | **136.0** |
| **3** | **Sunil Narine** | 67 | 66 | 502.7 | **130.4** |
| **4** | **Ehsan Khan** | 14 | 19 | 126.9 | **121.9** |
| **5** | **Josh Hazlewood** | 65 | 98 | 175.9 | **111.7** |
| **6** | **Nathan Ellis** | 43 | 58 | 229.7 | **109.8** |
| **7** | **Kuldeep Yadav** | 89 | 118 | 148.1 | **108.1** |
| **8** | **Varun Chakravarthy** | 95 | 135 | 159.4 | **106.0** |
| **9** | **Mustafizur Rahman** | 71 | 79 | 487.0 | **102.4** |
| **10** | **Bhuvneshwar Kumar** | 101 | 115 | 365.8 | **101.5** |

#### Detailed Bowling Metrics & Monte Carlo Stability
| Rank | Player Name | Matches | Balls | Actual Conc. | Exp Conc. | Wkts (WAA) | PRPA Total | Net WPA | P(Top 10) | Median Rank | 95% Rank CI | BPI Score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Jasprit Bumrah** | 75 | 1,780 | 2,069 | 2,591.8 | 85 (-10.7) | +747.1 | +2.82 | **98.8%** | 2.0 | [1 - 9] | **153.3** |
| **2** | **Rashid Khan** | 122 | 2,780 | 3,403 | 3,676.0 | 153 (+10.9) | +378.2 | +5.36 | **86.0%** | 3.0 | [1 - 28] | **136.0** |
| **3** | **Sunil Narine** | 67 | 1,556 | 1,754 | 2,116.9 | 66 (-9.2) | +502.7 | +2.18 | **84.8%** | 4.0 | [1 - 20] | **130.4** |
| **4** | **Ehsan Khan** | 14 | 331 | 384 | 467.5 | 19 (+0.2) | +126.9 | +1.23 | **64.0%** | 8.0 | [1 - 32] | **121.9** |
| **5** | **Josh Hazlewood** | 65 | 1,515 | 2,037 | 2,162.9 | 98 (+17.4) | +175.9 | +3.37 | **52.0%** | 10.0 | [1 - 44] | **111.7** |
| **6** | **Nathan Ellis** | 43 | 959 | 1,260 | 1,422.5 | 58 (+2.8) | +229.7 | +1.74 | **41.6%** | 13.0 | [2 - 39] | **109.8** |
| **7** | **Kuldeep Yadav** | 89 | 1,965 | 2,557 | 2,659.3 | 118 (+20.2) | +148.1 | +2.76 | **38.8%** | 14.0 | [2 - 44] | **108.1** |
| **8** | **Varun Chakravarthy** | 95 | 2,182 | 2,841 | 2,961.6 | 135 (+26.1) | +159.4 | +3.11 | **46.0%** | 12.0 | [2 - 47] | **106.0** |
| **9** | **Mustafizur Rahman** | 71 | 1,646 | 2,050 | 2,386.8 | 79 (-12.2) | +487.0 | +1.29 | **36.4%** | 16.0 | [2 - 44] | **102.4** |
| **10** | **Bhuvneshwar Kumar** | 101 | 2,314 | 3,069 | 3,327.8 | 115 (-7.8) | +365.8 | +1.93 | **30.8%** | 15.0 | [2 - 47] | **101.5** |

### Official Top 5 Batters — IPL Only

| Rank | Player Name | Matches | Legal Balls | Total Deliv | Actual Runs | Exp Runs | PRAA Total | BPI Score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Suryakumar Yadav** | 64 | 1,353 | 1,391 | 2,240 | 1936.7 | +400.5 | **137.8** |
| **2** | **Tim David** | 58 | 639 | 694 | 1,150 | 1043.6 | +162.5 | **129.2** |
| **3** | **Rajat Patidar** | 48 | 916 | 970 | 1,541 | 1376.5 | +238.0 | **128.7** |
| **4** | **Heinrich Klaasen** | 54 | 1,197 | 1,262 | 2,038 | 1836.9 | +295.7 | **124.9** |
| **5** | **Andre Russell** | 45 | 562 | 614 | 951 | 843.3 | +166.3 | **117.2** |

### Official Top 5 Bowlers — IPL Only

| Rank | Player Name | Matches | Legal Balls | Total Deliv | Actual Conc. | Exp Conc. | PRPA Total | BPI Score |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Jasprit Bumrah** | 52 | 1,209 | 1,262 | 1,445 | 1879.9 | +640.8 | **146.8** |
| **2** | **Sunil Narine** | 67 | 1,533 | 1,556 | 1,754 | 2115.5 | +506.2 | **133.0** |
| **3** | **Matheesha Pathirana** | 33 | 710 | 797 | 1,025 | 1243.0 | +308.9 | **120.2** |
| **4** | **Bhuvneshwar Kumar** | 74 | 1,651 | 1,715 | 2,325 | 2504.3 | +272.1 | **116.9** |
| **5** | **Lungi Ngidi** | 13 | 296 | 303 | 421 | 476.8 | +83.5 | **95.7** |

---

## 5. Mathematical Formulations

### 5.1 Delivery Leverage ($L_t$)
Delivery leverage measures the absolute shift in match win probability swung by delivery $t$:
$$L_t = |WP_{\text{post}, t} - WP_{\text{pre}, t}|$$

### 5.2 Match Closeness ($C_t$)
Peaks at 1.0 when win probability is 50%, decaying linearly to 0.0 when the outcome is virtually certain:
$$C_t = \text{clip}\left(1.0 - 2.0 \cdot |WP_{\text{pre}, t} - 0.5|, 0.0, 1.0\right)$$

### 5.3 Asymmetric Contextual Pressure
- **1st Innings Batting**:
  $$P_{\text{bat}, t} = \frac{0.30 \cdot W_{\text{rem\_stress}} + 0.30 \cdot CRR_{\text{drag}} + 0.40 \cdot Target_{\text{pace}}}{1.15} \times w_{\text{phase}}$$
- **2nd Innings (Chase) Batting**:
  $$P_{\text{bat}, t} = \frac{0.40 \cdot C_t + 0.35 \cdot RRR_{\text{stress}} + 0.25 \cdot W_{\text{rem\_stress}}}{1.15} \times w_{\text{phase}}$$
- **1st Innings Bowling**:
  $$P_{\text{bowl}, t} = \frac{0.35 \cdot \text{clip}(CRR / 10.0, 0, 1) + 0.35 \cdot W_{\text{lost\_stress}} + 0.30 \cdot I(\text{Death})}{1.15} \times w_{\text{phase}}$$
- **2nd Innings (Defense) Bowling**:
  $$P_{\text{bowl}, t} = \frac{0.45 \cdot C_t + 0.30 \cdot (1 - RRR_{\text{stress}}) \cdot Opp_{\text{resource}} + 0.25 \cdot I(\text{Death})}{1.15} \times w_{\text{phase}}$$

*The 1.15 denominator normalizes the metric into [0.0, 1.0] by dividing by the maximum phase multiplier ($w_{\text{death}} = 1.15$). High Pressure is standardized as $P_t > 0.666$.*

### 5.4 Situational Expectations & Attribution
- **Runs Above Average (RAA)**: $RAA_t = ActualRuns_t - \mathbb{E}[Runs_t]$
- **Pressure-Weighted RAA (PRAA)**: $PRAA_t = RAA_t \times (1 + P_{\text{bat}, t})$
- **Runs Prevented Above Average (RPA)**: $RPA_t = \mathbb{E}[RunsConceded_t] - BowlerConceded_t$
- **Pressure-Weighted RPA (PRPA)**: $PRPA_t = RPA_t \times (1 + P_{\text{bowl}, t})$
- **Win Probability Added (WPA)**: $WPA_{\text{batter}} = WP_{\text{post}} - WP_{\text{pre}}$, $WPA_{\text{bowler}} = WP_{\text{pre}} - WP_{\text{post}}$

### 5.5 Empirical Bayes Partial Pooling
To protect against small-sample noise, observed rates are shrunk toward empirical population priors:
$$\widehat{\theta}_i = \frac{N_i}{N_i + K} \cdot \bar{x}_i + \frac{K}{N_i + K} \cdot \mu_{\text{prior}}$$
with $K_{\text{bat}} = 80.0$, $K_{\text{bowl}} = 90.0$, and $K_{\text{hp}} = 30.0$ legal balls.

---

## 6. Model Validation & Reliability Results

### 6.1 Baseline Lift Comparison (5-Fold Out-of-Fold Cross-Validation)
- **Batting Runs Expectation**: LightGBM OOF MAE = **1.2847** vs Historical Phase Baseline MAE = **1.3046** (**+1.52% lift**, RMSE = 1.7141 vs 1.7301, **+0.93% lift**).
- **Bowling Conceded Expectation**: LightGBM OOF MAE = **1.2721** vs Historical Phase Baseline MAE = **1.2858** (**+1.06% lift**, RMSE = 1.7011 vs 1.7135, **+0.72% lift**).

### 6.2 Strict Chronological Holdout Validation
- Partitioned strictly by match date across 842 completed matches (80% train / 20% test).
- **671 Training Matches** (159,579 deliveries) and **171 Unseen Test Matches** (39,478 deliveries) with **zero match overlap** and `max(train_dates) < min(test_dates)`.
- **Innings 2 (Chases) Holdout**: $\text{ROC-AUC} = \mathbf{0.8763}$, $\text{Brier Score} = \mathbf{0.1482}$, $\text{Log-Loss} = \mathbf{0.4496}$.
- **Combined Holdout**: $\text{ROC-AUC} = \mathbf{0.7661}$, $\text{Brier Score} = \mathbf{0.1973}$, $\text{Log-Loss} = \mathbf{0.5708}$.

### 6.3 Monte Carlo Convergence (1,000 Iterations)
Bootstrap convergence testing evaluated across 250, 500, and 1,000 iterations:
- Mean absolute change in $P(\text{Top 10})$ between 500 and 1,000 iterations was **1.24%** (batting) and **1.71%** (bowling), well beneath the 2.5% threshold.
- Median simulated rank was **100% invariant** between 500 and 1,000 iterations for all top-10 candidates.

---

## 7. Pipeline Execution & Reproduction Commands

### Automated Test Suite (33/33 Passing)
```bash
python -m pytest tests/ -v
```

### Complete End-to-End Pipeline
```bash
# Phase 1: Data Ingestion, Referential Integrity Audit & State Reconstruction
python main.py --phase 1

# Phase 2: ML Expectation Models (LightGBM OOF), Bayesian Shrinkage & Ratings
python main.py --phase 2

# Phase 3: 1,000-Iteration Monte Carlo Stability, Dynamic Ablation & Diagnostic Charts
python main.py --phase 3

# Generate Official 15-Section Methodology Report & Compile PDF
python scripts/generate_methodology_report.py
```

### Launch Streamlit Intelligence Dashboard
```bash
streamlit run app.py
```

---

## 8. Directory Structure

```
rr analysis/
├── config/
│   ├── data_config.yaml          # Raw paths, cricket rules, expected schemas
│   ├── pressure_config.yaml      # Pressure bands, leverage & win probability parameters
│   ├── batting_config.yaml       # Batting model specifications & qualification thresholds
│   └── bowling_config.yaml       # Bowling model specifications & attribution rules
├── data/
│   ├── raw/                      # Original CSV files (strictly untouched & immutable)
│   ├── interim/                  # Intermediate checkpoints
│   ├── processed/
│   │   └── deliveries_enriched.parquet   # Phase 1 reconstructed state (200,999 rows × 52 cols)
│   └── features/
│       └── deliveries_modeled.parquet    # Phase 4 modeled features (200,999 rows × 70 cols)
├── src/
│   ├── ingestion/loader.py               # Raw data ingestion & schema discovery
│   ├── validation/validator.py           # Data quality & referential integrity audit
│   ├── reconstruction/
│   │   ├── entity_resolver.py            # Cross-entity linking & competition categorization
│   │   └── match_reconstructor.py        # Running scores, legal balls & RRR reconstruction
│   ├── match_state/state_engine.py       # S_t (Pre) -> S_{t+1} (Post) transition dynamics
│   ├── win_probability/wp_model.py       # Calibrated probabilistic Win Probability models
│   ├── pressure/pressure_engine.py       # Delivery leverage, WPA & Contextual Pressure indices
│   ├── features/feature_builder.py       # Causal, zero-leakage ML feature pipeline
│   ├── batting/
│   │   ├── expected_runs.py              # LightGBM batting expectation models & OOF validation
│   │   └── batting_rating.py             # RAA, PRAA, High-P RAA & Batting Pressure Index
│   ├── bowling/
│   │   ├── expected_conceded.py          # LightGBM bowling expectation & wicket models
│   │   └── bowling_rating.py             # RPA, PRPA, High-P RPA & Bowling Pressure Index
│   ├── bayesian/shrinkage.py             # Empirical Bayes partial pooling & uncertainty
│   ├── matchup/matchup_engine.py         # Matchup archetypes & Venue Scoring Indices
│   ├── simulation/monte_carlo.py         # 1,000-iteration Monte Carlo stability engine
│   ├── evaluation/
│   │   ├── split_utils.py                # Temporal holdout & k-fold partitioning
│   │   └── ablation_validator.py         # Dynamic measured ablation study
│   ├── explainability/visualizer.py      # Research charts & diagnostic figure generator
│   └── ranking/ranking_engine.py         # Official Top-10 leaderboards & competition splits
├── outputs/
│   ├── rankings/
│   │   ├── top_10_batters.csv            # Official Top 10 Batters (Combined)
│   │   ├── top_10_bowlers.csv            # Official Top 10 Bowlers (Combined)
│   │   ├── top_10_batters_ipl.csv        # Top 10 Batters (IPL Only)
│   │   ├── top_10_bowlers_ipl.csv        # Top 10 Bowlers (IPL Only)
│   │   ├── top_10_batters_t20i.csv       # Top 10 Batters (T20I Only)
│   │   ├── top_10_bowlers_t20i.csv       # Top 10 Bowlers (T20I Only)
│   │   ├── all_batters_ranked.csv        # Complete qualified batting rankings
│   │   └── all_bowlers_ranked.csv        # Complete qualified bowling rankings
│   ├── charts/
│   │   ├── win_prob_calibration.png      # Win Probability sigmoid reliability curve
│   │   ├── pressure_distribution.png     # Pressure distribution by innings phase
│   │   ├── top10_batters_matrix.png      # PRAA vs High-Pressure RAA bubble matrix
│   │   └── top10_bowlers_matrix.png      # PRPA vs High-Pressure RPA bubble matrix
│   └── reports/
│       ├── PRESSURE_X_METHODOLOGY_REPORT.pdf  # Official 15-section publication PDF (748 KB)
│       ├── PRESSURE_X_METHODOLOGY_REPORT.md   # Markdown version of methodology report
│       ├── PRESSURE_X_METHODOLOGY_REPORT.html # Publication HTML source for PDF compilation
│       ├── ablation_comparison.csv       # Component ablation lift statistics
│       ├── player_identity_audit.txt     # Canonical entity resolution audit (1,082 players)
│       ├── monte_carlo_convergence_report.txt # Convergence verification across 250, 500, 1000 runs
│       └── wp_chronological_validation.json # Out-of-sample holdout metrics
├── scripts/
│   ├── generate_methodology_report.py    # 15-section report compilation & PDF build script
│   ├── generate_player_audit.py          # Entity resolution validation script
│   └── verify_mc_convergence.py          # Monte Carlo convergence evaluation script
├── tests/
│   ├── test_phase1.py                    # Unit tests for Phase 1 cricket rules & states (15 tests)
│   ├── test_phase2_3.py                  # Unit tests for transitions, WP, bands & shrinkage (14 tests)
│   └── test_audit_integrity.py           # Unit tests for exact accounting & temporal splits (4 tests)
├── app.py                                # Executive Rajasthan Royals Streamlit Intelligence Dashboard
├── main.py                               # Unified pipeline orchestrator
├── requirements.txt                      # Frozen Python dependencies
└── README.md                             # Project documentation
```
