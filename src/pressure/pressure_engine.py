"""
PRESSURE-X — Phase 3: Pressure Intelligence Engine
===================================================
Develops scientifically defensible, multi-faceted pressure and leverage metrics:

1. Delivery Leverage (L_t):
   L_t = |WP_{t+1} - WP_t|
   Measures observed win probability change caused by the delivery outcome.

2. Win Probability Added (WPA):
   WPA_batter = WP_{t+1} - WP_t
   WPA_bowler = -(WP_{t+1} - WP_t) = WP_bowling_team_{t+1} - WP_bowling_team_t

3. Ex-Ante Match Closeness (State Tension):
   C_t = 1.0 - 2.0 * |WP_t - 0.5|  (peaks at 1.0 when WP=0.5, 0.0 when decided)

4. Contextual Batting Pressure Index (P_bat):
   Combines:
     - Match closeness / swing volatility
     - Chase stress: Required run rate vs benchmark (or run acceleration in 1st inn)
     - Resource scarcity: Low wickets or balls remaining
     - Phase multiplier: Death overs amplifies pressure

5. Contextual Bowling Pressure Index (P_bowl):
   Combines:
     - Match closeness
     - Opposition resource depth (opponents have high wickets in hand in death)
     - Margin for error: Low buffer of defendable runs

6. Pressure Bands:
   Bands deliveries into 'low' ([0, 0.33]), 'medium' ([0.33, 0.66]), 'high' ([0.66, 1.0])

Author: PRESSURE-X Engineering
Version: 2.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from win_probability.wp_model import WinProbabilityModel

logger = logging.getLogger(__name__)


class PressureIntelligenceEngine:
    """
    Engine to compute Leverage, WPA, Batting Pressure, Bowling Pressure,
    and categorize deliveries into calibrated pressure bands.
    """

    def __init__(self, config_path: str | Path = "config/pressure_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)

        self.wp_model = WinProbabilityModel()

    def fit_wp_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Fits the underlying win probability models."""
        return self.wp_model.train(df)

    def compute_pressure_metrics(self, df: pd.DataFrame, use_oof: bool = False) -> pd.DataFrame:
        """
        Enriches DataFrame with:
          - wp_pre: Win probability before delivery
          - wp_post: Win probability after delivery
          - leverage_score: |wp_post - wp_pre|
          - wpa_batter: wp_post - wp_pre
          - wpa_bowler: wp_pre - wp_post
          - match_closeness: 1.0 - 2.0 * |wp_pre - 0.5|
          - batting_pressure_score: Composite [0, 1]
          - bowling_pressure_score: Composite [0, 1]
          - batting_pressure_band: 'low' | 'medium' | 'high'
          - bowling_pressure_band: 'low' | 'medium' | 'high'
          - pressure_band: 'low' | 'medium' | 'high' (overall composite)
        """
        logger.info("Computing win probabilities, leverage, and pressure scores for %d deliveries...", len(df))
        df = df.copy()

        if use_oof and "match_date" in df.columns and df["match_id"].nunique() >= 5:
            logger.info("Using Out-of-Fold Chronological WP Predictions for leakage-free ratings...")
            wp_pre, wp_post, _ = self.wp_model.generate_oof_predictions(df)
        else:
            # 1. Compute Pre-delivery WP (S_t)
            wp_pre = self.wp_model.predict_state_wp(
                innings=df["innings"].to_numpy(),
                balls_remaining=df["balls_remaining_pre"].to_numpy(),
                wickets_remaining=df["wickets_remaining_pre"].to_numpy(),
                innings_runs=df["innings_runs_pre"].to_numpy(),
                crr=df["crr_pre"].to_numpy(),
                phases=df["innings_phase"],
                runs_required=df["runs_required_pre"].to_numpy() if "runs_required_pre" in df.columns else None,
                rrr=df["rrr_pre"].to_numpy() if "rrr_pre" in df.columns else None,
                target=df["target"].to_numpy() if "target" in df.columns else None,
            )

            # 2. Compute Post-delivery WP (S_{t+1})
            wp_post = self.wp_model.predict_state_wp(
                innings=df["innings"].to_numpy(),
                balls_remaining=df["balls_remaining_post"].to_numpy(),
                wickets_remaining=df["wickets_remaining_post"].to_numpy(),
                innings_runs=df["innings_runs_post"].to_numpy(),
                crr=df["crr_post"].to_numpy(),
                phases=df["innings_phase"],
                runs_required=df["runs_required_post"].to_numpy() if "runs_required_post" in df.columns else None,
                rrr=df["rrr_post"].to_numpy() if "rrr_post" in df.columns else None,
                target=df["target"].to_numpy() if "target" in df.columns else None,
            )

        df["wp_pre"] = wp_pre
        df["wp_post"] = wp_post

        # 3. Leverage and WPA
        df["leverage_score"] = np.abs(wp_post - wp_pre)
        df["wpa_batter"] = wp_post - wp_pre
        df["wpa_bowler"] = wp_pre - wp_post

        # 4. Match Closeness (peaks at 1.0 when WP=0.5, 0.0 when match is virtually decided)
        df["match_closeness"] = np.clip(1.0 - 2.0 * np.abs(wp_pre - 0.5), 0.0, 1.0)

        # 5. Resource Depletion & Urgency
        # Normalized wicket pressure: 0 when 10 wickets left, 1 when 1 wicket left
        wicket_stress = np.clip((10.0 - df["wickets_remaining_pre"]) / 9.0, 0.0, 1.0)

        # Innings phase weight
        phase_weights = {"powerplay": 0.70, "middle": 0.85, "death": 1.15}
        p_weights = df["innings_phase"].map(phase_weights).fillna(0.85).to_numpy()

        # RRR urgency in chases
        rrr_vals = df["rrr_pre"].fillna(8.0).to_numpy()
        # Normal T20 par RRR is ~8.0. Normalized stress: 0 at <=6.0, 1.0 at >=14.0
        rrr_stress = np.clip((rrr_vals - 6.0) / 8.0, 0.0, 1.0)

        # 1st innings pacing urgency (death overs or high wicket loss)
        balls_used = 120.0 - df["balls_remaining_pre"].to_numpy()
        inn1_urgency = np.clip((balls_used / 120.0) * 0.5 + wicket_stress * 0.5, 0.0, 1.0)

        # ---------------- Contextual Batting Pressure (P_bat) ----------------
        # When chasing: high closeness + high RRR + low wickets
        # When batting 1st: high closeness + phase urgency + low wickets
        is_inn2 = (df["innings"] == 2).to_numpy()

        batting_pressure_raw = np.where(
            is_inn2,
            0.40 * df["match_closeness"] + 0.35 * rrr_stress + 0.25 * wicket_stress,
            0.40 * df["match_closeness"] + 0.35 * inn1_urgency + 0.25 * wicket_stress,
        ) * p_weights

        df["batting_pressure_score"] = np.clip(batting_pressure_raw / 1.15, 0.0, 1.0)

        # ---------------- Contextual Bowling Pressure (P_bowl) ----------------
        # Bowling pressure is high when:
        # Match is close, opposition has batting resources remaining, or bowler is bowling death overs defending small margin
        opp_resource = np.clip(df["wickets_remaining_pre"] / 10.0, 0.0, 1.0)
        bowling_pressure_raw = np.where(
            is_inn2,
            0.45 * df["match_closeness"] + 0.30 * (1.0 - rrr_stress) * opp_resource + 0.25 * (p_weights >= 1.0).astype(float),
            0.45 * df["match_closeness"] + 0.35 * opp_resource + 0.20 * (p_weights >= 1.0).astype(float),
        ) * p_weights

        df["bowling_pressure_score"] = np.clip(bowling_pressure_raw / 1.15, 0.0, 1.0)

        # Overall composite delivery pressure score (for generic filtering)
        df["pressure_score"] = 0.5 * df["batting_pressure_score"] + 0.5 * df["bowling_pressure_score"]

        # Pressure Bands: Separate batting and bowling bands, plus overall composite
        df["batting_pressure_band"] = pd.cut(
            df["batting_pressure_score"],
            bins=[-0.01, 0.333, 0.666, 1.01],
            labels=["low", "medium", "high"],
        ).astype(str)

        df["bowling_pressure_band"] = pd.cut(
            df["bowling_pressure_score"],
            bins=[-0.01, 0.333, 0.666, 1.01],
            labels=["low", "medium", "high"],
        ).astype(str)

        df["pressure_band"] = pd.cut(
            df["pressure_score"],
            bins=[-0.01, 0.333, 0.666, 1.01],
            labels=["low", "medium", "high"],
        ).astype(str)

        logger.info(
            "Pressure computation complete.\nBatting band dist: %s\nBowling band dist: %s\nOverall band dist: %s",
            df["batting_pressure_band"].value_counts(normalize=True).round(3).to_dict(),
            df["bowling_pressure_band"].value_counts(normalize=True).round(3).to_dict(),
            df["pressure_band"].value_counts(normalize=True).round(3).to_dict(),
        )
        return df
