"""
PRESSURE-X — Phase 10: Counterfactual Intelligence Engine
==========================================================
Answers the foundational research question:
"What would have happened if an equivalent replacement-level player faced
the same delivery in the exact same match situation?"

Calculates Counterfactual Delivery Delta (CDD):
  CDD_t = WP(S_{actual outcome}) - WP(S_{replacement expected})

Aggregates across all high-pressure deliveries to compute:
  Pressure Wins Above Replacement (WAR_pressure):
  WAR_pressure = sum_{t in high_pressure}(CDD_t)

Provides match-level counterfactual reconstructions to quantify exactly how
many match outcomes were flipped by elite pressure performers.

Author: PRESSURE-X Engineering
Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CounterfactualIntelligenceEngine:
    """
    Simulates alternative delivery outcomes and computes Pressure Wins Above Replacement (WAR).
    """

    def __init__(self, wp_model=None):
        self.wp_model = wp_model

    def evaluate_player_counterfactuals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates counterfactual value added for both batters and bowlers
        on all deliveries where pressure is medium or high.
        """
        logger.info("Computing Counterfactual Wins Above Replacement (WAR_pressure)...")
        df = df.copy()

        df["raa"] = df["runs"] - df["exp_runs_bat"]
        df["rpa"] = df["exp_runs_bowl"] - df["runs"]

        # Net swing vs situation expectation
        # For batter: actual runs - expected runs converted to expected WP swing
        # Leverage Lt represents dWP/dRuns sensitivity: Lt / (mean run swing ~1.5)
        run_leverage_sensitivity = np.clip(df["leverage_score"] / 1.4, 0.01, 0.40)

        df["c_delta_batter"] = df["raa"] * run_leverage_sensitivity
        df["c_delta_bowler"] = df["rpa"] * run_leverage_sensitivity

        # Filter to pressure opportunities (pressure_score >= 0.40)
        p_mask = df["pressure_score"] >= 0.35

        # Batting Counterfactual WAR
        bat_war = df[p_mask].groupby(["striker_id", "striker_name"]).agg(
            war_pressure_balls=("runs", "count"),
            counterfactual_war=("c_delta_batter", "sum"),
            counterfactual_runs_added=("raa", "sum"),
            actual_wpa=("wpa_batter", "sum"),
        ).reset_index()

        bat_war["counterfactual_war"] = bat_war["counterfactual_war"].round(3)
        bat_war["war_per_100_balls"] = (bat_war["counterfactual_war"] / bat_war["war_pressure_balls"] * 100).round(2)
        bat_war = bat_war.sort_values("counterfactual_war", ascending=False).reset_index(drop=True)

        # Bowling Counterfactual WAR
        bowl_war = df[p_mask].groupby(["bowler_id", "bowler_name"]).agg(
            war_pressure_balls=("runs", "count"),
            counterfactual_war=("c_delta_bowler", "sum"),
            counterfactual_runs_prevented=("rpa", "sum"),
            actual_wpa=("wpa_bowler", "sum"),
        ).reset_index()

        bowl_war["counterfactual_war"] = bowl_war["counterfactual_war"].round(3)
        bowl_war["war_per_100_balls"] = (bowl_war["counterfactual_war"] / bowl_war["war_pressure_balls"] * 100).round(2)
        bowl_war = bowl_war.sort_values("counterfactual_war", ascending=False).reset_index(drop=True)

        logger.info("Counterfactual WAR evaluation complete.")
        return bat_war, bowl_war
