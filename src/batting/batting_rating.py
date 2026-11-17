"""
PRESSURE-X — Phase 5: Advanced Batting Intelligence Engine
===========================================================
Calculates multidimensional batting pressure performance:
  - Runs Above Expectation (RAA): Actual Runs - Expected Runs
  - Pressure-Weighted RAA (PRAA): RAA weighted by situational pressure multiplier
  - High-Pressure RAA: Performance specifically in high-pressure deliveries
  - Win Probability Added (WPA): Total win probability swung by striker
  - Boundaries Above Expectation (BAA): Actual Boundaries - Expected Boundaries
  - Dot Balls Avoided: Expected Dots - Actual Dots
  - Dismissal Risk Management: Expected Dismissals - Actual Dismissals
  - Match-to-Match Consistency & Reliability Index

Applies Bayesian shrinkage to ensure robust, evidence-backed rankings.

Author: PRESSURE-X Engineering
Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bayesian.shrinkage import BayesianShrinkageEngine

logger = logging.getLogger(__name__)


class BattingIntelligenceEngine:
    """
    Evaluates and ranks batters based on context-aware pressure performance.
    """

    def __init__(self, min_balls: int = 250, min_matches: int = 8, min_hp_balls: int = 20):
        self.min_balls = min_balls
        self.min_matches = min_matches
        self.min_hp_balls = min_hp_balls
        self.bayesian = BayesianShrinkageEngine(prior_balls_batting=80.0)

    def evaluate_batters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes deliveries with expected batting metrics and computes player ratings.
        """
        logger.info("Computing Batting Pressure Performance Index for all batters...")

        # Per-delivery metrics
        df = df.copy()
        if "is_legal_delivery" in df.columns:
            df["legal_ball"] = df["is_legal_delivery"].astype(int)
        else:
            df["legal_ball"] = 1

        df["raa"] = df["runs"] - df["exp_runs_bat"]
        df["praa"] = df["raa"] * (1.0 + df["batting_pressure_score"])
        df["is_boundary"] = (df["runs"] >= 4).astype(int)
        df["boundary_above_exp"] = df["is_boundary"] - df["prob_boundary_bat"]
        df["is_dot"] = (df["runs"] == 0).astype(int)
        df["dot_avoided"] = df["prob_dot_bat"] - df["is_dot"]
        df["wicket_avoided"] = df["prob_wicket_bat"] - df["_wicket"]

        # Aggregate by striker
        agg_dict = {
            "striker_name": "first",
            "match_id": "nunique",
            "legal_ball": "sum",
            "runs": ["count", "sum"],
            "exp_runs_bat": "sum",
            "raa": "sum",
            "praa": "sum",
            "wpa_batter": "sum",
            "boundary_above_exp": "sum",
            "dot_avoided": "sum",
            "wicket_avoided": "sum",
        }

        grouped = df.groupby("striker_id").agg(agg_dict)
        grouped.columns = [
            "player_name",
            "matches",
            "balls_faced",  # Strictly legal balls faced (denominator)
            "total_deliveries",
            "actual_runs",
            "expected_runs",
            "raa_total",
            "praa_total",
            "wpa_total",
            "boundary_above_exp",
            "dot_avoided",
            "wicket_avoided",
        ]

        # Filter by qualification criteria (legal balls faced >= min_balls)
        qualified = grouped[
            (grouped["balls_faced"] >= self.min_balls) & (grouped["matches"] >= self.min_matches)
        ].copy()

        # Rates per 100 legal balls faced
        qualified["strike_rate"] = (qualified["actual_runs"] / np.maximum(qualified["balls_faced"], 1)) * 100.0
        qualified["raa_per_100"] = (qualified["raa_total"] / qualified["balls_faced"]) * 100.0
        qualified["praa_per_100"] = (qualified["praa_total"] / qualified["balls_faced"]) * 100.0
        qualified["wpa_per_100"] = (qualified["wpa_total"] / qualified["balls_faced"]) * 100.0
        qualified["boundary_diff_per_100"] = (qualified["boundary_above_exp"] / qualified["balls_faced"]) * 100.0

        # High pressure slice with Bayesian shrinkage
        band_col = "batting_pressure_band" if "batting_pressure_band" in df.columns else "pressure_band"
        high_press = df[df[band_col] == "high"]
        hp_grp = high_press.groupby("striker_id").agg(
            hp_balls=("legal_ball", "sum"),
            hp_deliveries=("runs", "count"),
            hp_raa=("raa", "sum"),
            hp_wpa=("wpa_batter", "sum"),
        )
        qualified = qualified.join(hp_grp, how="left").fillna({"hp_balls": 0, "hp_deliveries": 0, "hp_raa": 0.0, "hp_wpa": 0.0})

        # Empirical Bayes shrinkage on High-Pressure RAA (Prior weight = 30.0 balls)
        hp_n = qualified["hp_balls"].to_numpy()
        hp_raw_rate = np.where(hp_n > 0, (qualified["hp_raa"].to_numpy() / np.maximum(hp_n, 1)) * 100.0, 0.0)
        shrunk_hp_rate = (hp_n / (hp_n + 30.0)) * hp_raw_rate
        qualified["raw_hp_raa_per_100"] = hp_raw_rate
        qualified["hp_raa_per_100"] = shrunk_hp_rate

        # Match-level consistency (standard deviation of match RAA)
        match_level = df.groupby(["striker_id", "match_id"])["raa"].sum().reset_index()
        match_std = match_level.groupby("striker_id")["raa"].std().fillna(10.0)
        qualified["match_std"] = match_std
        qualified["reliability_index"] = np.clip(1.0 / (1.0 + (qualified["match_std"] / 12.0)), 0.2, 1.0)

        # Bayesian Shrinkage on PRAA (Prior weight = 80.0 balls)
        pop_mean_praa = float(qualified["praa_per_100"].mean())
        shrunk_praa, se_praa, ci_lower = self.bayesian.shrink_series(
            observed=qualified["praa_per_100"],
            volumes=qualified["balls_faced"],
            prior_weight=80.0,
            population_mean=pop_mean_praa,
        )
        qualified["shrunk_praa_per_100"] = shrunk_praa
        qualified["rating_se"] = se_praa
        qualified["rating_ci_lower"] = ci_lower

        # Composite Batting Pressure Index (BPI)
        # Combines Shrunk Overall PRAA, Shrunk High-Pressure RAA, Cumulative WPA, and Reliability
        qualified["batting_pressure_index"] = (
            50.0
            + 1.0 * qualified["shrunk_praa_per_100"]
            + 0.8 * qualified["hp_raa_per_100"]
            + 12.0 * qualified["wpa_total"]
            + 5.0 * (qualified["reliability_index"] - 0.5)
        ).round(2)

        # Filter by high pressure volume qualification
        if self.min_hp_balls > 0:
            qualified = qualified[qualified["hp_balls"] >= self.min_hp_balls].copy()

        # Sort and rank
        ranked = qualified.sort_values("batting_pressure_index", ascending=False).reset_index()
        ranked["rank"] = np.arange(1, len(ranked) + 1)

        logger.info("Batting evaluation complete. %d batters qualified.", len(ranked))
        return ranked
