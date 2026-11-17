"""
PRESSURE-X — Phase 6: Advanced Bowling Intelligence Engine
===========================================================
Calculates multidimensional bowling pressure performance:
  - Runs Prevented Above Expectation (RPA): Expected Conceded - Actual Conceded
  - Pressure-Weighted RPA (PRPA): RPA weighted by situational bowling pressure
  - High-Pressure RPA: Performance in high-pressure deliveries
  - Bowler Wickets Above Expectation (WAA): Actual Bowler Wickets - Expected Wickets
  - Dot Balls Generated: Actual Dots - Expected Dots
  - Boundary Prevention: Expected Boundaries - Actual Boundaries Conceded
  - Win Probability Added (WPA_bowler): Total win probability swung by bowler
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
from bowling.expected_conceded import BOWLER_WICKET_TYPES, ExpectedBowlingModel

logger = logging.getLogger(__name__)


class BowlingIntelligenceEngine:
    """
    Evaluates and ranks bowlers based on context-aware pressure performance.
    """

    def __init__(self, min_balls: int = 250, min_matches: int = 8, min_hp_balls: int = 20):
        self.min_balls = min_balls
        self.min_matches = min_matches
        self.min_hp_balls = min_hp_balls
        self.bayesian = BayesianShrinkageEngine(prior_balls_bowling=90.0)

    def evaluate_bowlers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes deliveries with expected bowling metrics and computes player ratings.
        """
        logger.info("Computing Bowling Pressure Performance Index for all bowlers...")

        df = df.copy()
        if "is_legal_delivery" in df.columns:
            df["legal_ball"] = df["is_legal_delivery"].astype(int)
        else:
            df["legal_ball"] = 1

        # Bowler charged runs & credited wickets
        conceded, bowler_wicket = ExpectedBowlingModel.calculate_bowler_charge(df)
        df["conceded_actual"] = conceded
        df["bowler_wicket_actual"] = bowler_wicket

        # Deliveries for bowler (exclude byes/legbyes from ball count if illegal, but count all bowled)
        df["rpa"] = df["exp_runs_bowl"] - df["conceded_actual"]
        df["prpa"] = df["rpa"] * (1.0 + df["bowling_pressure_score"])

        df["is_bowler_dot"] = (df["conceded_actual"] == 0).astype(int)
        df["dot_gen_above_exp"] = df["is_bowler_dot"] - df["prob_dot_bowl"]

        df["is_bowler_boundary"] = (df["conceded_actual"] >= 4).astype(int)
        df["boundary_prevented"] = df["prob_boundary_bowl"] - df["is_bowler_boundary"]

        df["wicket_above_exp"] = df["bowler_wicket_actual"] - df["prob_wicket_bowl"]

        # Aggregate by bowler
        agg_dict = {
            "bowler_name": "first",
            "match_id": "nunique",
            "legal_ball": "sum",
            "conceded_actual": ["count", "sum"],
            "exp_runs_bowl": "sum",
            "bowler_wicket_actual": "sum",
            "rpa": "sum",
            "prpa": "sum",
            "wpa_bowler": "sum",
            "wicket_above_exp": "sum",
            "dot_gen_above_exp": "sum",
            "boundary_prevented": "sum",
        }

        grouped = df.groupby("bowler_id").agg(agg_dict)
        grouped.columns = [
            "player_name",
            "matches",
            "balls_bowled",  # Strictly legal balls bowled (denominator)
            "total_deliveries",
            "actual_conceded",
            "expected_conceded",
            "actual_wickets",
            "rpa_total",
            "prpa_total",
            "wpa_total",
            "wicket_above_exp",
            "dot_gen_above_exp",
            "boundary_prevented",
        ]

        # Filter qualified (legal balls bowled >= min_balls)
        qualified = grouped[
            (grouped["balls_bowled"] >= self.min_balls) & (grouped["matches"] >= self.min_matches)
        ].copy()

        # Rates per 100 legal balls bowled
        qualified["overs_bowled"] = qualified["balls_bowled"] / 6.0
        qualified["economy_rate"] = qualified["actual_conceded"] / np.maximum(qualified["overs_bowled"], 1e-6)
        qualified["rpa_per_100"] = (qualified["rpa_total"] / qualified["balls_bowled"]) * 100.0
        qualified["prpa_per_100"] = (qualified["prpa_total"] / qualified["balls_bowled"]) * 100.0
        qualified["wpa_per_100"] = (qualified["wpa_total"] / qualified["balls_bowled"]) * 100.0
        qualified["wickets_per_100"] = (qualified["actual_wickets"] / qualified["balls_bowled"]) * 100.0
        qualified["waa_per_100"] = (qualified["wicket_above_exp"] / qualified["balls_bowled"]) * 100.0

        # High pressure slice with Bayesian shrinkage
        band_col = "bowling_pressure_band" if "bowling_pressure_band" in df.columns else "pressure_band"
        high_press = df[df[band_col] == "high"]
        hp_grp = high_press.groupby("bowler_id").agg(
            hp_balls=("legal_ball", "sum"),
            hp_deliveries=("conceded_actual", "count"),
            hp_rpa=("rpa", "sum"),
            hp_wpa=("wpa_bowler", "sum"),
        )
        qualified = qualified.join(hp_grp, how="left").fillna({"hp_balls": 0, "hp_deliveries": 0, "hp_rpa": 0.0, "hp_wpa": 0.0})

        # Empirical Bayes shrinkage on High-Pressure RPA (Prior weight = 30.0 balls)
        hp_n = qualified["hp_balls"].to_numpy()
        hp_raw_rate = np.where(hp_n > 0, (qualified["hp_rpa"].to_numpy() / np.maximum(hp_n, 1)) * 100.0, 0.0)
        shrunk_hp_rate = (hp_n / (hp_n + 30.0)) * hp_raw_rate
        qualified["raw_hp_rpa_per_100"] = hp_raw_rate
        qualified["hp_rpa_per_100"] = shrunk_hp_rate

        # Match-level consistency (standard deviation of match RPA)
        match_level = df.groupby(["bowler_id", "match_id"])["rpa"].sum().reset_index()
        match_std = match_level.groupby("bowler_id")["rpa"].std().fillna(8.0)
        qualified["match_std"] = match_std
        qualified["reliability_index"] = np.clip(1.0 / (1.0 + (qualified["match_std"] / 10.0)), 0.2, 1.0)

        # Bayesian Shrinkage on PRPA (Prior weight = 90.0 balls)
        pop_mean_prpa = float(qualified["prpa_per_100"].mean())
        shrunk_prpa, se_prpa, ci_lower = self.bayesian.shrink_series(
            observed=qualified["prpa_per_100"],
            volumes=qualified["balls_bowled"],
            prior_weight=90.0,
            population_mean=pop_mean_prpa,
        )
        qualified["shrunk_prpa_per_100"] = shrunk_prpa
        qualified["rating_se"] = se_prpa
        qualified["rating_ci_lower"] = ci_lower

        # Composite Bowling Pressure Index (BPI)
        # 50 benchmark + Shrunk PRPA + Shrunk HP RPA + Wickets Above Exp + WPA + Reliability
        qualified["bowling_pressure_index"] = (
            50.0
            + 1.0 * qualified["shrunk_prpa_per_100"]
            + 0.8 * qualified["hp_rpa_per_100"]
            + 3.0 * qualified["waa_per_100"]
            + 12.0 * qualified["wpa_total"]
            + 5.0 * (qualified["reliability_index"] - 0.5)
        ).round(2)

        # Filter by high pressure volume qualification
        if self.min_hp_balls > 0:
            qualified = qualified[qualified["hp_balls"] >= self.min_hp_balls].copy()

        # Sort and rank
        ranked = qualified.sort_values("bowling_pressure_index", ascending=False).reset_index()
        ranked["rank"] = np.arange(1, len(ranked) + 1)

        logger.info("Bowling evaluation complete. %d bowlers qualified.", len(ranked))
        return ranked
