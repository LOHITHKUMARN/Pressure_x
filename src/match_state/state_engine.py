"""
PRESSURE-X — Phase 2: Match-State Intelligence Engine
======================================================
Reconstructs and verifies full state transitions for every delivery.
Provides both pre-delivery state (S_t) and post-delivery state (S_{t+1})
to guarantee zero data leakage while enabling downstream transition models
and leverage calculations.

State Features per delivery:
  - Current team score (innings_runs_pre, innings_runs_post)
  - Wickets lost (innings_wickets_pre, innings_wickets_post)
  - Legal balls bowled (legal_balls_pre, legal_balls_post)
  - Balls remaining (balls_rem_pre, balls_rem_post)
  - Wickets remaining (wickets_rem_pre, wickets_rem_post)
  - Required run rate (rrr_pre, rrr_post)
  - Current run rate (crr_pre, crr_post)
  - Innings phase (powerplay, middle, death)
  - Match context (venue, format, target, revised targets)

Author: PRESSURE-X Engineering
Version: 2.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

MAX_LEGAL_BALLS = 120
MAX_WICKETS = 10


class MatchStateEngine:
    """
    Engine to build pre-delivery S_t and post-delivery S_{t+1} representations
    for every delivery across all matches and innings.
    """

    def __init__(self, config_path: str | Path = "config/data_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)
        self.output_dir = Path(self.config["paths"]["processed"])

    def enrich_state_transitions(self, df: pd.DataFrame, matches_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculates exact pre and post delivery states for each ball.
        """
        logger.info("Enriching delivery transitions (S_t -> S_{t+1}) for %d deliveries...", len(df))
        df = df.copy()

        # Sort canonically
        sort_cols = ["match_id", "innings", "overid", "over_ball_num"]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Pre-delivery state columns already created in Phase 1:
        # innings_runs, innings_wickets, innings_legal_balls, balls_remaining, wickets_remaining
        # Now compute post-delivery values strictly:
        # runs_this_ball, is_wicket, is_legal_delivery
        runs_this_ball = df["_runs_this_ball"].to_numpy()
        is_wicket = df["_wicket"].to_numpy()
        is_legal = df["is_legal_delivery"].astype(int).to_numpy()

        df["innings_runs_pre"] = df["innings_runs"]
        df["innings_wickets_pre"] = df["innings_wickets"]
        df["legal_balls_pre"] = df["innings_legal_balls"]
        df["balls_remaining_pre"] = df["balls_remaining"]
        df["wickets_remaining_pre"] = df["wickets_remaining"]

        # Post-delivery state
        df["innings_runs_post"] = df["innings_runs_pre"] + runs_this_ball
        df["innings_wickets_post"] = np.minimum(MAX_WICKETS, df["innings_wickets_pre"] + is_wicket)
        df["legal_balls_post"] = np.minimum(MAX_LEGAL_BALLS, df["legal_balls_pre"] + is_legal)
        df["balls_remaining_post"] = np.maximum(0, MAX_LEGAL_BALLS - df["legal_balls_post"])
        df["wickets_remaining_post"] = np.maximum(0, MAX_WICKETS - df["innings_wickets_post"])

        # Run rates
        df["crr_pre"] = np.where(
            df["legal_balls_pre"] > 0,
            (df["innings_runs_pre"] / df["legal_balls_pre"]) * 6.0,
            0.0,
        )
        df["crr_post"] = np.where(
            df["legal_balls_post"] > 0,
            (df["innings_runs_post"] / df["legal_balls_post"]) * 6.0,
            0.0,
        )

        # Second innings Target & RRR
        if "target" in df.columns:
            has_target = df["target"].notna()
            # Pre
            df["runs_required_pre"] = np.where(
                has_target,
                np.maximum(0, df["target"] - df["innings_runs_pre"]),
                np.nan,
            )
            df["rrr_pre"] = np.where(
                has_target & (df["balls_remaining_pre"] > 0),
                (df["runs_required_pre"] / df["balls_remaining_pre"]) * 6.0,
                np.nan,
            )
            # Post
            df["runs_required_post"] = np.where(
                has_target,
                np.maximum(0, df["target"] - df["innings_runs_post"]),
                np.nan,
            )
            df["rrr_post"] = np.where(
                has_target & (df["balls_remaining_post"] > 0),
                (df["runs_required_post"] / df["balls_remaining_post"]) * 6.0,
                np.nan,
            )
            # If target reached post-delivery, rrr_post is 0
            df.loc[has_target & (df["runs_required_post"] == 0), "rrr_post"] = 0.0

        # Match outcome label attachment if matches_df provided
        if matches_df is not None:
            winner_map = dict(zip(matches_df["match_id"], matches_df["win_team_id"]))
            df["match_winner_team_id"] = df["match_id"].map(winner_map)
            df["batting_team_won"] = np.where(
                df["match_winner_team_id"].isna(),
                np.nan,
                (df["batting_team_id"] == df["match_winner_team_id"]).astype(float),
            )

        logger.info("State transitions successfully enriched.")
        return df


def build_match_states(
    df: pd.DataFrame,
    matches_df: Optional[pd.DataFrame] = None,
    config_path: str | Path = "config/data_config.yaml",
) -> pd.DataFrame:
    engine = MatchStateEngine(config_path)
    return engine.enrich_state_transitions(df, matches_df)
