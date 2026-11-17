"""
PRESSURE-X — Feature Engineering Module
=========================================
Extracts strictly causal, pre-delivery features for Expected Performance models:
  - Match state: overid, over_ball_num, innings, legal_balls_pre, balls_remaining_pre
  - Scoring state: innings_runs_pre, crr_pre, runs_required_pre, rrr_pre, target
  - Resource state: wickets_remaining_pre, innings_wickets_pre
  - Pressure state: wp_pre, match_closeness, batting_pressure_score, bowling_pressure_score, pressure_score
  - Innings Phase: powerplay, middle, death (one-hot or ordinal)
  - Matchup context:
      - batter_hand (RHB=0, LHB=1)
      - pace_or_spin (Pace=0, Spin=1)
      - matchup_category (RHB_v_Pace, RHB_v_Spin, LHB_v_Pace, LHB_v_Spin)

Guarantees 100% causal alignment: NO post-delivery leakage.

Author: PRESSURE-X Engineering
Version: 2.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Transforms enriched delivery records into ML feature matrices.
    """

    FEATURE_COLS = [
        "overid",
        "over_ball_num",
        "innings",
        "legal_balls_pre",
        "balls_remaining_pre",
        "innings_runs_pre",
        "innings_wickets_pre",
        "wickets_remaining_pre",
        "crr_pre",
        "runs_required_pre",
        "rrr_pre",
        "target",
        "wp_pre",
        "match_closeness",
        "batting_pressure_score",
        "bowling_pressure_score",
        "pressure_score",
        "phase_powerplay",
        "phase_middle",
        "phase_death",
        "is_lhb",
        "is_spin",
        "matchup_lhb_spin",
        "matchup_lhb_pace",
        "matchup_rhb_spin",
        "matchup_rhb_pace",
    ]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts and encodes pre-delivery features into a DataFrame.
        """
        feats = pd.DataFrame(index=df.index)

        feats["overid"] = df["overid"].astype(float)
        feats["over_ball_num"] = df["over_ball_num"].astype(float)
        feats["innings"] = df["innings"].astype(float)

        feats["legal_balls_pre"] = df["legal_balls_pre"].astype(float)
        feats["balls_remaining_pre"] = df["balls_remaining_pre"].astype(float)
        feats["innings_runs_pre"] = df["innings_runs_pre"].astype(float)
        feats["innings_wickets_pre"] = df["innings_wickets_pre"].astype(float)
        feats["wickets_remaining_pre"] = df["wickets_remaining_pre"].astype(float)
        feats["crr_pre"] = df["crr_pre"].fillna(0.0).astype(float)

        # Chase variables
        feats["runs_required_pre"] = df["runs_required_pre"].fillna(0.0).astype(float)
        feats["rrr_pre"] = df["rrr_pre"].fillna(0.0).clip(0.0, 36.0).astype(float)
        feats["target"] = df["target"].fillna(160.0).astype(float)

        # Pressure and win probability
        feats["wp_pre"] = df["wp_pre"].fillna(0.5).astype(float)
        feats["match_closeness"] = df["match_closeness"].fillna(0.5).astype(float)
        feats["batting_pressure_score"] = df["batting_pressure_score"].fillna(0.5).astype(float)
        feats["bowling_pressure_score"] = df["bowling_pressure_score"].fillna(0.5).astype(float)
        feats["pressure_score"] = df["pressure_score"].fillna(0.5).astype(float)

        # Phase one-hot
        phase = df["innings_phase"].fillna("middle")
        feats["phase_powerplay"] = (phase == "powerplay").astype(float)
        feats["phase_middle"] = (phase == "middle").astype(float)
        feats["phase_death"] = (phase == "death").astype(float)

        # Matchups
        bat_type = df["batting_type"].fillna("RHB")
        is_lhb = (bat_type == "LHB").astype(float)
        pace_spin = df["pace_or_spin"].fillna("Pace")
        is_spin = (pace_spin == "Spin").astype(float)

        feats["is_lhb"] = is_lhb
        feats["is_spin"] = is_spin

        feats["matchup_lhb_spin"] = is_lhb * is_spin
        feats["matchup_lhb_pace"] = is_lhb * (1.0 - is_spin)
        feats["matchup_rhb_spin"] = (1.0 - is_lhb) * is_spin
        feats["matchup_rhb_pace"] = (1.0 - is_lhb) * (1.0 - is_spin)

        return feats[self.FEATURE_COLS]
