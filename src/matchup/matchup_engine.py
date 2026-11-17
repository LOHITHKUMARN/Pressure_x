"""
PRESSURE-X — Phase 9: Matchup & Environmental Intelligence Engine
==================================================================
Quantifies contextual modifiers that influence expected delivery outcomes:
  1. Venue Scoring Index (VSI):
     VSI(venue, phase) = (Venue runs/ball in phase) / (Global runs/ball in phase)
     Adjusts expected performance for high-scoring batting paradises vs bowling minefields.

  2. Matchup Interaction Dimensions:
     - Batter Hand (RHB / LHB) vs Bowler Sub-Type (Pace / Off-Spin / Leg-Spin / SLA)
     - Pitch / Innings Phase interactions (Death overs vs Powerplay matchup dynamics)

  3. Empirical Bayes Regularized Matchup Delta:
     Shrinks small-sample specific matchups toward broad category baselines.

Author: PRESSURE-X Engineering
Version: 2.0.0
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MatchupIntelligenceEngine:
    """
    Computes venue context indices and regularized matchup modifiers.
    """

    def __init__(self, prior_balls_venue: float = 300.0, prior_balls_matchup: float = 100.0):
        self.prior_balls_venue = prior_balls_venue
        self.prior_balls_matchup = prior_balls_matchup
        self.venue_index_map: Dict[Tuple[str, str], float] = {}
        self.matchup_delta_map: Dict[str, float] = {}
        self.global_run_rate: float = 1.34

    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Learns venue environmental ratings and matchup deltas from delivery history.
        """
        logger.info("Fitting Matchup & Environmental Intelligence Engine...")
        self.global_run_rate = float(df["_runs_this_ball"].mean())

        # 1. Venue Scoring Index by Phase
        venue_col = "venue_id" if "venue_id" in df.columns else None
        if venue_col is None and "venue" in df.columns:
            venue_col = "venue"

        if venue_col:
            v_agg = df.groupby([venue_col, "innings_phase"])["_runs_this_ball"].agg(["count", "mean"]).reset_index()
            for _, r in v_agg.iterrows():
                v_id = r[venue_col]
                phase = r["innings_phase"]
                n = r["count"]
                obs_mean = r["mean"]
                # Shrink towards global rate
                shrunk = (n / (n + self.prior_balls_venue)) * obs_mean + (self.prior_balls_venue / (n + self.prior_balls_venue)) * self.global_run_rate
                vsi = shrunk / self.global_run_rate
                self.venue_index_map[(str(v_id), str(phase))] = float(vsi)

        # 2. Granular Matchup Archetypes: (batting_type, bowling_type, innings_phase)
        m_agg = df.groupby(["batting_type", "bowling_type", "innings_phase"])["_runs_this_ball"].agg(["count", "mean"]).reset_index()
        for _, r in m_agg.iterrows():
            key = f"{r['batting_type']}_{r['bowling_type']}_{r['innings_phase']}"
            n = r["count"]
            obs_mean = r["mean"]
            delta = obs_mean - self.global_run_rate
            shrunk_delta = (n / (n + self.prior_balls_matchup)) * delta
            self.matchup_delta_map[key] = float(shrunk_delta)

        metrics = {
            "venues_modeled": len(self.venue_index_map),
            "matchup_archetypes": len(self.matchup_delta_map),
            "global_run_rate": self.global_run_rate,
        }
        logger.info("Matchup Intelligence Engine fitted: %s", metrics)
        return metrics

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appends venue_scoring_index and matchup_run_delta to deliveries.
        """
        df = df.copy()
        venue_col = "venue_id" if "venue_id" in df.columns else ("venue" if "venue" in df.columns else None)

        if venue_col and self.venue_index_map:
            v_keys = list(zip(df[venue_col].astype(str), df["innings_phase"].astype(str)))
            df["venue_scoring_index"] = [self.venue_index_map.get(k, 1.0) for k in v_keys]
        else:
            df["venue_scoring_index"] = 1.0

        m_keys = (
            df["batting_type"].fillna("RHB").astype(str) + "_" +
            df["bowling_type"].fillna("Right-arm medium").astype(str) + "_" +
            df["innings_phase"].fillna("middle").astype(str)
        )
        df["matchup_run_delta"] = [self.matchup_delta_map.get(k, 0.0) for k in m_keys]

        return df
