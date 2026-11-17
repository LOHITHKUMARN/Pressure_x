"""
PRESSURE-X — Phase 12: Monte Carlo Simulation & Ranking Stability Engine
=========================================================================
Quantifies ranking uncertainty through match-clustered Monte Carlo simulations:
  - Evaluates P(Player ranks in Top 10)
  - Computes 95% Credible Interval for player ranks: [Rank_2.5%, Rank_97.5%]
  - Analyzes sensitivity to match inclusion and sample variance
  - Strictly aligns bootstrap scoring with the production BPI formula:
      * Batting: Baseline 50 + Shrunk PRAA + Shrunk HP-RAA + 12*WPA + 5*(Reliability - 0.5)
      * Bowling: Baseline 50 + Shrunk PRPA + Shrunk HP-RPA + 3*WAA + 12*WPA + 5*(Reliability - 0.5)

Author: PRESSURE-X Engineering
Version: 2.1.0
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from bowling.expected_conceded import ExpectedBowlingModel

logger = logging.getLogger(__name__)


class MonteCarloSimulationEngine:
    """
    Performs match-level bootstrap simulations to evaluate leaderboard stability.
    Uses the exact production Bayesian-shrunk pressure performance indices.
    """

    def __init__(self, n_iterations: int = 1000, random_state: int = 42):
        self.n_iterations = n_iterations
        self.random_state = random_state

    def simulate_batting_stability(
        self,
        deliveries_df: pd.DataFrame,
        top_candidates: List[any],
    ) -> pd.DataFrame:
        """
        Simulates match-clustered bootstrap samples to evaluate batter rank stability
        using the production Batting Pressure Index (BPI) formula.
        """
        logger.info("Running %d Monte Carlo simulations for batting stability...", self.n_iterations)
        rng = np.random.default_rng(self.random_state)

        matches = deliveries_df["match_id"].unique()
        n_matches = len(matches)

        # Filter deliveries to qualified player pool for speed
        pool_df = deliveries_df[deliveries_df["striker_id"].isin(top_candidates)].copy()
        pool_df["raa"] = pool_df["runs"] - pool_df["exp_runs_bat"]
        pool_df["praa"] = pool_df["raa"] * (1.0 + pool_df["batting_pressure_score"])

        # Legal balls & High pressure indicator
        if "is_legal_delivery" in pool_df.columns:
            pool_df["legal_ball"] = pool_df["is_legal_delivery"].astype(float)
        else:
            pool_df["legal_ball"] = 1.0

        hp_col = "batting_pressure_band" if "batting_pressure_band" in pool_df.columns else "pressure_band"
        hp_mask = pool_df[hp_col] == "high"
        pool_df["hp_ball"] = (hp_mask & (pool_df["legal_ball"] > 0)).astype(float)
        pool_df["hp_raa"] = np.where(hp_mask, pool_df["raa"], 0.0)

        # Pre-aggregate to (striker_id, match_id) level for fast iteration
        match_agg = (
            pool_df.groupby(["striker_id", "match_id"])
            .agg(
                balls=("legal_ball", "sum"),
                praa=("praa", "sum"),
                raa=("raa", "sum"),
                wpa=("wpa_batter", "sum"),
                hp_balls=("hp_ball", "sum"),
                hp_raa=("hp_raa", "sum"),
            )
            .reset_index()
        )

        rank_records = {pid: [] for pid in top_candidates}

        for _ in range(self.n_iterations):
            sampled_matches = rng.choice(matches, size=n_matches, replace=True)
            match_counts = pd.Series(sampled_matches).value_counts()

            # Active matches with positive sampled weight
            sim_df = match_agg[match_agg["match_id"].isin(match_counts.index)].copy()
            sim_df["w"] = sim_df["match_id"].map(match_counts)

            sim_balls = (sim_df["balls"] * sim_df["w"]).groupby(sim_df["striker_id"]).sum()
            sim_praa = (sim_df["praa"] * sim_df["w"]).groupby(sim_df["striker_id"]).sum()
            sim_wpa = (sim_df["wpa"] * sim_df["w"]).groupby(sim_df["striker_id"]).sum()
            sim_hp_balls = (sim_df["hp_balls"] * sim_df["w"]).groupby(sim_df["striker_id"]).sum()
            sim_hp_raa = (sim_df["hp_raa"] * sim_df["w"]).groupby(sim_df["striker_id"]).sum()

            # Match-level consistency (weighted std of match RAA)
            weighted_raa_sum = (sim_df["raa"] * sim_df["w"]).groupby(sim_df["striker_id"]).sum()
            weighted_m_count = sim_df["w"].groupby(sim_df["striker_id"]).sum()
            mean_raa = weighted_raa_sum / np.maximum(weighted_m_count, 1)

            sim_df["dev_sq"] = sim_df["w"] * ((sim_df["raa"] - sim_df["striker_id"].map(mean_raa)) ** 2)
            dev_sum = sim_df["dev_sq"].groupby(sim_df["striker_id"]).sum()
            var_raa = np.where(weighted_m_count > 1, dev_sum / np.maximum(weighted_m_count - 1, 1), 100.0)
            std_raa = np.sqrt(np.maximum(var_raa, 0.0))
            reliability = np.clip(1.0 / (1.0 + (std_raa / 12.0)), 0.2, 1.0)

            # Production Bayesian Shrinkage
            praa_rate = np.where(sim_balls > 0, (sim_praa / np.maximum(sim_balls, 1)) * 100.0, 0.0)
            shrunk_praa = (sim_balls / (sim_balls + 80.0)) * praa_rate

            hp_raw_rate = np.where(sim_hp_balls > 0, (sim_hp_raa / np.maximum(sim_hp_balls, 1)) * 100.0, 0.0)
            shrunk_hp_rate = (sim_hp_balls / (sim_hp_balls + 30.0)) * hp_raw_rate

            # Exact Production Formula
            score = (
                50.0
                + 1.0 * shrunk_praa
                + 0.8 * shrunk_hp_rate
                + 12.0 * sim_wpa
                + 5.0 * (reliability - 0.5)
            )

            # Reindex to all candidates (filling absent candidates with 0 score)
            score = score.reindex(top_candidates, fill_value=0.0)
            ranked_pids = list(score.sort_values(ascending=False).index)

            for rank_idx, pid in enumerate(ranked_pids, start=1):
                if pid in rank_records:
                    rank_records[pid].append(rank_idx)

        summary_rows = []
        for pid in top_candidates:
            ranks = np.array(rank_records[pid])
            if len(ranks) == 0:
                continue
            prob_top10 = float(np.mean(ranks <= 10))
            prob_top5 = float(np.mean(ranks <= 5))
            median_rank = float(np.median(ranks))
            ci_low = float(np.percentile(ranks, 2.5))
            ci_high = float(np.percentile(ranks, 97.5))
            stability_score = float(1.0 / (1.0 + np.std(ranks)))

            summary_rows.append({
                "striker_id": pid,
                "prob_top_10": prob_top10,
                "prob_top_5": prob_top5,
                "median_sim_rank": median_rank,
                "rank_ci_95": f"[{ci_low:.0f} - {ci_high:.0f}]",
                "stability_score": stability_score,
            })

        logger.info("Batting Monte Carlo simulation completed.")
        return pd.DataFrame(summary_rows)

    def simulate_bowling_stability(
        self,
        deliveries_df: pd.DataFrame,
        top_candidates: List[any],
    ) -> pd.DataFrame:
        """
        Simulates match-clustered bootstrap samples to evaluate bowler rank stability
        using the production Bowling Pressure Index (BPI) formula.
        """
        logger.info("Running %d Monte Carlo simulations for bowling stability...", self.n_iterations)
        rng = np.random.default_rng(self.random_state)

        matches = deliveries_df["match_id"].unique()
        n_matches = len(matches)

        pool_df = deliveries_df[deliveries_df["bowler_id"].isin(top_candidates)].copy()

        # Accurate bowler charge and credited wickets
        conceded, bowler_wicket = ExpectedBowlingModel.calculate_bowler_charge(pool_df)
        pool_df["conceded_actual"] = conceded
        pool_df["bowler_wicket_actual"] = bowler_wicket

        pool_df["rpa"] = pool_df["exp_runs_bowl"] - pool_df["conceded_actual"]
        pool_df["prpa"] = pool_df["rpa"] * (1.0 + pool_df["bowling_pressure_score"])
        pool_df["wicket_above_exp"] = pool_df["bowler_wicket_actual"] - pool_df["prob_wicket_bowl"]

        # Legal balls & High pressure indicator
        if "is_legal_delivery" in pool_df.columns:
            pool_df["legal_ball"] = pool_df["is_legal_delivery"].astype(float)
        else:
            pool_df["legal_ball"] = 1.0

        hp_col = "bowling_pressure_band" if "bowling_pressure_band" in pool_df.columns else "pressure_band"
        hp_mask = pool_df[hp_col] == "high"
        pool_df["hp_ball"] = (hp_mask & (pool_df["legal_ball"] > 0)).astype(float)
        pool_df["hp_rpa"] = np.where(hp_mask, pool_df["rpa"], 0.0)

        # Pre-aggregate to (bowler_id, match_id) level
        match_agg = (
            pool_df.groupby(["bowler_id", "match_id"])
            .agg(
                balls=("legal_ball", "sum"),
                prpa=("prpa", "sum"),
                rpa=("rpa", "sum"),
                wpa=("wpa_bowler", "sum"),
                waa=("wicket_above_exp", "sum"),
                hp_balls=("hp_ball", "sum"),
                hp_rpa=("hp_rpa", "sum"),
            )
            .reset_index()
        )

        rank_records = {pid: [] for pid in top_candidates}

        for _ in range(self.n_iterations):
            sampled_matches = rng.choice(matches, size=n_matches, replace=True)
            match_counts = pd.Series(sampled_matches).value_counts()

            sim_df = match_agg[match_agg["match_id"].isin(match_counts.index)].copy()
            sim_df["w"] = sim_df["match_id"].map(match_counts)

            sim_balls = (sim_df["balls"] * sim_df["w"]).groupby(sim_df["bowler_id"]).sum()
            sim_prpa = (sim_df["prpa"] * sim_df["w"]).groupby(sim_df["bowler_id"]).sum()
            sim_wpa = (sim_df["wpa"] * sim_df["w"]).groupby(sim_df["bowler_id"]).sum()
            sim_waa = (sim_df["waa"] * sim_df["w"]).groupby(sim_df["bowler_id"]).sum()
            sim_hp_balls = (sim_df["hp_balls"] * sim_df["w"]).groupby(sim_df["bowler_id"]).sum()
            sim_hp_rpa = (sim_df["hp_rpa"] * sim_df["w"]).groupby(sim_df["bowler_id"]).sum()

            # Match-level consistency (weighted std of match RPA)
            weighted_rpa_sum = (sim_df["rpa"] * sim_df["w"]).groupby(sim_df["bowler_id"]).sum()
            weighted_m_count = sim_df["w"].groupby(sim_df["bowler_id"]).sum()
            mean_rpa = weighted_rpa_sum / np.maximum(weighted_m_count, 1)

            sim_df["dev_sq"] = sim_df["w"] * ((sim_df["rpa"] - sim_df["bowler_id"].map(mean_rpa)) ** 2)
            dev_sum = sim_df["dev_sq"].groupby(sim_df["bowler_id"]).sum()
            var_rpa = np.where(weighted_m_count > 1, dev_sum / np.maximum(weighted_m_count - 1, 1), 64.0)
            std_rpa = np.sqrt(np.maximum(var_rpa, 0.0))
            reliability = np.clip(1.0 / (1.0 + (std_rpa / 10.0)), 0.2, 1.0)

            # Production Bayesian Shrinkage
            prpa_rate = np.where(sim_balls > 0, (sim_prpa / np.maximum(sim_balls, 1)) * 100.0, 0.0)
            shrunk_prpa = (sim_balls / (sim_balls + 90.0)) * prpa_rate

            hp_raw_rate = np.where(sim_hp_balls > 0, (sim_hp_rpa / np.maximum(sim_hp_balls, 1)) * 100.0, 0.0)
            shrunk_hp_rate = (sim_hp_balls / (sim_hp_balls + 30.0)) * hp_raw_rate

            waa_per_100 = np.where(sim_balls > 0, (sim_waa / np.maximum(sim_balls, 1)) * 100.0, 0.0)

            # Exact Production Formula
            score = (
                50.0
                + 1.0 * shrunk_prpa
                + 0.8 * shrunk_hp_rate
                + 3.0 * waa_per_100
                + 12.0 * sim_wpa
                + 5.0 * (reliability - 0.5)
            )

            score = score.reindex(top_candidates, fill_value=0.0)
            ranked_pids = list(score.sort_values(ascending=False).index)

            for rank_idx, pid in enumerate(ranked_pids, start=1):
                if pid in rank_records:
                    rank_records[pid].append(rank_idx)

        summary_rows = []
        for pid in top_candidates:
            ranks = np.array(rank_records[pid])
            if len(ranks) == 0:
                continue
            prob_top10 = float(np.mean(ranks <= 10))
            prob_top5 = float(np.mean(ranks <= 5))
            median_rank = float(np.median(ranks))
            ci_low = float(np.percentile(ranks, 2.5))
            ci_high = float(np.percentile(ranks, 97.5))
            stability_score = float(1.0 / (1.0 + np.std(ranks)))

            summary_rows.append({
                "bowler_id": pid,
                "prob_top_10": prob_top10,
                "prob_top_5": prob_top5,
                "median_sim_rank": median_rank,
                "rank_ci_95": f"[{ci_low:.0f} - {ci_high:.0f}]",
                "stability_score": stability_score,
            })

        logger.info("Bowling Monte Carlo simulation completed.")
        return pd.DataFrame(summary_rows)
