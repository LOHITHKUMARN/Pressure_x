"""
PRESSURE-X — Automated Tests: Phase 2, 3, 4, 7
===============================================
Tests for match state transitions, win probability boundary conditions,
leverage metrics, pressure bands, and empirical Bayes shrinkage.

Run: pytest tests/test_phase2_3.py -v

Author: PRESSURE-X Engineering
Version: 2.0.0
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bayesian.shrinkage import BayesianShrinkageEngine
from match_state.state_engine import MatchStateEngine
from win_probability.wp_model import WinProbabilityModel
from pressure.pressure_engine import PressureIntelligenceEngine
from batting.batting_rating import BattingIntelligenceEngine
from bowling.bowling_rating import BowlingIntelligenceEngine
from simulation.monte_carlo import MonteCarloSimulationEngine


class TestStateTransitions:

    def test_post_state_increment(self):
        engine = MatchStateEngine()
        df = pd.DataFrame({
            "match_id": [1],
            "innings": [1],
            "overid": [1],
            "over_ball_num": [1],
            "innings_runs": [10],
            "innings_wickets": [1],
            "innings_legal_balls": [6],
            "balls_remaining": [114],
            "wickets_remaining": [9],
            "_runs_this_ball": [4],
            "_wicket": [0],
            "is_legal_delivery": [True],
            "target": [np.nan],
        })
        enriched = engine.enrich_state_transitions(df)
        assert enriched.iloc[0]["innings_runs_post"] == 14
        assert enriched.iloc[0]["innings_wickets_post"] == 1
        assert enriched.iloc[0]["legal_balls_post"] == 7
        assert enriched.iloc[0]["balls_remaining_post"] == 113

    def test_wicket_decrements_remaining(self):
        engine = MatchStateEngine()
        df = pd.DataFrame({
            "match_id": [1],
            "innings": [1],
            "overid": [2],
            "over_ball_num": [1],
            "innings_runs": [20],
            "innings_wickets": [2],
            "innings_legal_balls": [7],
            "balls_remaining": [113],
            "wickets_remaining": [8],
            "_runs_this_ball": [0],
            "_wicket": [1],
            "is_legal_delivery": [True],
            "target": [np.nan],
        })
        enriched = engine.enrich_state_transitions(df)
        assert enriched.iloc[0]["innings_wickets_post"] == 3
        assert enriched.iloc[0]["wickets_remaining_post"] == 7


class TestWinProbabilityBoundaries:

    def test_target_achieved_wp_equals_one(self):
        wp_model = WinProbabilityModel()
        # When runs_required <= 0 in innings 2 (exact 1.0)
        p = wp_model.predict_state_wp(
            innings=np.array([2]),
            balls_remaining=np.array([10]),
            wickets_remaining=np.array([5]),
            innings_runs=np.array([165]),
            crr=np.array([9.0]),
            phases=pd.Series(["death"]),
            runs_required=np.array([0]),  # reached target
            rrr=np.array([0.0]),
            target=np.array([165]),
        )
        assert p[0] == 1.0

    def test_all_out_wp_equals_zero(self):
        wp_model = WinProbabilityModel()
        # When wickets_remaining <= 0 and runs_required > 0 (exact 0.0)
        p = wp_model.predict_state_wp(
            innings=np.array([2]),
            balls_remaining=np.array([20]),
            wickets_remaining=np.array([0]),  # all out
            innings_runs=np.array([130]),
            crr=np.array([7.8]),
            phases=pd.Series(["death"]),
            runs_required=np.array([35]),
            rrr=np.array([10.5]),
            target=np.array([165]),
        )
        assert p[0] == 0.0

    def test_balls_exhausted_wp_equals_zero(self):
        wp_model = WinProbabilityModel()
        # When balls_remaining <= 0 and runs_required > 0 (exact 0.0)
        p = wp_model.predict_state_wp(
            innings=np.array([2]),
            balls_remaining=np.array([0]),  # balls exhausted
            wickets_remaining=np.array([6]),
            innings_runs=np.array([150]),
            crr=np.array([7.5]),
            phases=pd.Series(["death"]),
            runs_required=np.array([15]),
            rrr=np.array([99.0]),
            target=np.array([165]),
        )
        assert p[0] == 0.0


class TestChronologicalValidation:

    def test_chronological_split_disjoint_matches(self):
        """Proves that chronological splitting produces strictly disjoint match sets with 0 overlap."""
        # Create synthetic matches across dates
        df = pd.DataFrame({
            "match_id": [101, 101, 102, 102, 103, 103, 104, 104, 105, 105],
            "match_date": ["2022-01-01", "2022-01-01", "2022-02-01", "2022-02-01",
                           "2023-01-01", "2023-01-01", "2024-01-01", "2024-01-01",
                           "2025-01-01", "2025-01-01"],
            "innings": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "batting_team_won": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "is_abandoned": [False] * 10,
            "innings_phase": ["middle"] * 10,
            "crr_pre": [7.0] * 10,
            "rrr_pre": [8.0] * 10,
            "runs_required_pre": [40.0] * 10,
            "balls_remaining_pre": [60] * 10,
            "wickets_remaining_pre": [7] * 10,
            "innings_runs_pre": [70] * 10,
            "target": [160.0] * 10,
        })
        wp_model = WinProbabilityModel()
        report = wp_model.evaluate_chronological_holdout(df, train_split_ratio=0.80)

        assert report["disjoint_match_guarantee"] is True
        assert report["train_matches"] == 4
        assert report["test_matches"] == 1
        assert report["test_date_range"][0] >= report["train_date_range"][1]

    def test_test_matches_never_in_training(self, tmp_path):
        """Proves that test match IDs are never passed into training partitions."""
        from evaluation.ablation_validator import AblationValidationEngine
        df = pd.DataFrame({
            "match_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "match_date": ["2022-01-01", "2022-01-01", "2022-06-01", "2022-06-01",
                           "2023-01-01", "2023-01-01", "2024-01-01", "2024-01-01",
                           "2025-01-01", "2025-01-01"],
            "innings": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "batting_team_won": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "is_abandoned": [False] * 10,
            "innings_phase": ["middle"] * 10,
            "crr_pre": [7.0] * 10,
            "rrr_pre": [8.0] * 10,
            "runs_required_pre": [40.0] * 10,
            "balls_remaining_pre": [60] * 10,
            "wickets_remaining_pre": [7] * 10,
            "innings_runs_pre": [70] * 10,
            "target": [160.0] * 10,
        })
        engine = AblationValidationEngine(train_split_ratio=0.80, reports_dir=tmp_path)
        report = engine.run_chronological_validation(df)
        assert report["train_matches"] == 4
        assert report["test_matches"] == 1
        assert report["disjoint_match_guarantee"] is True


class TestPressureBands:

    @pytest.fixture
    def mock_delivery_data(self):
        """Creates sample deliveries representing distinct match situations."""
        return pd.DataFrame({
            "match_id": [1, 1],
            "innings": [2, 2],
            "balls_remaining_pre": [12, 12],
            "wickets_remaining_pre": [2, 9],  # Row 0: 8 wickets down (collapse). Row 1: 1 wicket down
            "innings_runs_pre": [130, 150],
            "crr_pre": [7.2, 8.3],
            "innings_phase": ["death", "death"],
            "runs_required_pre": [35, 15],  # Row 0: high RRR (17.5). Row 1: low RRR (7.5)
            "rrr_pre": [17.5, 7.5],
            "target": [165, 165],
            "balls_remaining_post": [11, 11],
            "wickets_remaining_post": [2, 9],
            "innings_runs_post": [131, 154],
            "crr_post": [7.21, 8.48],
            "runs_required_post": [34, 11],
            "rrr_post": [18.5, 6.0],
        })

    def test_separate_pressure_bands_created(self, mock_delivery_data):
        engine = PressureIntelligenceEngine()
        enriched = engine.compute_pressure_metrics(mock_delivery_data)

        # Ensure all three bands are populated
        for col in ["batting_pressure_band", "bowling_pressure_band", "pressure_band"]:
            assert col in enriched.columns
            assert set(enriched[col].unique()).issubset({"low", "medium", "high"})

    def test_asymmetric_pressure_batting_collapse(self, mock_delivery_data):
        """In a batting collapse chasing 17.5 RRR with 2 wickets left, batting pressure significantly exceeds bowling pressure."""
        engine = PressureIntelligenceEngine()
        enriched = engine.compute_pressure_metrics(mock_delivery_data)

        collapse_ball = enriched.iloc[0]
        # Batter faces high situational stress while bowler enjoys low pressure
        assert collapse_ball["batting_pressure_score"] > collapse_ball["bowling_pressure_score"]
        assert collapse_ball["bowling_pressure_band"] == "low"
        assert collapse_ball["batting_pressure_band"] in ["medium", "high"]

    def test_rating_engines_use_respective_pressure_bands(self):
        """Confirms that batting ratings query batting_pressure_band and bowling ratings query bowling_pressure_band."""
        df_bat = pd.DataFrame({
            "striker_id": [10, 10, 10],
            "striker_name": ["Batter A"] * 3,
            "match_id": [1, 2, 3],
            "runs": [4, 6, 1],
            "exp_runs_bat": [3.0, 3.0, 3.0],
            "batting_pressure_score": [0.8, 0.5, 0.2],
            "prob_boundary_bat": [0.2, 0.2, 0.2],
            "prob_dot_bat": [0.4, 0.4, 0.4],
            "prob_wicket_bat": [0.05, 0.05, 0.05],
            "_wicket": [0, 0, 0],
            "wpa_batter": [0.05, 0.10, -0.05],
            "batting_pressure_band": ["high", "medium", "low"],
            "pressure_band": ["medium", "medium", "medium"],  # generic composite is NOT high
        })

        b_engine = BattingIntelligenceEngine(min_balls=1, min_matches=1, min_hp_balls=0)
        b_ratings = b_engine.evaluate_batters(df_bat)
        # Only row 0 had batting_pressure_band == "high"
        assert b_ratings.iloc[0]["hp_balls"] == 1

        df_bowl = pd.DataFrame({
            "bowler_id": [20, 20, 20],
            "bowler_name": ["Bowler B"] * 3,
            "match_id": [1, 2, 3],
            "runs": [4, 1, 0],
            "_runs_this_ball": [4, 1, 0],
            "extras_type": [np.nan, np.nan, np.nan],
            "dismissal_type": [np.nan, np.nan, "bowled"],
            "_wicket": [0, 0, 1],
            "exp_runs_bowl": [3.0, 3.0, 3.0],
            "bowling_pressure_score": [0.2, 0.7, 0.9],
            "prob_dot_bowl": [0.4, 0.4, 0.4],
            "prob_boundary_bowl": [0.2, 0.2, 0.2],
            "prob_wicket_bowl": [0.05, 0.05, 0.05],
            "wpa_bowler": [-0.05, 0.05, 0.10],
            "bowling_pressure_band": ["low", "high", "high"],
            "pressure_band": ["medium", "medium", "medium"],  # generic composite is NOT high
        })

        bowl_engine = BowlingIntelligenceEngine(min_balls=1, min_matches=1, min_hp_balls=0)
        bowl_ratings = bowl_engine.evaluate_bowlers(df_bowl)
        # Rows 1 and 2 had bowling_pressure_band == "high"
        assert bowl_ratings.iloc[0]["hp_balls"] == 2


class TestBayesianShrinkage:

    def test_small_sample_shrinks_to_prior(self):
        engine = BayesianShrinkageEngine(prior_balls_batting=80.0)
        # Player with only 5 balls and an extreme PRAA of +100
        observed = pd.Series([100.0])
        volumes = pd.Series([5.0])
        shrunk, se, ci_lower = engine.shrink_series(
            observed=observed,
            volumes=volumes,
            prior_weight=80.0,
            population_mean=0.0,
        )
        # Shrunk value should be (5 / 85) * 100 = ~5.88, heavily pulled toward 0
        assert shrunk.iloc[0] < 10.0
        assert shrunk.iloc[0] > 4.0

    def test_large_sample_retains_observed_performance(self):
        engine = BayesianShrinkageEngine(prior_balls_batting=80.0)
        # Veteran player with 800 balls and PRAA of +25.0
        observed = pd.Series([25.0])
        volumes = pd.Series([800.0])
        shrunk, se, ci_lower = engine.shrink_series(
            observed=observed,
            volumes=volumes,
            prior_weight=80.0,
            population_mean=0.0,
        )
        # (800 / 880) * 25.0 = 22.72
        assert shrunk.iloc[0] > 22.0
        assert shrunk.iloc[0] <= 25.0


class TestMonteCarloStability:

    def test_monte_carlo_batting_stability_structure(self):
        df_bat = pd.DataFrame({
            "match_id": [1, 1, 2, 2, 3, 3] * 5,
            "striker_id": [101, 102] * 15,
            "runs": [4, 1, 6, 0, 2, 4] * 5,
            "exp_runs_bat": [2.5, 1.5, 3.0, 1.2, 1.8, 2.2] * 5,
            "batting_pressure_score": [0.8, 0.4, 0.9, 0.3, 0.5, 0.7] * 5,
            "batting_pressure_band": ["high", "medium", "high", "low", "medium", "high"] * 5,
            "wpa_batter": [0.05, -0.01, 0.08, -0.03, 0.01, 0.04] * 5,
        })
        mc = MonteCarloSimulationEngine(n_iterations=20, random_state=42)
        res = mc.simulate_batting_stability(df_bat, [101, 102])

        assert len(res) == 2
        for col in ["striker_id", "prob_top_10", "prob_top_5", "median_sim_rank", "rank_ci_95", "stability_score"]:
            assert col in res.columns
        assert (res["prob_top_10"] >= 0.0).all() and (res["prob_top_10"] <= 1.0).all()
        # Striker 101 has consistently higher RAA and WPA, should have higher/better median rank than 102
        rank_101 = res.loc[res["striker_id"] == 101, "median_sim_rank"].values[0]
        rank_102 = res.loc[res["striker_id"] == 102, "median_sim_rank"].values[0]
        assert rank_101 <= rank_102

    def test_monte_carlo_bowling_stability_structure(self):
        df_bowl = pd.DataFrame({
            "match_id": [1, 1, 2, 2, 3, 3] * 5,
            "bowler_id": [201, 202] * 15,
            "runs": [1, 4, 0, 6, 1, 4] * 5,
            "_runs_this_ball": [1, 4, 0, 6, 1, 4] * 5,
            "extras_type": [np.nan] * 30,
            "dismissal_type": ["bowled", np.nan, "caught", np.nan, np.nan, np.nan] * 5,
            "_wicket": [1, 0, 1, 0, 0, 0] * 5,
            "exp_runs_bowl": [2.5, 2.5, 2.5, 2.5, 2.5, 2.5] * 5,
            "prob_wicket_bowl": [0.08, 0.08, 0.08, 0.08, 0.08, 0.08] * 5,
            "bowling_pressure_score": [0.8, 0.4, 0.9, 0.3, 0.5, 0.7] * 5,
            "bowling_pressure_band": ["high", "medium", "high", "low", "medium", "high"] * 5,
            "wpa_bowler": [0.06, -0.04, 0.08, -0.08, 0.02, -0.03] * 5,
        })
        mc = MonteCarloSimulationEngine(n_iterations=20, random_state=42)
        res = mc.simulate_bowling_stability(df_bowl, [201, 202])

        assert len(res) == 2
        for col in ["bowler_id", "prob_top_10", "prob_top_5", "median_sim_rank", "rank_ci_95", "stability_score"]:
            assert col in res.columns
        assert (res["prob_top_10"] >= 0.0).all() and (res["prob_top_10"] <= 1.0).all()
        # Bowler 201 has wickets, higher RPA and WPA, should rank better (lower index)
        rank_201 = res.loc[res["bowler_id"] == 201, "median_sim_rank"].values[0]
        rank_202 = res.loc[res["bowler_id"] == 202, "median_sim_rank"].values[0]
        assert rank_201 <= rank_202


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
