"""
PRESSURE-X — Automated Audit Integrity & Consistency Tests
===========================================================
Validates mathematical accounting, absence of data leakage,
ranking index integrity, and probability bounds.

Author: PRESSURE-X Engineering
Version: 2.2.0
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation.split_utils import get_chronological_holdout_split, get_chronological_kfold_splits


class TestDataAccountingConsistency:

    def test_delivery_accounting_exact_reconciliation(self):
        """Verifies that legal + illegal + unclassified equals total deliveries exactly."""
        raw_path = Path("data/raw/ball_by_ball.csv")
        if not raw_path.exists():
            pytest.skip("Raw ball_by_ball.csv not found")

        df = pd.read_csv(raw_path, low_memory=False)
        total_deliveries = len(df)
        assert total_deliveries == 200999

        extra_col = "extras_type"
        extra_vals = df[extra_col].fillna("").astype(str).str.lower().str.strip()

        illegal_types = {"wide", "wides", "noball", "no_ball", "overstep", "other"}
        legal_extra_types = {"bye", "byes", "legbye", "legbyes", "penalty"}
        no_extra_types = {"", "0", "nan", "none", "null", "false"}

        n_illegal = int(extra_vals.isin(illegal_types).sum())
        n_legal_extras = int(extra_vals.isin(legal_extra_types).sum())
        n_no_extra = int(extra_vals.isin(no_extra_types).sum())
        n_legal = n_no_extra + n_legal_extras

        assert n_illegal == 8554  # 7714 wides + 576 overstep + 264 other
        assert n_legal == 192445  # 189167 no extra + 2748 legbyes + 529 byes + 1 penalty
        assert n_legal + n_illegal == total_deliveries


class TestLeakageFreeTemporalSplits:

    def test_chronological_holdout_strict_disjoint_and_ordered(self):
        """Verifies zero match overlap and strict max(train_dates) < min(test_dates)."""
        df = pd.DataFrame({
            "match_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "match_date": ["2022-04-01", "2022-04-01",
                           "2022-04-02", "2022-04-02",
                           "2023-05-01", "2023-05-01",
                           "2023-05-01", "2023-05-01",
                           "2024-03-20", "2024-03-20",
                           "2024-05-10", "2024-05-10"],
        })

        tr_idx, te_idx, meta = get_chronological_holdout_split(df, train_split_ratio=0.80)
        train_matches = set(meta["train_matches"])
        test_matches = set(meta["test_matches"])

        assert train_matches.isdisjoint(test_matches)
        assert meta["train_date_range"][1] < meta["test_date_range"][0]

    def test_chronological_kfold_splits_all_disjoint(self):
        """Proves every fold is completely disjoint with 0 cross-fold contamination."""
        df = pd.DataFrame({
            "match_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "match_date": [f"2023-0{i+1}-01" for i in range(10)],
        })
        splits = get_chronological_kfold_splits(df, n_splits=5)
        assert len(splits) == 5
        for tr_idx, val_idx in splits:
            tr_m = set(df.iloc[tr_idx]["match_id"])
            val_m = set(df.iloc[val_idx]["match_id"])
            assert tr_m.isdisjoint(val_m)


class TestRankingIntegrity:

    def test_leaderboard_consecutive_ranks_and_bounds(self):
        """Verifies that published rankings have ranks 1..10, valid BPI, and no NaNs."""
        top_bat_path = Path("outputs/rankings/top_10_batters.csv")
        top_bowl_path = Path("outputs/rankings/top_10_bowlers.csv")

        if top_bat_path.exists():
            df_bat = pd.read_csv(top_bat_path)
            assert df_bat["rank"].tolist() == list(range(1, len(df_bat) + 1))
            assert not df_bat["batting_pressure_index"].isna().any()
            assert not df_bat["striker_id"].isna().any()
            assert df_bat["striker_id"].nunique() == len(df_bat)
            if "prob_top_10" in df_bat.columns:
                assert (df_bat["prob_top_10"] >= 0.0).all() and (df_bat["prob_top_10"] <= 1.0).all()

        if top_bowl_path.exists():
            df_bowl = pd.read_csv(top_bowl_path)
            assert df_bowl["rank"].tolist() == list(range(1, len(df_bowl) + 1))
            assert not df_bowl["bowling_pressure_index"].isna().any()
            assert not df_bowl["bowler_id"].isna().any()
            assert df_bowl["bowler_id"].nunique() == len(df_bowl)
            if "prob_top_10" in df_bowl.columns:
                assert (df_bowl["prob_top_10"] >= 0.0).all() and (df_bowl["prob_top_10"] <= 1.0).all()
