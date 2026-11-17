"""
PRESSURE-X — Automated Tests: Phase 1
=======================================
Tests for cricket rule logic, data transformations and
key reconstruction invariants.

Run:  pytest tests/test_phase1.py -v

Author: PRESSURE-X Engineering
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reconstruction.match_reconstructor import MatchReconstructionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    cfg = Path(__file__).parent.parent / "config" / "data_config.yaml"
    return MatchReconstructionEngine(cfg)


@pytest.fixture
def simple_innings_df():
    """12 deliveries: 2 overs, 1 wicket, 1 wide."""
    return pd.DataFrame({
        "match_id":    [1] * 12,
        "innings_id":  [1] * 12,
        "over_number": [1, 1, 1, 1, 1, 1, 1,   2, 2, 2, 2, 2],
        "ball_number": [1, 2, 3, 4, 5, 6, 6.1,  1, 2, 3, 4, 5],
        "extra_type":  ["", "", "", "", "", "", "wide",
                        "", "", "", "", ""],
        "runs_batter": [4, 0, 1, 6, 0, 2, 0,
                        1, 0, 4, 0, 3],
        "extra_runs":  [0, 0, 0, 0, 0, 0, 1,
                        0, 0, 0, 0, 0],
        "runs_total":  [4, 0, 1, 6, 0, 2, 1,
                        1, 0, 4, 0, 3],
        "is_wicket":   [0, 0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0],
    })


# ---------------------------------------------------------------------------
# Tests: legal delivery detection
# ---------------------------------------------------------------------------

class TestLegalDeliveries:

    def test_wide_is_not_legal(self, engine):
        row = pd.Series({"extra_type": "wide", "runs_batter": 0})
        assert engine._is_legal(row, "extra_type", None, None) is False

    def test_noball_is_not_legal(self, engine):
        row = pd.Series({"extra_type": "no_ball"})
        assert engine._is_legal(row, "extra_type", None, None) is False

    def test_normal_delivery_is_legal(self, engine):
        row = pd.Series({"extra_type": "", "runs_batter": 4})
        assert engine._is_legal(row, "extra_type", None, None) is True

    def test_bye_is_legal(self, engine):
        """Byes count as a legal delivery — they just don't credit the batter."""
        row = pd.Series({"extra_type": "bye"})
        assert engine._is_legal(row, "extra_type", None, None) is True

    def test_leg_bye_is_legal(self, engine):
        row = pd.Series({"extra_type": "leg_bye"})
        assert engine._is_legal(row, "extra_type", None, None) is True


# ---------------------------------------------------------------------------
# Tests: innings phase
# ---------------------------------------------------------------------------

class TestPhaseClassification:

    def test_powerplay(self, engine):
        for over in range(1, 7):
            assert engine._phase(over) == "powerplay"

    def test_middle(self, engine):
        for over in range(7, 16):
            assert engine._phase(over) == "middle"

    def test_death(self, engine):
        for over in range(16, 21):
            assert engine._phase(over) == "death"

    def test_super_over_phase(self, engine):
        """Over 21 (super over) should still classify — classified as death."""
        assert engine._phase(21) == "death"


# ---------------------------------------------------------------------------
# Tests: running totals invariants
# ---------------------------------------------------------------------------

class TestRunningTotals:

    def test_first_delivery_state_is_zero(self, simple_innings_df):
        """Before ball 1 is bowled, scores should be 0."""
        from ingestion.loader import RawDataPack
        from reconstruction.entity_resolver import EntityRegistry
        cfg = Path(__file__).parent.parent / "config" / "data_config.yaml"

        engine = MatchReconstructionEngine(cfg)

        # Minimal pack and registry for a synthetic test
        pack = RawDataPack(ball_by_ball=simple_innings_df)
        registry = EntityRegistry()

        df = engine.reconstruct(pack, registry)
        first_ball = df.iloc[0]
        assert first_ball["innings_runs"] == 0
        assert first_ball["innings_wickets"] == 0
        assert first_ball["innings_legal_balls"] == 0

    def test_wickets_accumulate_correctly(self, simple_innings_df):
        from ingestion.loader import RawDataPack
        from reconstruction.entity_resolver import EntityRegistry
        cfg = Path(__file__).parent.parent / "config" / "data_config.yaml"

        engine = MatchReconstructionEngine(cfg)
        pack = RawDataPack(ball_by_ball=simple_innings_df)
        registry = EntityRegistry()

        df = engine.reconstruct(pack, registry)
        # The wicket is on ball 5 of over 1 (index 4 in input, 0-based)
        # State BEFORE ball 5 → 0 wickets
        pre_wicket = df[df["innings_ball_index"] == 4].iloc[0]
        assert pre_wicket["innings_wickets"] == 0
        # State BEFORE ball 6 (after wicket fell) → 1 wicket
        post_wicket = df[df["innings_ball_index"] == 5].iloc[0]
        assert post_wicket["innings_wickets"] == 1

    def test_wide_does_not_increment_legal_balls(self, simple_innings_df):
        from ingestion.loader import RawDataPack
        from reconstruction.entity_resolver import EntityRegistry
        cfg = Path(__file__).parent.parent / "config" / "data_config.yaml"

        engine = MatchReconstructionEngine(cfg)
        pack = RawDataPack(ball_by_ball=simple_innings_df)
        registry = EntityRegistry()

        df = engine.reconstruct(pack, registry)
        wide_row = df[df["innings_ball_index"] == 6].iloc[0]  # the wide
        # Preceding state must still show 6 legal balls (the whole of over 1)
        # (the wide is ball index 6 — before it, 6 legal balls have been bowled)
        assert wide_row["innings_legal_balls"] == 6

    def test_wickets_remaining_decrements(self, simple_innings_df):
        from ingestion.loader import RawDataPack
        from reconstruction.entity_resolver import EntityRegistry
        cfg = Path(__file__).parent.parent / "config" / "data_config.yaml"

        engine = MatchReconstructionEngine(cfg)
        pack = RawDataPack(ball_by_ball=simple_innings_df)
        registry = EntityRegistry()

        df = engine.reconstruct(pack, registry)
        last = df.iloc[-1]
        # 1 wicket fell → wickets_remaining should be 9
        assert last["wickets_remaining"] == 9


# ---------------------------------------------------------------------------
# Tests: target / RRR
# ---------------------------------------------------------------------------

class TestTargetAndRRR:

    def test_first_innings_has_no_rrr(self):
        df = pd.DataFrame({
            "match_id": [1] * 6,
            "innings_id": [1] * 6,
            "over_number": [1] * 6,
            "ball_number": list(range(1, 7)),
            "extra_type": [""] * 6,
            "runs_batter": [4, 6, 1, 0, 2, 0],
            "runs_total":  [4, 6, 1, 0, 2, 0],
            "is_wicket":   [0] * 6,
        })
        from ingestion.loader import RawDataPack
        from reconstruction.entity_resolver import EntityRegistry
        cfg = Path(__file__).parent.parent / "config" / "data_config.yaml"

        engine = MatchReconstructionEngine(cfg)
        pack = RawDataPack(ball_by_ball=df)
        registry = EntityRegistry()

        out = engine.reconstruct(pack, registry)
        assert out["required_run_rate"].isna().all()

    def test_second_innings_has_target(self):
        """2-innings match; 2nd innings must have a valid target."""
        df = pd.DataFrame({
            "match_id":    [1] * 12,
            "innings_id":  [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "over_number": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "ball_number": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "extra_type":  [""] * 12,
            "runs_batter": [4, 6, 2, 0, 1, 3, 1, 0, 2, 4, 0, 6],
            "runs_total":  [4, 6, 2, 0, 1, 3, 1, 0, 2, 4, 0, 6],
            "is_wicket":   [0] * 12,
        })
        from ingestion.loader import RawDataPack
        from reconstruction.entity_resolver import EntityRegistry
        cfg = Path(__file__).parent.parent / "config" / "data_config.yaml"

        engine = MatchReconstructionEngine(cfg)
        pack = RawDataPack(ball_by_ball=df)
        registry = EntityRegistry()

        out = engine.reconstruct(pack, registry)
        second_inn = out[out["innings_id"] == 2]
        # First innings scored 16, so target = 17
        assert second_inn["target"].dropna().iloc[0] == 17


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
