"""
PRESSURE-X — Module 4: Match Reconstruction Engine
====================================================
Reconstructs the full delivery-by-delivery match state for every
delivery in ball_by_ball.csv.

For each delivery it calculates and appends:
  - Innings running totals (runs, wickets, legal balls, extras)
  - Balls remaining, runs required, required run rate, current run rate
  - Innings phase (powerplay / middle / death)
  - Legal ball flag (excludes wides and no-balls from over count)
  - Delivery sequence index within the innings
  - Flags: is_super_over, is_dls, is_abandoned

The output is a clean, delivery-enriched Parquet dataset — the foundation
for every subsequent PRESSURE-X model.

Author: PRESSURE-X Engineering
Version: 1.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from ingestion.loader import RawDataPack
from reconstruction.entity_resolver import EntityRegistry

logger = logging.getLogger(__name__)


INNINGS_MAX_LEGAL_BALLS = 120   # 20 overs × 6 legal balls


# ---------------------------------------------------------------------------
# Column-name registry (resolved once per file load)
# ---------------------------------------------------------------------------


class ColumnMap:
    """Resolves column names from the actual DataFrame columns."""

    def __init__(self, df: pd.DataFrame):
        self._cols = set(df.columns)

    def get(self, *candidates) -> Optional[str]:
        for c in candidates:
            if c in self._cols:
                return c
        return None

    def require(self, *candidates) -> str:
        c = self.get(*candidates)
        if c is None:
            raise KeyError(f"Could not find any of: {candidates}")
        return c


# ---------------------------------------------------------------------------
# Match State Engine
# ---------------------------------------------------------------------------


class MatchReconstructionEngine:
    """
    Module 4: Match Reconstruction
    Takes the raw ball_by_ball DataFrame and enriches every delivery
    with computed match-state fields.
    """

    def __init__(self, config_path: str | Path = "config/data_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)
        self.pw_overs = self.config["cricket_rules"]["powerplay_overs"]
        self.mid_overs = self.config["cricket_rules"]["middle_overs"]
        self.death_overs = self.config["cricket_rules"]["death_overs"]
        self.max_wickets = self.config["cricket_rules"]["max_wickets"]
        self.output_dir = Path(self.config["paths"]["processed"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Legal delivery detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_legal(row: pd.Series, extra_col: Optional[str], wide_col: Optional[str], noball_col: Optional[str]) -> bool:
        """A delivery is legal (counts towards the over) iff it is not a wide or no-ball."""
        # Prefer a dedicated is_wide / is_no_ball column
        if wide_col and row.get(wide_col, 0) in (1, True, "1", "True", "yes"):
            return False
        if noball_col and row.get(noball_col, 0) in (1, True, "1", "True", "yes"):
            return False
        # Fall back to extra_type string
        if extra_col:
            et = str(row.get(extra_col, "") or "").lower().strip()
            # Wides, overstep (no-balls), and other (non-overstep no-balls) require re-bowls
            if et in ("wide", "wides", "no ball", "no_ball", "noball", "nb", "wd", "overstep", "other"):
                return False
        return True

    # ------------------------------------------------------------------
    # Innings phase classification
    # ------------------------------------------------------------------

    def _phase(self, over: int) -> str:
        """Return 'powerplay', 'middle', or 'death' for a given over (1-indexed)."""
        if self.pw_overs[0] <= over <= self.pw_overs[1]:
            return "powerplay"
        if self.mid_overs[0] <= over <= self.mid_overs[1]:
            return "middle"
        return "death"

    # ------------------------------------------------------------------
    # Core reconstruction loop
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        pack: RawDataPack,
        registry: EntityRegistry,
    ) -> pd.DataFrame:
        bbb = pack.ball_by_ball.copy()
        cm = ColumnMap(bbb)

        # ------ resolve critical column names ------
        # Actual column names confirmed from live schema audit:
        # match_id, innings, overid, over_ball_num, runs, is_wicket,
        # extras_type, extras1_runs, extras_2_runs, striker_id, bowler_id
        match_col    = cm.require("match_id", "matchId")
        innings_col  = cm.require("innings", "innings_id", "inningsId", "innings_number")
        over_col     = cm.require("overid", "over_number", "over", "overNumber")
        ball_col     = cm.require("over_ball_num", "ball_number", "ball", "ballNumber", "delivery")
        runs_bat_col = cm.get("runs", "runs_batter", "batsman_runs", "batter_runs", "runs_off_bat")
        ext1_col     = cm.get("extras1_runs")
        ext2_col     = cm.get("extras_2_runs")
        runs_ext_col = cm.get("extra_runs", "extras", "runs_extras")
        runs_tot_col = cm.get("runs_total", "total_runs")
        wicket_col   = cm.get("is_wicket", "wicket", "isWicket", "wicket_fell")
        extra_col    = cm.get("extras_type", "extra_type", "extraType")
        wide_col     = cm.get("is_wide", "wide", "isWide")
        noball_col   = cm.get("is_no_ball", "no_ball", "isNoBall")
        striker_col  = cm.get("striker_id", "striker", "batsman_id", "batter_id")
        bowler_col   = cm.get("bowler_id", "bowler")

        # ------ derive per-ball runs ------
        # In this dataset: 'runs' = batter runs; extras in 'extras1_runs' / 'extras_2_runs'
        ext1_col = cm.get("extras1_runs")
        ext2_col = cm.get("extras_2_runs")

        if runs_tot_col:
            bbb["_runs_this_ball"] = pd.to_numeric(bbb[runs_tot_col], errors="coerce").fillna(0)
        elif runs_bat_col:
            bat_runs = pd.to_numeric(bbb[runs_bat_col], errors="coerce").fillna(0)
            ext_runs = pd.Series(0.0, index=bbb.index)
            if ext1_col:
                ext_runs += pd.to_numeric(bbb[ext1_col], errors="coerce").fillna(0)
            if ext2_col:
                ext_runs += pd.to_numeric(bbb[ext2_col], errors="coerce").fillna(0)
            if not ext1_col and not ext2_col and runs_ext_col:
                ext_runs += pd.to_numeric(bbb[runs_ext_col], errors="coerce").fillna(0)
            bbb["_runs_this_ball"] = bat_runs + ext_runs
        else:
            bbb["_runs_this_ball"] = 0
            logger.warning("Could not locate runs column — using 0")

        # ------ batter runs (attributed only to striker) ------
        if runs_bat_col:
            bbb["runs_batter"] = pd.to_numeric(bbb[runs_bat_col], errors="coerce").fillna(0)
        else:
            bbb["runs_batter"] = 0

        # ------ is_legal flag ------
        bbb["is_legal_delivery"] = bbb.apply(
            lambda r: self._is_legal(r, extra_col, wide_col, noball_col), axis=1
        )

        # ------ wicket flag ------
        if wicket_col:
            bbb["_wicket"] = pd.to_numeric(bbb[wicket_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        else:
            bbb["_wicket"] = 0

        # ------ innings phase ------
        bbb["innings_phase"] = bbb[over_col].apply(
            lambda o: self._phase(int(o)) if pd.notna(o) else "unknown"
        )

        # ------ sort correctly ------
        sort_cols = [match_col, innings_col, over_col, ball_col]
        bbb = bbb.sort_values(sort_cols).reset_index(drop=True)

        # ------ running match-state calculation ------
        logger.info("Computing delivery-level match states …")

        # Initialise output columns
        bbb["innings_runs"] = 0
        bbb["innings_wickets"] = 0
        bbb["innings_legal_balls"] = 0
        bbb["innings_ball_index"] = 0  # delivery sequence within innings

        # Group by (match, innings) and compute running totals
        groups = bbb.groupby([match_col, innings_col], sort=False)

        innings_runs_arr     = np.zeros(len(bbb), dtype=np.int32)
        innings_wickets_arr  = np.zeros(len(bbb), dtype=np.int32)
        innings_legal_arr    = np.zeros(len(bbb), dtype=np.int32)
        innings_ball_idx_arr = np.zeros(len(bbb), dtype=np.int32)

        for (mid, inn), grp in groups:
            idx = grp.index.to_numpy()
            runs_arr    = grp["_runs_this_ball"].to_numpy()
            wicket_arr  = grp["_wicket"].to_numpy()
            legal_arr   = grp["is_legal_delivery"].to_numpy()

            cum_runs    = np.cumsum(runs_arr)
            cum_wickets = np.cumsum(wicket_arr)
            cum_legal   = np.cumsum(legal_arr)

            # State BEFORE this delivery (shift right by 1)
            innings_runs_arr[idx]     = np.concatenate([[0], cum_runs[:-1]])
            innings_wickets_arr[idx]  = np.concatenate([[0], cum_wickets[:-1]])
            innings_legal_arr[idx]    = np.concatenate([[0], cum_legal[:-1]])
            innings_ball_idx_arr[idx] = np.arange(len(idx))

        bbb["innings_runs"]        = innings_runs_arr
        bbb["innings_wickets"]     = innings_wickets_arr
        bbb["innings_legal_balls"] = innings_legal_arr
        bbb["innings_ball_index"]  = innings_ball_idx_arr

        # ------ wickets remaining ------
        bbb["wickets_remaining"] = self.max_wickets - bbb["innings_wickets"]

        # ------ balls remaining ------
        bbb["balls_remaining"] = (INNINGS_MAX_LEGAL_BALLS - bbb["innings_legal_balls"]).clip(lower=0)

        # ------ current run rate (CRR) ------
        # CRR = runs scored / legal balls bowled × 6  (runs per over)
        bbb["current_run_rate"] = np.where(
            bbb["innings_legal_balls"] > 0,
            (bbb["innings_runs"] / bbb["innings_legal_balls"]) * 6,
            0.0,
        )

        # ------ target / required run rate (only for 2nd innings) ------
        # We need to look up the first-innings total for each match
        logger.info("Computing required run rates for 2nd innings …")
        bbb = self._attach_targets(bbb, match_col, innings_col)

        # ------ match-level flags from registry ------
        bbb["is_super_over"] = bbb[match_col].isin(registry.super_over_match_ids)
        bbb["is_dls"]        = bbb[match_col].isin(registry.dls_match_ids)
        bbb["is_abandoned"]  = bbb[match_col].isin(registry.abandoned_match_ids)
        bbb["competition_format"] = bbb[match_col].map(
            {mid: m.competition_format for mid, m in registry.matches.items()}
        ).astype(str).fillna("Unknown")

        logger.info("Match reconstruction complete: %d deliveries enriched", len(bbb))
        return bbb

    # ------------------------------------------------------------------
    # Target attachment (2nd innings RRR)
    # ------------------------------------------------------------------

    def _attach_targets(
        self,
        bbb: pd.DataFrame,
        match_col: str,
        innings_col: str,
    ) -> pd.DataFrame:
        """Compute first-innings totals and attach target / RRR to 2nd innings."""

        # Identify innings numbers
        innings_vals = sorted(bbb[innings_col].dropna().unique())
        if len(innings_vals) < 2:
            bbb["target"] = np.nan
            bbb["runs_required"] = np.nan
            bbb["required_run_rate"] = np.nan
            return bbb

        first_inn_id = innings_vals[0]
        second_inn_id = innings_vals[1]

        # First innings final total per match
        first_inn = bbb[bbb[innings_col] == first_inn_id].copy()
        totals = (
            first_inn.groupby(match_col)["_runs_this_ball"]
            .sum()
            .rename("first_innings_total")
        )

        # Merge target into second innings rows
        bbb = bbb.merge(totals, on=match_col, how="left")

        # target = first innings total + 1
        bbb["target"] = np.where(
            bbb[innings_col] == second_inn_id,
            bbb["first_innings_total"] + 1,
            np.nan,
        )

        # Runs required at moment of delivery (before this ball is bowled)
        bbb["runs_required"] = np.where(
            bbb[innings_col] == second_inn_id,
            (bbb["target"] - bbb["innings_runs"]).clip(lower=0),
            np.nan,
        )

        # Required run rate
        bbb["required_run_rate"] = np.where(
            (bbb[innings_col] == second_inn_id) & (bbb["balls_remaining"] > 0),
            (bbb["runs_required"] / bbb["balls_remaining"]) * 6,
            np.nan,
        )

        bbb = bbb.drop(columns=["first_innings_total"], errors="ignore")
        return bbb

    # ------------------------------------------------------------------
    # Save to Parquet
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, filename: str = "deliveries_enriched.parquet") -> Path:
        out_path = self.output_dir / filename
        compression = self.config["output"]["parquet_compression"]
        df.to_parquet(out_path, engine="pyarrow", compression=compression, index=False)
        logger.info("Saved enriched deliveries → %s  (%d rows)", out_path, len(df))
        return out_path


def reconstruct_matches(
    pack: RawDataPack,
    registry: EntityRegistry,
    config_path: str | Path = "config/data_config.yaml",
    save: bool = True,
) -> pd.DataFrame:
    engine = MatchReconstructionEngine(config_path)
    df = engine.reconstruct(pack, registry)
    if save:
        engine.save(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.loader import load_raw_data
    from reconstruction.entity_resolver import resolve_entities

    pack = load_raw_data()
    registry = resolve_entities(pack)
    df = reconstruct_matches(pack, registry, save=True)
    print("\nSample enriched columns:")
    cols = ["innings_runs", "innings_wickets", "innings_legal_balls",
            "wickets_remaining", "balls_remaining", "current_run_rate",
            "runs_required", "required_run_rate", "innings_phase",
            "competition_format"]
    available = [c for c in cols if c in df.columns]
    print(df[available].head(10).to_string())
