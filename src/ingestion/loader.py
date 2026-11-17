"""
PRESSURE-X — Module 1: Data Ingestion
======================================
Loads all six CSV files from raw/, validates file presence and
encoding, and returns typed DataFrames ready for downstream modules.

Author: PRESSURE-X Engineering
Version: 1.0.0
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema definitions — expected dtypes for each file
# ---------------------------------------------------------------------------

SCHEMA: Dict[str, Dict[str, str]] = {
    "ball_by_ball": {
        # Will be inferred; we enforce critical columns exist
        "required_columns": [
            "match_id",
            "innings_id",
            "over_number",
            "ball_number",
            "striker_id",
            "non_striker_id",
            "bowler_id",
        ],
    },
    "matches": {
        "required_columns": [
            "match_id",
            "competition_id",
        ],
    },
    "competitions": {
        "required_columns": [
            "competition_id",
        ],
    },
    "teams": {
        "required_columns": [
            "team_id",
        ],
    },
    "playing_eleven": {
        "required_columns": [
            "match_id",
            "player_id",
        ],
    },
    "competition_team_player_mapping": {
        "required_columns": [
            "competition_id",
            "player_id",
        ],
    },
}


@dataclass
class RawDataPack:
    """Container for all six loaded DataFrames."""

    ball_by_ball: pd.DataFrame = field(default_factory=pd.DataFrame)
    matches: pd.DataFrame = field(default_factory=pd.DataFrame)
    competitions: pd.DataFrame = field(default_factory=pd.DataFrame)
    teams: pd.DataFrame = field(default_factory=pd.DataFrame)
    playing_eleven: pd.DataFrame = field(default_factory=pd.DataFrame)
    competition_team_player_mapping: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        lines = ["RawDataPack summary:"]
        for name in self.__dataclass_fields__:
            df = getattr(self, name)
            lines.append(f"  {name:40s} → {len(df):>8,} rows × {len(df.columns):>3} cols")
        return "\n".join(lines)


class DataIngestionEngine:
    """
    Module 1: Data Ingestion
    Reads all CSVs, validates file presence, checks encoding,
    and returns a RawDataPack.
    """

    def __init__(self, config_path: str | Path = "config/data_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.raw_dir = Path(self.config["paths"]["raw"])

    def _load_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _resolve_path(self, filename: str) -> Path:
        return self.raw_dir / filename

    def _load_csv(self, name: str, filename: str) -> pd.DataFrame:
        path = self._resolve_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

        logger.info("Loading %s from %s …", name, path)

        # Try UTF-8 first, fall back to latin-1 (common in sports data exports)
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=encoding, low_memory=False)
                logger.info("  ✓ %s loaded: %d rows, %d cols (encoding=%s)",
                            name, len(df), len(df.columns), encoding)
                return df
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode {path} with any supported encoding.")

    def _validate_columns(self, name: str, df: pd.DataFrame) -> None:
        schema = SCHEMA.get(name, {})
        required = schema.get("required_columns", [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("⚠ %s is missing columns: %s", name, missing)
            logger.warning("  Available columns: %s", list(df.columns))
            # Don't raise — column names may differ slightly in the actual data;
            # the validation module will give a full column audit.

    def load(self) -> RawDataPack:
        """Load all six CSV files and return a RawDataPack."""
        cfg = self.config["files"]

        pack = RawDataPack()

        pack.ball_by_ball = self._load_csv("ball_by_ball", cfg["ball_by_ball"])
        pack.matches = self._load_csv("matches", cfg["matches"])
        pack.competitions = self._load_csv("competitions", cfg["competitions"])
        pack.teams = self._load_csv("teams", cfg["teams"])
        pack.playing_eleven = self._load_csv("playing_eleven", cfg["playing_eleven"])
        pack.competition_team_player_mapping = self._load_csv(
            "competition_team_player_mapping", cfg["competition_team_player_mapping"]
        )

        # Column validation (soft warnings, not hard failures)
        for name in SCHEMA:
            df = getattr(pack, name)
            self._validate_columns(name, df)

        logger.info("\n%s", pack.summary())
        return pack


def load_raw_data(
    config_path: str | Path = "config/data_config.yaml",
) -> RawDataPack:
    """Convenience wrapper: load all six CSV files."""
    engine = DataIngestionEngine(config_path)
    return engine.load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    pack = load_raw_data()
    print(pack.summary())
