"""
PRESSURE-X — Module 3: Entity Resolution
==========================================
Builds a unified entity registry linking:
  - Players ↔ Matches ↔ Competitions ↔ Teams
  - Derives competition format (IPL / T20I / Other)
  - Flags abandoned / incomplete / super-over matches
  - Builds player competition eligibility maps
  - Outputs a resolved EntityRegistry object

Author: PRESSURE-X Engineering
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd
import yaml

from ingestion.loader import RawDataPack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CompetitionMeta:
    competition_id: any
    name: str
    format: str          # "IPL" | "T20I" | "Other_T20" | "Other"
    season: Optional[str]
    country: Optional[str]
    match_ids: Set[any] = field(default_factory=set)
    team_ids: Set[any] = field(default_factory=set)
    player_ids: Set[any] = field(default_factory=set)


@dataclass
class MatchMeta:
    match_id: any
    competition_id: any
    competition_format: str
    team1_id: Optional[any]
    team2_id: Optional[any]
    venue: Optional[str]
    date: Optional[str]
    winner_id: Optional[any]
    result_type: str           # "normal" | "tie" | "no_result" | "super_over" | "dls"
    is_abandoned: bool
    is_super_over: bool
    is_dls: bool
    innings_count: int
    player_ids: Set[any] = field(default_factory=set)


@dataclass
class PlayerMeta:
    player_id: any
    name: str
    batting_style: Optional[str]
    bowling_style: Optional[str]
    competition_ids: Set[any] = field(default_factory=set)
    match_ids: Set[any] = field(default_factory=set)
    team_ids: Set[any] = field(default_factory=set)


@dataclass
class EntityRegistry:
    competitions: Dict[any, CompetitionMeta] = field(default_factory=dict)
    matches: Dict[any, MatchMeta] = field(default_factory=dict)
    players: Dict[any, PlayerMeta] = field(default_factory=dict)
    teams: Dict[any, str] = field(default_factory=dict)   # team_id → team_name

    # Derived convenience sets
    ipl_match_ids: Set[any] = field(default_factory=set)
    t20i_match_ids: Set[any] = field(default_factory=set)
    abandoned_match_ids: Set[any] = field(default_factory=set)
    super_over_match_ids: Set[any] = field(default_factory=set)
    dls_match_ids: Set[any] = field(default_factory=set)

    def summary(self) -> str:
        lines = [
            "EntityRegistry summary:",
            f"  Competitions : {len(self.competitions):,}",
            f"  Matches      : {len(self.matches):,}",
            f"  Players      : {len(self.players):,}",
            f"  Teams        : {len(self.teams):,}",
            f"  IPL matches  : {len(self.ipl_match_ids):,}",
            f"  T20I matches : {len(self.t20i_match_ids):,}",
            f"  Abandoned    : {len(self.abandoned_match_ids):,}",
            f"  Super overs  : {len(self.super_over_match_ids):,}",
            f"  DLS/revised  : {len(self.dls_match_ids):,}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entity Resolution Engine
# ---------------------------------------------------------------------------


class EntityResolutionEngine:
    """
    Module 3: Entity Resolution
    Builds an EntityRegistry from the six raw DataFrames.
    """

    def __init__(self, config_path: str | Path = "config/data_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)
        self.ipl_kw = [k.lower() for k in self.config["formats"]["ipl_keywords"]]
        self.t20i_kw = [k.lower() for k in self.config["formats"]["t20i_keywords"]]
        self.t20_kw = [k.lower() for k in self.config["formats"]["t20_keywords"]]

    # ------------------------------------------------------------------
    # Column-name resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _col(df: pd.DataFrame, *candidates) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    @staticmethod
    def _val(row: pd.Series, *candidates) -> Optional[any]:
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                return row[c]
        return None

    # ------------------------------------------------------------------
    # Competition format classification
    # ------------------------------------------------------------------

    def _classify_format(self, name: str) -> str:
        nl = name.lower()
        if any(k in nl for k in self.ipl_kw):
            return "IPL"
        if any(k in nl for k in self.t20i_kw):
            return "T20I"
        if any(k in nl for k in self.t20_kw):
            return "Other_T20"
        return "Other"

    # ------------------------------------------------------------------
    # Match result classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_result(row: pd.Series) -> tuple[str, bool, bool, bool]:
        """Return (result_type, is_abandoned, is_super_over, is_dls)."""
        result_col_val = ""
        for col in ["result", "match_result", "outcome", "matchResult"]:
            if col in row.index and pd.notna(row[col]):
                result_col_val = str(row[col]).lower()
                break

        is_abandoned = any(k in result_col_val for k in ["abandon", "no result", "cancelled"])
        is_super_over = any(k in result_col_val for k in ["super over", "superover"])
        is_dls = any(k in result_col_val for k in ["d/l", "dls", "duckworth", "revised"])

        if is_abandoned:
            result_type = "no_result"
        elif is_super_over:
            result_type = "super_over"
        elif is_dls:
            result_type = "dls"
        elif "tie" in result_col_val:
            result_type = "tie"
        else:
            result_type = "normal"

        return result_type, is_abandoned, is_super_over, is_dls

    # ------------------------------------------------------------------
    # Build competitions registry
    # ------------------------------------------------------------------

    def _build_competitions(self, df: pd.DataFrame) -> Dict[any, CompetitionMeta]:
        reg: Dict[any, CompetitionMeta] = {}
        id_col = self._col(df, "comp_id", "competition_id", "competitionId", "id")
        name_col = self._col(df, "comp_name", "competition_name", "name", "competitionName", "competition")
        format_col = self._col(df, "match_format", "format", "competition_format")
        season_col = self._col(df, "season", "year", "edition")
        country_col = self._col(df, "country", "host_country", "venue_country")

        for _, row in df.iterrows():
            cid = self._val(row, "comp_id", "competition_id", "competitionId", "id")
            if cid is None:
                continue
            name = str(self._val(row, "comp_name", "competition_name", "name", "competitionName") or "Unknown")
            # 1. IPL must be checked first
            nl = name.lower()
            fmt_str = str(row[format_col]).upper() if format_col and pd.notna(row.get(format_col)) else ""
            if any(k in nl for k in self.ipl_kw) or "IPL" in fmt_str:
                fmt = "IPL"
            elif any(k in nl for k in self.t20i_kw) or "T20I" in fmt_str:
                fmt = "T20I"
            elif any(k in nl for k in self.t20_kw) or "T20" in fmt_str:
                fmt = "Other_T20"
            else:
                fmt = self._classify_format(name)
            reg[cid] = CompetitionMeta(
                competition_id=cid,
                name=name,
                format=fmt,
                season=self._val(row, "season", "year", "edition"),
                country=self._val(row, "country", "host_country"),
            )
        return reg

    # ------------------------------------------------------------------
    # Build matches registry
    # ------------------------------------------------------------------

    def _build_matches(
        self,
        matches_df: pd.DataFrame,
        competition_reg: Dict[any, CompetitionMeta],
    ) -> Dict[any, MatchMeta]:
        reg: Dict[any, MatchMeta] = {}

        for _, row in matches_df.iterrows():
            mid = self._val(row, "match_id", "matchId", "id")
            if mid is None:
                continue

            # Actual column is comp_id not competition_id
            cid = self._val(row, "comp_id", "competition_id", "competitionId")
            comp_fmt = competition_reg[cid].format if cid in competition_reg else "Unknown"

            result_type, is_abandoned, is_super_over, is_dls = self._classify_result(row)

            # Actual columns: home_team_id, away_team_id, win_team_id, match_date
            reg[mid] = MatchMeta(
                match_id=mid,
                competition_id=cid,
                competition_format=comp_fmt,
                team1_id=self._val(row, "home_team_id", "team1_id", "team1"),
                team2_id=self._val(row, "away_team_id", "team2_id", "team2"),
                venue=self._val(row, "venue_id", "venue", "ground", "stadium", "venue_name"),
                date=self._val(row, "match_date", "date", "matchDate"),
                winner_id=self._val(row, "win_team_id", "winner_id", "winning_team_id", "winner"),
                result_type=result_type,
                is_abandoned=is_abandoned,
                is_super_over=is_super_over,
                is_dls=is_dls,
                innings_count=0,
            )

            # Back-populate competition match_id sets
            if cid in competition_reg:
                competition_reg[cid].match_ids.add(mid)

        return reg

    # ------------------------------------------------------------------
    # Build teams registry
    # ------------------------------------------------------------------

    def _build_teams(self, teams_df: pd.DataFrame) -> Dict[any, str]:
        reg: Dict[any, str] = {}
        id_col = self._col(teams_df, "team_id", "teamId", "id")
        name_col = self._col(teams_df, "team_name", "name", "teamName")
        if id_col and name_col:
            for _, row in teams_df.iterrows():
                tid = row[id_col]
                name = str(row[name_col]) if pd.notna(row[name_col]) else "Unknown"
                if pd.notna(tid):
                    reg[tid] = name
        return reg

    # ------------------------------------------------------------------
    # Build players registry
    # ------------------------------------------------------------------

    def _build_players(
        self,
        playing_xi: pd.DataFrame,
        ctpm: pd.DataFrame,
        bbb: pd.DataFrame,
        match_reg: Dict[any, MatchMeta],
    ) -> Dict[any, PlayerMeta]:
        reg: Dict[any, PlayerMeta] = {}

        # Column resolution
        pid_col = self._col(playing_xi, "player_id", "playerId", "id")
        name_col = self._col(playing_xi, "player_name", "name", "playerName")
        bat_col = self._col(playing_xi, "batting_style", "batting_hand", "bat_style")
        bowl_col = self._col(playing_xi, "bowling_style", "bowl_style", "bowling_type")
        match_col_xi = self._col(playing_xi, "match_id", "matchId")
        team_col_xi = self._col(playing_xi, "team_id", "teamId")

        # Seed from playing_xi
        for _, row in playing_xi.iterrows():
            pid = self._val(row, "player_id", "playerId", "id")
            if pid is None:
                continue

            if pid not in reg:
                reg[pid] = PlayerMeta(
                    player_id=pid,
                    name=str(self._val(row, "player_name", "name", "playerName") or pid),
                    batting_style=self._val(row, "batting_style", "batting_hand"),
                    bowling_style=self._val(row, "bowling_style", "bowling_type"),
                )

            if match_col_xi:
                mid = row.get(match_col_xi)
                if pd.notna(mid):
                    reg[pid].match_ids.add(mid)
                    if mid in match_reg:
                        match_reg[mid].player_ids.add(pid)

            if team_col_xi:
                tid = row.get(team_col_xi)
                if pd.notna(tid):
                    reg[pid].team_ids.add(tid)

        # Enrich from competition_team_player_mapping
        pid_col_ctpm = self._col(ctpm, "player_id", "playerId")
        cid_col_ctpm = self._col(ctpm, "competition_id", "competitionId")
        if pid_col_ctpm and cid_col_ctpm:
            for _, row in ctpm.iterrows():
                pid = row[pid_col_ctpm]
                cid = row[cid_col_ctpm]
                if pd.notna(pid):
                    if pid not in reg:
                        reg[pid] = PlayerMeta(
                            player_id=pid,
                            name=str(pid),
                            batting_style=None,
                            bowling_style=None,
                        )
                    if pd.notna(cid):
                        reg[pid].competition_ids.add(cid)

        # Enrich batting/bowling style from ball_by_ball if columns exist
        striker_col = self._col(bbb, "striker_id", "striker")
        bat_style_col = self._col(bbb, "striker_batting_style", "batter_style", "batting_style")
        bowler_col = self._col(bbb, "bowler_id", "bowler")
        bowl_style_col = self._col(bbb, "bowler_bowling_style", "bowling_style", "bowler_style")

        if striker_col and bat_style_col:
            style_map = bbb[[striker_col, bat_style_col]].dropna().drop_duplicates()
            for _, row in style_map.iterrows():
                pid = row[striker_col]
                if pid in reg and reg[pid].batting_style is None:
                    reg[pid].batting_style = row[bat_style_col]

        if bowler_col and bowl_style_col:
            style_map = bbb[[bowler_col, bowl_style_col]].dropna().drop_duplicates()
            for _, row in style_map.iterrows():
                pid = row[bowler_col]
                if pid in reg and reg[pid].bowling_style is None:
                    reg[pid].bowling_style = row[bowl_style_col]

        return reg

    # ------------------------------------------------------------------
    # Main resolve() method
    # ------------------------------------------------------------------

    def resolve(self, pack: RawDataPack) -> EntityRegistry:
        logger.info("Building competition registry …")
        competition_reg = self._build_competitions(pack.competitions)

        logger.info("Building match registry …")
        match_reg = self._build_matches(pack.matches, competition_reg)

        logger.info("Building team registry …")
        team_reg = self._build_teams(pack.teams)

        logger.info("Building player registry …")
        player_reg = self._build_players(
            pack.playing_eleven,
            pack.competition_team_player_mapping,
            pack.ball_by_ball,
            match_reg,
        )

        # Derive convenience sets
        ipl_matches = {mid for mid, m in match_reg.items() if m.competition_format == "IPL"}
        t20i_matches = {mid for mid, m in match_reg.items() if m.competition_format == "T20I"}
        abandoned = {mid for mid, m in match_reg.items() if m.is_abandoned}
        super_overs = {mid for mid, m in match_reg.items() if m.is_super_over}
        dls = {mid for mid, m in match_reg.items() if m.is_dls}

        registry = EntityRegistry(
            competitions=competition_reg,
            matches=match_reg,
            players=player_reg,
            teams=team_reg,
            ipl_match_ids=ipl_matches,
            t20i_match_ids=t20i_matches,
            abandoned_match_ids=abandoned,
            super_over_match_ids=super_overs,
            dls_match_ids=dls,
        )

        logger.info("\n%s", registry.summary())
        return registry


def resolve_entities(
    pack: RawDataPack,
    config_path: str | Path = "config/data_config.yaml",
) -> EntityRegistry:
    engine = EntityResolutionEngine(config_path)
    return engine.resolve(pack)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.loader import load_raw_data

    pack = load_raw_data()
    registry = resolve_entities(pack)
    print(registry.summary())
