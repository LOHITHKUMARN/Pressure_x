"""
PRESSURE-X — Module 2: Data Validation
========================================
Runs a comprehensive audit of all six loaded DataFrames:
  - Row count checks against expected values
  - Null / missing value profiles
  - Duplicate detection
  - Value-range sanity checks
  - Cross-file referential integrity
  - Delivery-level anomaly detection (illegal ball counts, over counts, etc.)

Produces a structured DataQualityReport with per-file and cross-file findings.
Nothing is modified here — validation only.

Author: PRESSURE-X Engineering
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from ingestion.loader import RawDataPack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FileQualityReport:
    name: str
    row_count: int
    expected_rows: Optional[int]
    col_count: int
    columns: List[str]
    null_counts: Dict[str, int]
    null_pct: Dict[str, float]
    duplicate_rows: int
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CrossFileReport:
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    orphaned_match_ids: List[Any] = field(default_factory=list)
    orphaned_player_ids: List[Any] = field(default_factory=list)
    missing_competition_refs: List[Any] = field(default_factory=list)


@dataclass
class DeliveryAnomalyReport:
    total_deliveries: int = 0
    legal_deliveries: int = 0
    illegal_deliveries: int = 0
    extra_deliveries: int = 0
    unclassified_extra_deliveries: int = 0
    over_count_anomalies: int = 0
    ball_count_anomalies: int = 0
    negative_runs: int = 0
    impossible_runs_per_ball: int = 0
    missing_striker: int = 0
    missing_bowler: int = 0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    files: Dict[str, FileQualityReport] = field(default_factory=dict)
    cross_file: CrossFileReport = field(default_factory=CrossFileReport)
    deliveries: DeliveryAnomalyReport = field(default_factory=DeliveryAnomalyReport)
    competition_classification: Dict[str, int] = field(default_factory=dict)
    overall_pass: bool = True

    def print_summary(self) -> None:
        print("\n" + "=" * 70)
        print("PRESSURE-X  Data Quality Report")
        print("=" * 70)

        for name, rep in self.files.items():
            status = "✓" if not rep.issues else "✗"
            row_note = ""
            if rep.expected_rows and rep.row_count != rep.expected_rows:
                row_note = f" (expected {rep.expected_rows:,})"
            print(f"\n  {status} {name}")
            print(f"     rows={rep.row_count:,}{row_note}  cols={rep.col_count}  dupes={rep.duplicate_rows:,}")
            if rep.issues:
                for iss in rep.issues:
                    print(f"     ✗ {iss}")
            if rep.warnings:
                for w in rep.warnings:
                    print(f"     ⚠ {w}")

        print("\n  Cross-file integrity")
        for iss in self.cross_file.issues:
            print(f"     ✗ {iss}")
        for w in self.cross_file.warnings:
            print(f"     ⚠ {w}")

        print("\n  Delivery anomalies")
        d = self.deliveries
        print(
            f"     total={d.total_deliveries:,}  "
            f"legal={d.legal_deliveries:,}  "
            f"illegal={d.illegal_deliveries:,}  "
            f"extras={d.extra_deliveries:,}  "
            f"unclassified={d.unclassified_extra_deliveries:,}"
        )
        for warning in d.warnings:
            print(f"     WARNING: {warning}")
        for iss in d.issues:
            print(f"     ⚠ {iss}")

        print("\n  Competition classification")
        for k, v in self.competition_classification.items():
            print(f"     {k}: {v:,} competitions")

        overall = "PASS ✓" if self.overall_pass else "FAIL ✗"
        print(f"\n  Overall: {overall}")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main validation engine
# ---------------------------------------------------------------------------


class DataValidationEngine:
    """
    Module 2: Data Validation
    Performs all quality checks without modifying any data.
    """

    def __init__(self, config_path: str | Path = "config/data_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)
        self.expected_rows: Dict[str, int] = self.config["validation"]
        self.ipl_kw = [k.lower() for k in self.config["formats"]["ipl_keywords"]]
        self.t20i_kw = [k.lower() for k in self.config["formats"]["t20i_keywords"]]
        self.t20_kw = [k.lower() for k in self.config["formats"]["t20_keywords"]]

    # ------------------------------------------------------------------
    # Per-file reports
    # ------------------------------------------------------------------

    def _file_report(self, name: str, df: pd.DataFrame) -> FileQualityReport:
        expected_key = f"expected_{name}_rows".replace("competition_team_player_mapping", "competition_player")
        expected = self.expected_rows.get(expected_key)

        null_counts = df.isnull().sum().to_dict()
        null_pct = {k: round(v / max(len(df), 1) * 100, 2) for k, v in null_counts.items()}
        duplicate_rows = int(df.duplicated().sum())

        issues, warnings = [], []

        if expected and len(df) != expected:
            diff = len(df) - expected
            issues.append(f"Row count {len(df):,} differs from expected {expected:,} (Δ={diff:+,})")

        if duplicate_rows > 0:
            warnings.append(f"{duplicate_rows:,} fully duplicate rows detected")

        high_null = {k: f"{v:.1f}%" for k, v in null_pct.items() if v > 20}
        if high_null:
            warnings.append(f"High-null columns (>20%): {high_null}")

        return FileQualityReport(
            name=name,
            row_count=len(df),
            expected_rows=expected,
            col_count=len(df.columns),
            columns=list(df.columns),
            null_counts={k: int(v) for k, v in null_counts.items()},
            null_pct=null_pct,
            duplicate_rows=duplicate_rows,
            issues=issues,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Cross-file referential integrity
    # ------------------------------------------------------------------

    def _cross_file_check(self, pack: RawDataPack) -> CrossFileReport:
        report = CrossFileReport()

        bbb = pack.ball_by_ball
        matches = pack.matches
        competitions = pack.competitions
        playing_xi = pack.playing_eleven
        ctpm = pack.competition_team_player_mapping

        # Determine column names dynamically
        def _col(df, *candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        bbb_match_col = _col(bbb, "match_id", "matchId", "match")
        matches_match_col = _col(matches, "match_id", "matchId", "id")
        comp_col_matches = _col(matches, "comp_id", "competition_id", "competitionId")
        comp_col_comp = _col(competitions, "comp_id", "competition_id", "competitionId", "id")
        player_col_xi = _col(playing_xi, "player_id", "playerId")
        player_col_ctpm = _col(ctpm, "player_id", "playerId")

        # Match IDs in ball-by-ball but not in matches
        if bbb_match_col and matches_match_col:
            bbb_match_ids = set(bbb[bbb_match_col].dropna().unique())
            known_match_ids = set(matches[matches_match_col].dropna().unique())
            orphan = bbb_match_ids - known_match_ids
            if orphan:
                report.orphaned_match_ids = sorted(orphan)[:20]
                report.issues.append(
                    f"{len(orphan):,} match_ids in ball_by_ball not found in matches table"
                )
        else:
            report.warnings.append("Could not resolve match_id column names for cross-file check")

        # Competition IDs in matches but not in competitions
        if comp_col_matches and comp_col_comp:
            match_comp_ids = set(matches[comp_col_matches].dropna().unique())
            known_comp_ids = set(competitions[comp_col_comp].dropna().unique())
            missing = match_comp_ids - known_comp_ids
            if missing:
                report.missing_competition_refs = sorted(missing)[:20]
                report.issues.append(
                    f"{len(missing):,} competition_ids in matches not found in competitions table"
                )

        # Players in ball-by-ball but not in playing_xi
        if player_col_xi:
            known_players = set(playing_xi[player_col_xi].dropna().unique())
            for col in ["striker_id", "non_striker_id", "bowler_id", "strikerId", "bowlerId"]:
                if col in bbb.columns:
                    bbb_players = set(bbb[col].dropna().unique())
                    orphan_p = bbb_players - known_players
                    if orphan_p:
                        n = len(orphan_p)
                        report.warnings.append(
                            f"{n:,} {col} values in ball_by_ball not found in playing_eleven"
                        )
                    break

        return report

    # ------------------------------------------------------------------
    # Delivery-level anomalies
    # ------------------------------------------------------------------

    def _delivery_checks(self, bbb: pd.DataFrame) -> DeliveryAnomalyReport:
        rep = DeliveryAnomalyReport()
        rep.total_deliveries = len(bbb)

        def _col(df, *candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        # --------------------------------------------------
        # Extras and legal-delivery classification
        # --------------------------------------------------
        extra_col = _col(
            bbb,
            "extras_type",
            "extra_type",
            "extraType",
            "is_extra",
            "extras",
        )
        if extra_col:
            extra_values = (
                bbb[extra_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace("-", "_", regex=False)
                .str.replace(" ", "_", regex=False)
            )
            # Values that indicate an extra was recorded.
            no_extra_values = {"", "0", "nan", "none", "null", "false"}
            has_extra = ~extra_values.isin(no_extra_values)
            rep.extra_deliveries = int(has_extra.sum())

            # Known illegal delivery types (wides and no-balls: overstep + other non-overstep no-balls)
            illegal_types = {
                "wide",
                "wides",
                "noball",
                "no_ball",
                "no_balls",
                "noballs",
                "overstep",
                "other",  # Audited: 264 non-overstep no-balls (1-run penalty + batter runs, requires re-bowl)
            }
            # Known legal extras (byes, legbyes, penalty)
            legal_extra_types = {
                "bye",
                "byes",
                "legbye",
                "legbyes",
                "leg_bye",
                "leg_byes",
                "penalty",  # Audited: 1 Law 41 5-run fielding penalty on legal ball (over completed in 6 balls)
            }
            # Unclassified extras (none remaining after forensic audit)
            unclassified_types = set()

            is_illegal = extra_values.isin(illegal_types)
            is_unclassified = extra_values.isin(unclassified_types)
            is_known_legal_extra = extra_values.isin(legal_extra_types)
            is_no_extra = extra_values.isin(no_extra_values)

            rep.illegal_deliveries = int(is_illegal.sum())
            rep.unclassified_extra_deliveries = int(is_unclassified.sum())
            rep.legal_deliveries = int((is_no_extra | is_known_legal_extra).sum())

            if rep.unclassified_extra_deliveries > 0:
                rep.warnings.append(
                    f"{rep.unclassified_extra_deliveries:,} deliveries have unclassified extra type ('other') — pending investigation"
                )
        else:
            rep.legal_deliveries = rep.total_deliveries
            rep.warnings.append("Could not locate extras_type column")

        # Runs sanity
        runs_col = _col(bbb, "runs_batter", "batsman_runs", "batter_runs", "runs_off_bat")
        total_runs_col = _col(bbb, "runs_total", "total_runs", "runs")
        if runs_col:
            neg = int((bbb[runs_col] < 0).sum())
            if neg > 0:
                rep.negative_runs = neg
                rep.issues.append(f"{neg} deliveries with negative batter runs")

            impossible = int((bbb[runs_col] > 7).sum())
            if impossible > 0:
                rep.impossible_runs_per_ball = impossible
                rep.issues.append(
                    f"{impossible} deliveries with >7 batter runs (unusual — verify)"
                )

        # Over number sanity
        over_col = _col(bbb, "over_number", "over", "overNumber")
        if over_col:
            max_over = int(bbb[over_col].max()) if not bbb[over_col].isna().all() else 0
            if max_over > 20:
                rep.over_count_anomalies = int((bbb[over_col] > 20).sum())
                rep.issues.append(
                    f"{rep.over_count_anomalies} deliveries with over_number > 20 (super over?)"
                )

        # Ball number sanity
        ball_col = _col(bbb, "ball_number", "ball", "ballNumber", "delivery")
        if ball_col:
            high_ball = int((bbb[ball_col] > 10).sum())
            if high_ball > 0:
                rep.ball_count_anomalies = high_ball
                rep.issues.append(
                    f"{high_ball} deliveries with ball_number > 10 (validate extras encoding)"
                )

        # Missing striker / bowler
        striker_col = _col(bbb, "striker_id", "striker", "batsman_id")
        bowler_col = _col(bbb, "bowler_id", "bowler", "bowlerId")
        if striker_col:
            rep.missing_striker = int(bbb[striker_col].isna().sum())
        if bowler_col:
            rep.missing_bowler = int(bbb[bowler_col].isna().sum())
        if rep.missing_striker > 0:
            rep.issues.append(f"{rep.missing_striker:,} deliveries missing striker_id")
        if rep.missing_bowler > 0:
            rep.issues.append(f"{rep.missing_bowler:,} deliveries missing bowler_id")

        return rep

    # ------------------------------------------------------------------
    # Competition classification
    # ------------------------------------------------------------------

    def _classify_competitions(self, competitions: pd.DataFrame) -> Dict[str, int]:
        name_col = None
        for c in ["comp_name", "competition_name", "name", "competition", "competitionName"]:
            if c in competitions.columns:
                name_col = c
                break

        format_col = None
        for c in ["match_format", "format", "competition_format"]:
            if c in competitions.columns:
                format_col = c
                break

        if name_col is None and format_col is None:
            return {"unclassified": len(competitions)}

        counts = {"IPL": 0, "T20I": 0, "Other_T20": 0, "Other_format": 0}
        for _, row in competitions.iterrows():
            # Safely read competition name.
            raw_name = (
                row[name_col]
                if name_col and pd.notna(row[name_col])
                else ""
            )
            name_val = str(raw_name).strip().lower()

            # Safely read competition format.
            raw_format = (
                row[format_col]
                if format_col and pd.notna(row[format_col])
                else ""
            )
            fmt_val = str(raw_format).strip().upper()

            # --------------------------------------------------
            # 1. IPL must be checked first.
            # --------------------------------------------------
            is_ipl = (
                any(keyword in name_val for keyword in self.ipl_kw)
                or "IPL" in fmt_val
            )
            if is_ipl:
                counts["IPL"] += 1
                continue

            # --------------------------------------------------
            # 2. International T20 must be checked before
            #    generic T20.
            # --------------------------------------------------
            is_t20i = (
                any(keyword in name_val for keyword in self.t20i_kw)
                or "T20I" in fmt_val
            )
            if is_t20i:
                counts["T20I"] += 1
                continue

            # --------------------------------------------------
            # 3. Other T20 competitions.
            # --------------------------------------------------
            is_other_t20 = (
                any(keyword in name_val for keyword in self.t20_kw)
                or "T20" in fmt_val
            )
            if is_other_t20:
                counts["Other_T20"] += 1
                continue

            # --------------------------------------------------
            # 4. Everything else.
            # --------------------------------------------------
            counts["Other_format"] += 1

        return counts

    # ------------------------------------------------------------------
    # Main validate() method
    # ------------------------------------------------------------------

    def validate(self, pack: RawDataPack) -> DataQualityReport:
        report = DataQualityReport()

        # Map name → (df, expected_key)
        file_map = {
            "ball_by_ball": pack.ball_by_ball,
            "matches": pack.matches,
            "competitions": pack.competitions,
            "teams": pack.teams,
            "playing_eleven": pack.playing_eleven,
            "competition_team_player_mapping": pack.competition_team_player_mapping,
        }

        any_critical = False
        for name, df in file_map.items():
            frep = self._file_report(name, df)
            report.files[name] = frep
            if frep.issues:
                any_critical = True

        report.cross_file = self._cross_file_check(pack)
        if report.cross_file.issues:
            any_critical = True

        report.deliveries = self._delivery_checks(pack.ball_by_ball)
        if report.deliveries.issues:
            # Delivery anomalies are warnings, not critical failures
            pass

        report.competition_classification = self._classify_competitions(pack.competitions)

        # Overall pass: only fail on hard referential integrity issues
        report.overall_pass = not any_critical
        return report


def validate_data(
    pack: RawDataPack,
    config_path: str | Path = "config/data_config.yaml",
) -> DataQualityReport:
    engine = DataValidationEngine(config_path)
    return engine.validate(pack)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingestion.loader import load_raw_data

    pack = load_raw_data()
    report = validate_data(pack)
    report.print_summary()
