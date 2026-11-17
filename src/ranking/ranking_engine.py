"""
PRESSURE-X — Phase 14-16: Ranking & Explainability Engine
==========================================================
Produces:
  1. Top-10 Batting Leaderboard
  2. Top-10 Bowling Leaderboard
  3. Dynamic, data-driven explainability narratives for each ranked player
  4. Formatted Markdown and CSV submission outputs

Author: PRESSURE-X Engineering
Version: 2.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class RankingEngine:
    """
    Ranks players and generates transparent, publication-grade leaderboards.
    """

    def __init__(self, outputs_dir: str | Path = "outputs"):
        self.outputs_dir = Path(outputs_dir)
        self.rankings_dir = self.outputs_dir / "rankings"
        self.reports_dir = self.outputs_dir / "reports"
        self.rankings_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_batter_explanation(row: pd.Series) -> str:
        """
        Synthesizes model-driven explanation of why the batter ranks at this position.
        """
        name = row["player_name"]
        rank = int(row["rank"])
        praa = row["praa_per_100"]
        hp_raa = row["hp_raa_per_100"]
        wpa = row["wpa_total"]
        matches = int(row["matches"])
        balls = int(row["balls_faced"])
        hp_balls = int(row["hp_balls"])
        rel = row["reliability_index"]

        return (
            f"Ranks #{rank} with an overall Pressure Index of {row['batting_pressure_index']:.1f}. "
            f"Delivered +{praa:.1f} pressure-weighted runs above expectation per 100 balls across {balls} deliveries "
            f"({matches} matches). In {hp_balls} high-pressure deliveries, out-scored situational benchmarks by "
            f"{hp_raa:+.1f} runs/100b, generating a massive {wpa:+.2f} net win probability added with a "
            f"{rel*100:.0f}% reliability consistency score."
        )

    @staticmethod
    def generate_bowler_explanation(row: pd.Series) -> str:
        """
        Synthesizes model-driven explanation of why the bowler ranks at this position.
        """
        name = row["player_name"]
        rank = int(row["rank"])
        prpa = row["prpa_per_100"]
        hp_rpa = row["hp_rpa_per_100"]
        wpa = row["wpa_total"]
        matches = int(row["matches"])
        balls = int(row["balls_bowled"])
        hp_balls = int(row["hp_balls"])
        wickets = int(row["actual_wickets"])
        waa = row["wicket_above_exp"]
        rel = row["reliability_index"]

        return (
            f"Ranks #{rank} with an overall Pressure Index of {row['bowling_pressure_index']:.1f}. "
            f"Suppressed opposition scoring by +{prpa:.1f} pressure-weighted runs prevented per 100 balls over "
            f"{balls} deliveries ({matches} matches, {wickets} wickets, {waa:+.1f} wickets above expectation). "
            f"Maintained an elite {hp_rpa:+.1f} runs prevented/100b in {hp_balls} high-pressure deliveries, swinging "
            f"{wpa:+.2f} total win probability for the bowling side ({rel*100:.0f}% reliability score)."
        )

    def produce_rankings(
        self,
        batters_df: pd.DataFrame,
        bowlers_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates Top-10 Batting and Top-10 Bowling leaderboards with explanations.
        """
        top10_batters = batters_df.head(10).copy()
        top10_bowlers = bowlers_df.head(10).copy()

        top10_batters["explanation"] = top10_batters.apply(self.generate_batter_explanation, axis=1)
        top10_bowlers["explanation"] = top10_bowlers.apply(self.generate_bowler_explanation, axis=1)

        # Export CSVs
        batters_df.to_csv(self.rankings_dir / "all_batters_ranked.csv", index=False)
        top10_batters.to_csv(self.rankings_dir / "top_10_batters.csv", index=False)

        bowlers_df.to_csv(self.rankings_dir / "all_bowlers_ranked.csv", index=False)
        top10_bowlers.to_csv(self.rankings_dir / "top_10_bowlers.csv", index=False)

        # Export readable markdown summary
        self._write_markdown_report(top10_batters, top10_bowlers)

        logger.info("Top-10 leaderboards and explanations successfully produced and saved.")
        return top10_batters, top10_bowlers

    def _write_markdown_report(self, top_batters: pd.DataFrame, top_bowlers: pd.DataFrame) -> None:
        md_lines = [
            "# PRESSURE-X: Official Top-10 Pressure Leaderboards",
            "## SupeRR Selector Season 2 | Advanced Sports Analytics",
            "",
            "### Top 10 Batters (Pressure-Adjusted Performance & Reliability)",
            "| Rank | Batter | Matches | Balls | Runs | Exp Runs | RAA | PRAA/100 | High-P RAA/100 | WPA | Index |",
            "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for _, r in top_batters.iterrows():
            md_lines.append(
                f"| **{int(r['rank'])}** | **{r['player_name']}** | {int(r['matches'])} | {int(r['balls_faced'])} | "
                f"{int(r['actual_runs'])} | {r['expected_runs']:.1f} | {r['raa_total']:+.1f} | "
                f"{r['praa_per_100']:+.1f} | {r['hp_raa_per_100']:+.1f} | {r['wpa_total']:+.2f} | **{r['batting_pressure_index']:.1f}** |"
            )

        md_lines.extend([
            "",
            "### Top 10 Bowlers (Pressure-Adjusted Performance & Reliability)",
            "| Rank | Bowler | Matches | Balls | Conc. | Exp Conc. | Wkts | WAA | PRPA/100 | High-P RPA/100 | WPA | Index |",
            "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ])
        for _, r in top_bowlers.iterrows():
            md_lines.append(
                f"| **{int(r['rank'])}** | **{r['player_name']}** | {int(r['matches'])} | {int(r['balls_bowled'])} | "
                f"{int(r['actual_conceded'])} | {r['expected_conceded']:.1f} | {int(r['actual_wickets'])} | {r['wicket_above_exp']:+.1f} | "
                f"{r['prpa_per_100']:+.1f} | {r['hp_rpa_per_100']:+.1f} | {r['wpa_total']:+.2f} | **{r['bowling_pressure_index']:.1f}** |"
            )

        md_lines.extend(["", "---", "### Detailed Player Explanations", "#### Batting:", ""])
        for _, r in top_batters.iterrows():
            md_lines.append(f"- **#{int(r['rank'])} {r['player_name']}**: {r['explanation']}\n")

        md_lines.extend(["#### Bowling:", ""])
        for _, r in top_bowlers.iterrows():
            md_lines.append(f"- **#{int(r['rank'])} {r['player_name']}**: {r['explanation']}\n")

        report_path = self.reports_dir / "official_rankings_report.md"
        report_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info("Markdown report saved to %s", report_path)

    def produce_competition_rankings(
        self,
        df: pd.DataFrame,
        bat_intel: any,
        bowl_intel: any,
        min_balls_ipl: int = 150,
        min_balls_t20i: int = 150,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Produces competition-specific leaderboards:
          - IPL-only Top 10 Batters & Bowlers
          - T20I-only Top 10 Batters & Bowlers
        Saved to outputs/rankings/.
        """
        results = {}

        # 1. IPL Rankings
        if "competition_format" in df.columns:
            ipl_df = df[df["competition_format"] == "IPL"].copy()
            if len(ipl_df) > 0:
                logger.info("Evaluating IPL-specific leaderboards (%d deliveries)...", len(ipl_df))
                bat_intel_ipl = bat_intel.__class__(min_balls=min_balls_ipl, min_matches=6, min_hp_balls=15)
                bowl_intel_ipl = bowl_intel.__class__(min_balls=min_balls_ipl, min_matches=6, min_hp_balls=15)
                bat_ipl = bat_intel_ipl.evaluate_batters(ipl_df)
                bowl_ipl = bowl_intel_ipl.evaluate_bowlers(ipl_df)

                top_bat_ipl = bat_ipl.head(10).copy()
                top_bowl_ipl = bowl_ipl.head(10).copy()
                top_bat_ipl["rank"] = range(1, len(top_bat_ipl) + 1)
                top_bowl_ipl["rank"] = range(1, len(top_bowl_ipl) + 1)

                top_bat_ipl.to_csv(self.rankings_dir / "top_10_batters_ipl.csv", index=False)
                top_bowl_ipl.to_csv(self.rankings_dir / "top_10_bowlers_ipl.csv", index=False)
                bat_ipl.to_csv(self.rankings_dir / "all_batters_ranked_ipl.csv", index=False)
                bowl_ipl.to_csv(self.rankings_dir / "all_bowlers_ranked_ipl.csv", index=False)
                results["IPL"] = (top_bat_ipl, top_bowl_ipl)
                logger.info("Saved IPL rankings → top_10_batters_ipl.csv, top_10_bowlers_ipl.csv")

            # 2. T20I Rankings
            t20i_df = df[df["competition_format"] == "T20I"].copy()
            if len(t20i_df) > 0:
                logger.info("Evaluating T20I-specific leaderboards (%d deliveries)...", len(t20i_df))
                bat_intel_t20i = bat_intel.__class__(min_balls=min_balls_t20i, min_matches=6, min_hp_balls=15)
                bowl_intel_t20i = bowl_intel.__class__(min_balls=min_balls_t20i, min_matches=6, min_hp_balls=15)
                bat_t20i = bat_intel_t20i.evaluate_batters(t20i_df)
                bowl_t20i = bowl_intel_t20i.evaluate_bowlers(t20i_df)

                top_bat_t20i = bat_t20i.head(10).copy()
                top_bowl_t20i = bowl_t20i.head(10).copy()
                top_bat_t20i["rank"] = range(1, len(top_bat_t20i) + 1)
                top_bowl_t20i["rank"] = range(1, len(top_bowl_t20i) + 1)

                top_bat_t20i.to_csv(self.rankings_dir / "top_10_batters_t20i.csv", index=False)
                top_bowl_t20i.to_csv(self.rankings_dir / "top_10_bowlers_t20i.csv", index=False)
                bat_t20i.to_csv(self.rankings_dir / "all_batters_ranked_t20i.csv", index=False)
                bowl_t20i.to_csv(self.rankings_dir / "all_bowlers_ranked_t20i.csv", index=False)
                results["T20I"] = (top_bat_t20i, top_bowl_t20i)
                logger.info("Saved T20I rankings → top_10_batters_t20i.csv, top_10_bowlers_t20i.csv")

        return results
