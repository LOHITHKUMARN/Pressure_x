"""
PRESSURE-X — Phase 15 & Research Visualizations Engine
======================================================
Generates publication-grade research figures:
  1. Win Probability Calibration Curve
  2. Situational Pressure & Leverage Distribution by Phase
  3. Batting Pressure vs Volume Quadrant Matrix
  4. Bowling Pressure vs Volume Quadrant Matrix
  5. Monte Carlo Ranking Stability & Credibility Intervals

Author: PRESSURE-X Engineering
Version: 2.0.0
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

# Ensure Matplotlib uses local temp dir
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

# Style parameters
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class ResearchVisualizer:
    """
    Produces research-grade visualizations for PRESSURE-X methodology report.
    """

    def __init__(self, charts_dir: str | Path = "outputs/charts"):
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def plot_wp_calibration(self, df: pd.DataFrame) -> Path:
        """
        Plots reliability diagram for 2nd innings win probability model.
        """
        valid = df[(df["innings"] == 2) & df["batting_team_won"].notna()].copy()
        y_true = valid["batting_team_won"].astype(int)
        y_prob = valid["wp_pre"]

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

        fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect Calibration")
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2.2, color="#1f77b4", label="PRESSURE-X WP Model")

        ax.set_title("Innings 2 Win Probability Calibration Curve", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Mean Predicted Win Probability", fontsize=11)
        ax.set_ylabel("Observed Win Proportion", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#dddddd")
        plt.tight_layout()

        out_path = self.charts_dir / "win_prob_calibration.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved WP calibration plot → %s", out_path)
        return out_path

    def plot_pressure_distribution(self, df: pd.DataFrame) -> Path:
        """
        Plots distribution of pressure scores broken down by innings phase.
        """
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        phases = ["powerplay", "middle", "death"]
        colors = ["#2ca02c", "#ff7f0e", "#d62728"]

        for phase, col in zip(phases, colors):
            sub = df[df["innings_phase"] == phase]["pressure_score"].dropna()
            ax.hist(sub, bins=35, alpha=0.55, density=True, label=phase.capitalize(), color=col)

        ax.set_title("Situational Pressure Score Distribution by Innings Phase", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Contextual Pressure Score (0 = Minimal, 1 = Extreme)", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#dddddd")
        plt.tight_layout()

        out_path = self.charts_dir / "pressure_distribution.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved pressure distribution plot → %s", out_path)
        return out_path

    def plot_top_batters_matrix(self, top10_batters: pd.DataFrame) -> Path:
        """
        Plots RAA per 100 balls vs High-Pressure RAA per 100 balls with WPA bubble sizes.
        """
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

        x = top10_batters["praa_per_100"]
        y = top10_batters["hp_raa_per_100"]
        sizes = np.clip(top10_batters["wpa_total"] * 45.0, 60, 600)

        scatter = ax.scatter(x, y, s=sizes, c=top10_batters["batting_pressure_index"], cmap="viridis", alpha=0.85, edgecolors="#333333", linewidth=1.2)

        for _, r in top10_batters.iterrows():
            ax.annotate(
                f"{r['player_name']}\n(#{int(r['rank'])})",
                (r["praa_per_100"] + 0.8, r["hp_raa_per_100"] + 0.5),
                fontsize=8.5,
                fontweight="semibold",
            )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Batting Pressure Index", fontsize=10)

        ax.axhline(0, color="#aaaaaa", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="#aaaaaa", linestyle="--", linewidth=0.8)

        ax.set_title("Top 10 Batters: Pressure-Weighted RAA vs High-Pressure RAA\n(Bubble Size = Total Win Probability Added)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Pressure-Weighted RAA per 100 Balls", fontsize=11)
        ax.set_ylabel("High-Pressure RAA per 100 Balls", fontsize=11)
        plt.tight_layout()

        out_path = self.charts_dir / "top10_batters_matrix.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved top batters matrix plot → %s", out_path)
        return out_path

    def plot_top_bowlers_matrix(self, top10_bowlers: pd.DataFrame) -> Path:
        """
        Plots PRPA per 100 balls vs High-Pressure RPA per 100 balls with WPA bubble sizes.
        """
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

        x = top10_bowlers["prpa_per_100"]
        y = top10_bowlers["hp_rpa_per_100"]
        sizes = np.clip(top10_bowlers["wpa_total"] * 60.0, 60, 600)

        scatter = ax.scatter(x, y, s=sizes, c=top10_bowlers["bowling_pressure_index"], cmap="plasma", alpha=0.85, edgecolors="#333333", linewidth=1.2)

        for _, r in top10_bowlers.iterrows():
            ax.annotate(
                f"{r['player_name']}\n(#{int(r['rank'])})",
                (r["prpa_per_100"] + 0.6, r["hp_rpa_per_100"] + 0.5),
                fontsize=8.5,
                fontweight="semibold",
            )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Bowling Pressure Index", fontsize=10)

        ax.axhline(0, color="#aaaaaa", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="#aaaaaa", linestyle="--", linewidth=0.8)

        ax.set_title("Top 10 Bowlers: Pressure-Weighted RPA vs High-Pressure RPA\n(Bubble Size = Total Win Probability Added)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Pressure-Weighted Runs Prevented (PRPA) per 100 Balls", fontsize=11)
        ax.set_ylabel("High-Pressure Runs Prevented per 100 Balls", fontsize=11)
        plt.tight_layout()

        out_path = self.charts_dir / "top10_bowlers_matrix.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Saved top bowlers matrix plot → %s", out_path)
        return out_path
