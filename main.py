"""
PRESSURE-X — Phase 1 Pipeline Runner
======================================
Orchestrates the four Phase 1 modules in order:
  1. Data Ingestion
  2. Data Validation
  3. Entity Resolution
  4. Match Reconstruction

Produces:
  - data/processed/deliveries_enriched.parquet
  - outputs/reports/data_quality_report.txt
  - outputs/reports/entity_registry_summary.txt

Usage:
    cd "c:\\Users\\ASUS\\Desktop\\rr analysis"
    python main.py --phase 1

Author: PRESSURE-X Engineering
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Ensure src/ is on the path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion.loader import load_raw_data
from validation.validator import validate_data
from reconstruction.entity_resolver import resolve_entities
from reconstruction.match_reconstructor import reconstruct_matches
from match_state.state_engine import build_match_states
from pressure.pressure_engine import PressureIntelligenceEngine
from batting.expected_runs import ExpectedBattingModel
from bowling.expected_conceded import ExpectedBowlingModel
from batting.batting_rating import BattingIntelligenceEngine
from bowling.bowling_rating import BowlingIntelligenceEngine
from ranking.ranking_engine import RankingEngine
from matchup.matchup_engine import MatchupIntelligenceEngine
from simulation.monte_carlo import MonteCarloSimulationEngine
from evaluation.ablation_validator import AblationValidationEngine
from explainability.visualizer import ResearchVisualizer


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase1_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("PRESSURE-X Phase 1 Pipeline starting …")
    logging.info("Log: %s", log_file)


def run_phase1(config_path: Path, save_reports: bool = True) -> None:
    logger = logging.getLogger("pressure_x.pipeline")
    reports_dir = ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Module 1: Data Ingestion
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("MODULE 1: Data Ingestion")
    logger.info("=" * 60)
    pack = load_raw_data(config_path)

    # ------------------------------------------------------------------ #
    # Module 2: Data Validation
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("MODULE 2: Data Validation")
    logger.info("=" * 60)
    quality_report = validate_data(pack, config_path)
    quality_report.print_summary()

    if save_reports:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            quality_report.print_summary()
        (reports_dir / "data_quality_report.txt").write_text(buf.getvalue(), encoding="utf-8")
        logger.info("Data quality report saved → outputs/reports/data_quality_report.txt")

    if not quality_report.overall_pass:
        logger.warning("Data validation found issues — proceeding with caution.")

    # ------------------------------------------------------------------ #
    # Module 3: Entity Resolution
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("MODULE 3: Entity Resolution")
    logger.info("=" * 60)
    registry = resolve_entities(pack, config_path)
    print(registry.summary())

    if save_reports:
        (reports_dir / "entity_registry_summary.txt").write_text(
            registry.summary(), encoding="utf-8"
        )
        logger.info("Entity registry summary saved → outputs/reports/entity_registry_summary.txt")

    # ------------------------------------------------------------------ #
    # Module 4: Match Reconstruction
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("MODULE 4: Match Reconstruction")
    logger.info("=" * 60)
    enriched_df = reconstruct_matches(pack, registry, config_path, save=True)

    logger.info("Phase 1 complete.")
    logger.info("Enriched dataset: %d rows × %d columns", len(enriched_df), len(enriched_df.columns))
    logger.info("Columns added: %s",
                [c for c in enriched_df.columns if c not in pack.ball_by_ball.columns])

    # Quick sanity print
    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(f"  Deliveries processed  : {len(enriched_df):,}")
    print(f"  Matches               : {enriched_df['match_id'].nunique() if 'match_id' in enriched_df.columns else 'n/a':,}")
    print(f"  Competitions          : {len(registry.competitions):,}")
    print(f"  Players               : {len(registry.players):,}")
    comp_counts = enriched_df["competition_format"].value_counts() if "competition_format" in enriched_df.columns else {}
    for fmt, cnt in comp_counts.items():
        print(f"  {fmt:20s}: {cnt:,} deliveries")
    print("=" * 60)
    print("\nOutput: data/processed/deliveries_enriched.parquet")
    print("Reports: outputs/reports/")
    print("\nPhase 1 ✓ — Ready for Phase 2: Match-State Intelligence Engine")


def run_phase2(config_path: Path) -> None:
    logger = logging.getLogger("pressure_x.pipeline")
    logger.info("=" * 60)
    logger.info("STARTING PRESSURE-X CORE INTELLIGENCE PIPELINE")
    logger.info("=" * 60)

    # 1. Load Phase 1 Enriched Deliveries & Raw Matches
    processed_path = ROOT / "data" / "processed" / "deliveries_enriched.parquet"
    if not processed_path.exists():
        logger.info("Processed deliveries not found. Running Phase 1 first...")
        run_phase1(config_path)

    logger.info("Loading enriched deliveries from %s...", processed_path)
    df = pd.read_parquet(processed_path)
    matches_df = pd.read_csv(ROOT / "data" / "raw" / "matches.csv")
    if "match_date" in matches_df.columns:
        date_map = matches_df.set_index("match_id")["match_date"].to_dict()
        df["match_date"] = df["match_id"].map(date_map)

    # 2. Phase 2: Match-State Transition Engine
    logger.info("=" * 60)
    logger.info("PHASE 2: Match-State Transition Engine")
    logger.info("=" * 60)
    df = build_match_states(df, matches_df, config_path)

    # 3. Phase 3: Pressure Intelligence Engine & Win Probability
    logger.info("=" * 60)
    logger.info("PHASE 3: Win Probability, Leverage & Pressure Intelligence")
    logger.info("=" * 60)
    pressure_engine = PressureIntelligenceEngine(ROOT / "config" / "pressure_config.yaml")
    df = pressure_engine.compute_pressure_metrics(df, use_oof=True)

    # 4. Phase 4: Expected Performance Intelligence (Batting & Bowling OOF)
    logger.info("=" * 60)
    logger.info("PHASE 4: Expected Batting & Bowling Models (Out-of-Fold Chronological)")
    logger.info("=" * 60)
    bat_model = ExpectedBattingModel(ROOT / "models")
    exp_runs, prob_b, prob_d, prob_w, bat_metrics = bat_model.generate_oof_predictions(df, n_splits=5)
    logger.info("Batting Model OOF Metrics: %s", bat_metrics)
    df["exp_runs_bat"] = exp_runs
    df["prob_boundary_bat"] = prob_b
    df["prob_dot_bat"] = prob_d
    df["prob_wicket_bat"] = prob_w

    bowl_model = ExpectedBowlingModel(ROOT / "models")
    exp_conc, prob_bw, prob_bd, prob_bb, bowl_metrics = bowl_model.generate_oof_predictions(df, n_splits=5)
    logger.info("Bowling Model OOF Metrics: %s", bowl_metrics)
    df["exp_runs_bowl"] = exp_conc
    df["prob_wicket_bowl"] = prob_bw
    df["prob_dot_bowl"] = prob_bd
    df["prob_boundary_bowl"] = prob_bb

    # Save full feature & model-enriched parquet
    features_out = ROOT / "data" / "features" / "deliveries_modeled.parquet"
    df.to_parquet(features_out, index=False)
    logger.info("Model-enriched deliveries saved to %s (%d rows)", features_out, len(df))

    # 5. Phase 5 & 7: Advanced Batting Intelligence & Bayesian Shrinkage
    logger.info("=" * 60)
    logger.info("PHASE 5-7: Advanced Batting Intelligence & Bayesian Shrinkage")
    logger.info("=" * 60)
    bat_intel = BattingIntelligenceEngine(min_balls=250, min_matches=8, min_hp_balls=20)
    batters_ranked = bat_intel.evaluate_batters(df)

    # 6. Phase 6 & 7: Advanced Bowling Intelligence & Bayesian Shrinkage
    logger.info("=" * 60)
    logger.info("PHASE 6-7: Advanced Bowling Intelligence & Bayesian Shrinkage")
    logger.info("=" * 60)
    bowl_intel = BowlingIntelligenceEngine(min_balls=250, min_matches=8, min_hp_balls=20)
    bowlers_ranked = bowl_intel.evaluate_bowlers(df)

    # 7. Phase 14-16: Ranking Engine & Explainability Generator
    logger.info("=" * 60)
    logger.info("PHASE 14-16: Ranking & Explainability Engine")
    logger.info("=" * 60)
    ranking_engine = RankingEngine(ROOT / "outputs")
    top_batters, top_bowlers = ranking_engine.produce_rankings(batters_ranked, bowlers_ranked)
    comp_rankings = ranking_engine.produce_competition_rankings(df, bat_intel, bowl_intel)

    print("\n" + "=" * 80)
    print("🏆 PRESSURE-X: OFFICIAL TOP 10 BATTING LEADERBOARD (Pressure Index)")
    print("=" * 80)
    for _, r in top_batters.iterrows():
        print(f"#{int(r['rank']):2d} | {r['player_name']:<24s} | Index: {r['batting_pressure_index']:5.1f} | "
              f"Runs: {int(r['actual_runs']):4d} (Exp: {r['expected_runs']:5.1f}) | RAA: {r['raa_total']:+5.1f} | "
              f"PRAA/100: {r['praa_per_100']:+5.1f} | HP-RAA/100: {r['hp_raa_per_100']:+5.1f} | WPA: {r['wpa_total']:+5.2f}")

    print("\n" + "=" * 80)
    print("🎯 PRESSURE-X: OFFICIAL TOP 10 BOWLING LEADERBOARD (Pressure Index)")
    print("=" * 80)
    for _, r in top_bowlers.iterrows():
        print(f"#{int(r['rank']):2d} | {r['player_name']:<24s} | Index: {r['bowling_pressure_index']:5.1f} | "
              f"Wkts: {int(r['actual_wickets']):2d} (WAA: {r['wicket_above_exp']:+4.1f}) | Conc: {int(r['actual_conceded']):4d} (Exp: {r['expected_conceded']:5.1f}) | "
              f"PRPA/100: {r['prpa_per_100']:+5.1f} | HP-RPA/100: {r['hp_rpa_per_100']:+5.1f} | WPA: {r['wpa_total']:+5.2f}")

    print("\n" + "=" * 80)
    print("Artifacts generated:")
    print("  - outputs/rankings/top_10_batters.csv")
    print("  - outputs/rankings/top_10_bowlers.csv")
    print("  - outputs/rankings/all_batters_ranked.csv")
    print("  - outputs/rankings/all_bowlers_ranked.csv")
    print("  - outputs/reports/official_rankings_report.md")
    print("  - data/features/deliveries_modeled.parquet")
    print("=" * 80 + "\n")


def run_phase3(config_path: Path) -> None:
    logger = logging.getLogger("pressure_x.pipeline")
    logger.info("=" * 60)
    logger.info("STARTING PHASE 9-15: ADVANCED AI & RESEARCH VALIDATION LAYER")
    logger.info("=" * 60)

    features_path = ROOT / "data" / "features" / "deliveries_modeled.parquet"
    if not features_path.exists():
        logger.info("Modeled deliveries not found. Running Phase 2 first...")
        run_phase2(config_path)

    logger.info("Loading modeled deliveries from %s...", features_path)
    df = pd.read_parquet(features_path)
    matches_df = pd.read_csv(ROOT / "data" / "raw" / "matches.csv")
    if "match_date" not in df.columns and "match_date" in matches_df.columns:
        date_map = matches_df.set_index("match_id")["match_date"].to_dict()
        df["match_date"] = df["match_id"].map(date_map)

    top_batters_csv = ROOT / "outputs" / "rankings" / "top_10_batters.csv"
    top_bowlers_csv = ROOT / "outputs" / "rankings" / "top_10_bowlers.csv"
    all_batters_csv = ROOT / "outputs" / "rankings" / "all_batters_ranked.csv"
    all_bowlers_csv = ROOT / "outputs" / "rankings" / "all_bowlers_ranked.csv"
    top_batters = pd.read_csv(top_batters_csv)
    top_bowlers = pd.read_csv(top_bowlers_csv)

    # 1. Phase 9: Matchup & Environmental Intelligence Engine
    logger.info("=" * 60)
    logger.info("PHASE 9: Matchup & Environmental Intelligence")
    logger.info("=" * 60)
    matchup_engine = MatchupIntelligenceEngine()
    m_metrics = matchup_engine.fit(df)
    df = matchup_engine.transform(df)
    logger.info("Matchup & Environmental Metrics: %s", m_metrics)

    # 2. Phase 13: Chronological Out-of-Sample Holdout & Ablation Analysis
    logger.info("=" * 60)
    logger.info("PHASE 13: Chronological Out-of-Sample Holdout & Ablation Engine")
    logger.info("=" * 60)
    ablation_engine = AblationValidationEngine()
    holdout_metrics = ablation_engine.run_chronological_validation(df, matches_df)
    logger.info("Holdout Validation Metrics: %s", holdout_metrics)

    ablation_df = ablation_engine.run_ablation_comparison(df)
    ablation_df.to_csv(ROOT / "outputs" / "reports" / "ablation_comparison.csv", index=False)
    logger.info("Ablation results saved to outputs/reports/ablation_comparison.csv")

    # 3. Phase 12: Monte Carlo Simulation & Ranking Stability
    logger.info("=" * 60)
    logger.info("PHASE 12: Monte Carlo Simulation & Ranking Stability Analysis (1,000 Iterations)")
    logger.info("=" * 60)
    mc_engine = MonteCarloSimulationEngine(n_iterations=1000)

    all_batters = pd.read_csv(all_batters_csv)
    candidate_batter_ids = all_batters.head(50)["striker_id"].tolist()
    mc_bat_df = mc_engine.simulate_batting_stability(df, candidate_batter_ids)
    top_batters = top_batters.drop(columns=[c for c in ["prob_top_10", "prob_top_5", "median_sim_rank", "rank_ci_95", "stability_score"] if c in top_batters.columns], errors="ignore")
    top_batters = top_batters.merge(mc_bat_df, on="striker_id", how="left")
    top_batters.to_csv(top_batters_csv, index=False)

    all_bowlers = pd.read_csv(all_bowlers_csv)
    candidate_bowler_ids = all_bowlers.head(50)["bowler_id"].tolist()
    mc_bowl_df = mc_engine.simulate_bowling_stability(df, candidate_bowler_ids)
    top_bowlers = top_bowlers.drop(columns=[c for c in ["prob_top_10", "prob_top_5", "median_sim_rank", "rank_ci_95", "stability_score"] if c in top_bowlers.columns], errors="ignore")
    top_bowlers = top_bowlers.merge(mc_bowl_df, on="bowler_id", how="left")
    top_bowlers.to_csv(top_bowlers_csv, index=False)

    # 4. Phase 15: Research Visualizations
    logger.info("=" * 60)
    logger.info("PHASE 15: Research Visualizations & Charts Generation")
    logger.info("=" * 60)
    visualizer = ResearchVisualizer(ROOT / "outputs" / "charts")
    visualizer.plot_wp_calibration(df)
    visualizer.plot_pressure_distribution(df)
    visualizer.plot_top_batters_matrix(top_batters)
    visualizer.plot_top_bowlers_matrix(top_bowlers)

    print("\n" + "=" * 80)
    print("🔬 ADVANCED RESEARCH LAYER SUMMARY & MONTE CARLO STABILITY")
    print("=" * 80)
    print("\nTop 10 Batters (Monte Carlo Credibility Intervals):")
    for _, r in top_batters.iterrows():
        print(f"#{int(r['rank']):2d} | {r['player_name']:<24s} | Index: {r['batting_pressure_index']:5.1f} | "
              f"P(Top 10): {r['prob_top_10']*100:5.1f}% | Median Rank: {r['median_sim_rank']:4.1f} | 95% CI: {r['rank_ci_95']}")

    print("\nTop 10 Bowlers (Monte Carlo Credibility Intervals):")
    for _, r in top_bowlers.iterrows():
        print(f"#{int(r['rank']):2d} | {r['player_name']:<24s} | Index: {r['bowling_pressure_index']:5.1f} | "
              f"P(Top 10): {r['prob_top_10']*100:5.1f}% | Median Rank: {r['median_sim_rank']:4.1f} | 95% CI: {r['rank_ci_95']}")

    print("\n" + "=" * 80)
    print("Research Charts Generated:")
    print("  - outputs/charts/win_prob_calibration.png")
    print("  - outputs/charts/pressure_distribution.png")
    print("  - outputs/charts/top10_batters_matrix.png")
    print("  - outputs/charts/top10_bowlers_matrix.png")
    print("  - outputs/reports/ablation_comparison.csv")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PRESSURE-X Pipeline Runner")
    parser.add_argument("--phase", type=int, default=1, help="Pipeline phase to run (1=Data, 2=Core Model, 3=Advanced Research)")
    parser.add_argument("--config", type=str, default="config/data_config.yaml",
                        help="Path to data config YAML")
    parser.add_argument("--no-reports", action="store_true",
                        help="Skip saving text reports")
    args = parser.parse_args()

    setup_logging(ROOT / "logs")

    config_path = ROOT / args.config

    if args.phase == 1:
        run_phase1(config_path, save_reports=not args.no_reports)
    elif args.phase == 2:
        run_phase2(config_path)
    elif args.phase == 3:
        run_phase3(config_path)
    else:
        print(f"Phase {args.phase} not yet available. Supported: 1, 2, 3")
        sys.exit(1)


if __name__ == "__main__":
    main()
