"""
PRESSURE-X — Phase 4: Expected Bowling Performance Intelligence
================================================================
Models what an equivalent bowler would be expected to concede in each exact match situation:
  1. Expected Runs Conceded (exp_runs_bowl): Continuous expectation E[Conceded | S_t]
  2. Bowler Wicket Probability (prob_wicket_bowl): P(Bowler Credited Dismissal | S_t)
  3. Dot-ball Conceded Probability (prob_dot_bowl): P(Conceded == 0 | S_t)
  4. Boundary Conceded Probability (prob_boundary_bowl): P(Conceded >= 4 | S_t)

Strict Cricket Attribution Rules:
  - Leg-byes and Byes are fielding extras, NOT charged to the bowler.
  - Wides and No-balls (Overstep) ARE charged to the bowler.
  - RunOuts, Obstructing the field, and Retire Out are NOT credited to the bowler as wickets.
  - Caught, Bowled, LBW, Stump, and Hit Wicket ARE credited to the bowler.

Uses LightGBM models trained on strictly causal pre-delivery features.

Author: PRESSURE-X Engineering
Version: 2.0.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from features.feature_builder import FeatureBuilder
from evaluation.split_utils import get_chronological_kfold_splits

logger = logging.getLogger(__name__)

BOWLER_WICKET_TYPES = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}


class ExpectedBowlingModel:
    """
    Bowling Expected Performance Suite.
    """

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feature_builder = FeatureBuilder()

        self.reg_conceded: Optional[lgb.LGBMRegressor] = None
        self.clf_wicket: Optional[lgb.LGBMClassifier] = None
        self.clf_dot: Optional[lgb.LGBMClassifier] = None
        self.clf_boundary: Optional[lgb.LGBMClassifier] = None

        self.metrics: Dict[str, float] = {}

    @staticmethod
    def calculate_bowler_charge(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates exact bowler-charged runs and bowler-credited wickets.
        """
        extra_type = df["extras_type"].astype(object).fillna("").astype(str).str.lower()
        batter_runs = df["runs"].fillna(0).astype(float)
        total_ball_runs = df["_runs_this_ball"].fillna(0).astype(float)

        # Runs charged to bowler (fielding extras and penalty runs are NOT charged to bowler)
        is_fielding_extra = extra_type.isin(["byes", "legbyes", "penalty", "bye", "legbye"])
        bowler_runs_conceded = np.where(
            is_fielding_extra,
            batter_runs,
            total_ball_runs,
        )

        # Wickets credited to bowler
        dismissal = df["dismissal_type"].fillna("").astype(str)
        is_bowler_wicket = df["_wicket"].astype(bool) & dismissal.isin(BOWLER_WICKET_TYPES)

        return pd.Series(bowler_runs_conceded, index=df.index), is_bowler_wicket.astype(int)

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Trains all expected bowling models on delivery records.
        """
        logger.info("Extracting features for Bowling Expected Performance models...")
        X = self.feature_builder.build_features(df)

        y_conceded, y_wicket = self.calculate_bowler_charge(df)
        y_dot = (y_conceded == 0).astype(int)
        y_boundary = (y_conceded >= 4).astype(int)

        # 1. Expected Conceded Regressor
        logger.info("Training Expected Runs Conceded Regressor (%d rows)...", len(X))
        self.reg_conceded = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.06,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            verbose=-1,
        )
        self.reg_conceded.fit(X, y_conceded)
        pred_conc = self.reg_conceded.predict(X)
        rmse_c = float(np.sqrt(mean_squared_error(y_conceded, pred_conc)))
        mae_c = float(mean_absolute_error(y_conceded, pred_conc))
        logger.info("Expected Conceded: RMSE = %.4f, MAE = %.4f (Baseline mean = %.3f)", rmse_c, mae_c, y_conceded.mean())

        # 2. Bowler Wicket Probability Classifier
        logger.info("Training Bowler Wicket Classifier...")
        self.clf_wicket = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=25,
            max_depth=4,
            random_state=42,
            verbose=-1,
        )
        self.clf_wicket.fit(X, y_wicket)
        pred_w_prob = self.clf_wicket.predict_proba(X)[:, 1]
        auc_w = float(roc_auc_score(y_wicket, pred_w_prob))
        brier_w = float(brier_score_loss(y_wicket, pred_w_prob))
        logger.info("Bowler Wicket Classifier: AUC = %.4f, Brier = %.4f", auc_w, brier_w)

        # 3. Dot Ball Generation Classifier
        logger.info("Training Dot Ball Generation Classifier...")
        self.clf_dot = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.06,
            num_leaves=31,
            max_depth=5,
            random_state=42,
            verbose=-1,
        )
        self.clf_dot.fit(X, y_dot)
        pred_dot_prob = self.clf_dot.predict_proba(X)[:, 1]
        auc_dot = float(roc_auc_score(y_dot, pred_dot_prob))
        brier_dot = float(brier_score_loss(y_dot, pred_dot_prob))
        logger.info("Bowler Dot Classifier: AUC = %.4f, Brier = %.4f", auc_dot, brier_dot)

        # 4. Boundary Conceded Classifier
        logger.info("Training Boundary Conceded Classifier...")
        self.clf_boundary = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.06,
            num_leaves=31,
            max_depth=5,
            random_state=42,
            verbose=-1,
        )
        self.clf_boundary.fit(X, y_boundary)
        pred_b_prob = self.clf_boundary.predict_proba(X)[:, 1]
        auc_b = float(roc_auc_score(y_boundary, pred_b_prob))
        brier_b = float(brier_score_loss(y_boundary, pred_b_prob))
        logger.info("Boundary Conceded Classifier: AUC = %.4f, Brier = %.4f", auc_b, brier_b)

        self.metrics = {
            "bowl_conceded_rmse": rmse_c,
            "bowl_conceded_mae": mae_c,
            "bowl_wicket_auc": auc_w,
            "bowl_dot_auc": auc_dot,
            "bowl_boundary_auc": auc_b,
        }

        # Save artifacts
        joblib.dump(self.reg_conceded, self.models_dir / "bowl_expected_conceded.joblib")
        joblib.dump(self.clf_wicket, self.models_dir / "bowl_clf_wicket.joblib")
        joblib.dump(self.clf_dot, self.models_dir / "bowl_clf_dot.joblib")
        joblib.dump(self.clf_boundary, self.models_dir / "bowl_clf_boundary.joblib")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates predictions for all deliveries:
          returns (exp_conceded, prob_wicket, prob_dot, prob_boundary)
        """
        X = self.feature_builder.build_features(df)
        exp_conceded = np.clip(self.reg_conceded.predict(X), 0.0, 7.0)
        prob_wicket = self.clf_wicket.predict_proba(X)[:, 1]
        prob_dot = self.clf_dot.predict_proba(X)[:, 1]
        prob_boundary = self.clf_boundary.predict_proba(X)[:, 1]
        return exp_conceded, prob_wicket, prob_dot, prob_boundary

    def generate_oof_predictions(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generates out-of-fold predictions for Expected Bowling metrics across
        strictly chronological, match-disjoint partitions.
        Every delivery is evaluated by models that never trained on its match.
        Trains final production models on all data.
        """
        logger.info("Generating Out-of-Fold Chronological Expected Bowling Predictions (%d folds)...", n_splits)
        n = len(df)
        oof_conceded = np.zeros(n, dtype=np.float64)
        oof_wicket = np.zeros(n, dtype=np.float64)
        oof_dot = np.zeros(n, dtype=np.float64)
        oof_boundary = np.zeros(n, dtype=np.float64)

        X = self.feature_builder.build_features(df)
        y_conceded, y_wicket = self.calculate_bowler_charge(df)
        y_c = y_conceded.to_numpy()
        y_w = y_wicket.to_numpy()
        y_d = (y_c == 0).astype(int)
        y_b = (y_c >= 4).astype(int)

        splits = get_chronological_kfold_splits(df, n_splits=n_splits)

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            logger.info("Bowling OOF Fold %d/%d (train=%d, val=%d)...", fold_idx + 1, len(splits), len(tr_idx), len(val_idx))
            X_tr, y_c_tr, y_w_tr, y_d_tr, y_b_tr = X.iloc[tr_idx], y_c[tr_idx], y_w[tr_idx], y_d[tr_idx], y_b[tr_idx]
            X_val = X.iloc[val_idx]

            # 1. Regressor
            reg = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.06, num_leaves=31, max_depth=6, random_state=42, verbose=-1)
            reg.fit(X_tr, y_c_tr)
            oof_conceded[val_idx] = np.clip(reg.predict(X_val), 0.0, 7.0)

            # 2. Wicket
            clf_w = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=25, max_depth=4, random_state=42, verbose=-1)
            clf_w.fit(X_tr, y_w_tr)
            oof_wicket[val_idx] = clf_w.predict_proba(X_val)[:, 1]

            # 3. Dot
            clf_d = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.06, num_leaves=31, max_depth=5, random_state=42, verbose=-1)
            clf_d.fit(X_tr, y_d_tr)
            oof_dot[val_idx] = clf_d.predict_proba(X_val)[:, 1]

            # 4. Boundary
            clf_b = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.06, num_leaves=31, max_depth=5, random_state=42, verbose=-1)
            clf_b.fit(X_tr, y_b_tr)
            oof_boundary[val_idx] = clf_b.predict_proba(X_val)[:, 1]

        # Train full production models
        logger.info("Training Production Bowling Models on all historical data...")
        self.train(df)

        # Baseline error comparisons
        baseline_mean = float(np.mean(y_c))
        baseline_pred = np.full(n, baseline_mean)
        base_mae = float(mean_absolute_error(y_c, baseline_pred))
        base_rmse = float(np.sqrt(mean_squared_error(y_c, baseline_pred)))

        oof_mae = float(mean_absolute_error(y_c, oof_conceded))
        oof_rmse = float(np.sqrt(mean_squared_error(y_c, oof_conceded)))
        oof_w_auc = float(roc_auc_score(y_w, oof_wicket))
        oof_d_auc = float(roc_auc_score(y_d, oof_dot))
        oof_b_auc = float(roc_auc_score(y_b, oof_boundary))

        metrics = {
            "oof_conceded_mae": oof_mae,
            "baseline_conceded_mae": base_mae,
            "mae_improvement_pct": round((1.0 - oof_mae / base_mae) * 100.0, 2),
            "oof_conceded_rmse": oof_rmse,
            "baseline_conceded_rmse": base_rmse,
            "rmse_improvement_pct": round((1.0 - oof_rmse / base_rmse) * 100.0, 2),
            "oof_wicket_auc": oof_w_auc,
            "oof_dot_auc": oof_d_auc,
            "oof_boundary_auc": oof_b_auc,
        }
        logger.info("Bowling OOF Metrics vs Baseline: MAE=%.4f (base=%.4f, +%.1f%%), RMSE=%.4f (base=%.4f, +%.1f%%)",
                    oof_mae, base_mae, metrics["mae_improvement_pct"],
                    oof_rmse, base_rmse, metrics["rmse_improvement_pct"])
        return oof_conceded, oof_wicket, oof_dot, oof_boundary, metrics
