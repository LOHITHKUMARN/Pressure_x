"""
PRESSURE-X — Phase 4: Expected Batting Performance Intelligence
================================================================
Models what an equivalent batter would be expected to score in each exact match situation:
  1. Expected Runs (exp_runs_bat): Continuous expectation E[Runs | S_t]
  2. Boundary Probability (prob_boundary_bat): P(Runs >= 4 | S_t)
  3. Dot-ball Probability (prob_dot_bat): P(Runs == 0 | S_t)
  4. Dismissal Probability (prob_wicket_bat): P(Dismissal | S_t)

Uses LightGBM regression and classification models trained on strictly causal features.
Provides evaluation metrics (RMSE, MAE, LogLoss, Brier Score, AUC).

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


class ExpectedBattingModel:
    """
    Batting Expected Performance Suite.
    """

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.feature_builder = FeatureBuilder()

        self.reg_runs: Optional[lgb.LGBMRegressor] = None
        self.clf_boundary: Optional[lgb.LGBMClassifier] = None
        self.clf_dot: Optional[lgb.LGBMClassifier] = None
        self.clf_wicket: Optional[lgb.LGBMClassifier] = None

        self.metrics: Dict[str, float] = {}

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Trains all expected batting models on delivery records.
        """
        logger.info("Extracting features for Batting Expected Performance models...")
        X = self.feature_builder.build_features(df)

        # Targets
        y_runs = df["runs"].astype(float)
        y_boundary = (df["runs"] >= 4).astype(int)
        y_dot = (df["runs"] == 0).astype(int)
        y_wicket = df["_wicket"].astype(int)

        # 1. Expected Runs Regressor
        logger.info("Training Expected Runs Regressor (%d rows)...", len(X))
        self.reg_runs = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.06,
            num_leaves=31,
            max_depth=6,
            random_state=42,
            verbose=-1,
        )
        self.reg_runs.fit(X, y_runs)
        pred_runs = self.reg_runs.predict(X)
        rmse_runs = float(np.sqrt(mean_squared_error(y_runs, pred_runs)))
        mae_runs = float(mean_absolute_error(y_runs, pred_runs))
        logger.info("Expected Runs: RMSE = %.4f, MAE = %.4f (Baseline mean = %.3f)", rmse_runs, mae_runs, y_runs.mean())

        # 2. Boundary Probability Classifier
        logger.info("Training Boundary Probability Classifier...")
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
        logger.info("Boundary Classifier: AUC = %.4f, Brier = %.4f", auc_b, brier_b)

        # 3. Dot Ball Classifier
        logger.info("Training Dot Ball Classifier...")
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
        logger.info("Dot Classifier: AUC = %.4f, Brier = %.4f", auc_dot, brier_dot)

        # 4. Dismissal Classifier
        logger.info("Training Dismissal Risk Classifier...")
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
        logger.info("Dismissal Classifier: AUC = %.4f, Brier = %.4f", auc_w, brier_w)

        self.metrics = {
            "bat_runs_rmse": rmse_runs,
            "bat_runs_mae": mae_runs,
            "bat_boundary_auc": auc_b,
            "bat_dot_auc": auc_dot,
            "bat_wicket_auc": auc_w,
        }

        # Save artifacts
        joblib.dump(self.reg_runs, self.models_dir / "bat_expected_runs.joblib")
        joblib.dump(self.clf_boundary, self.models_dir / "bat_clf_boundary.joblib")
        joblib.dump(self.clf_dot, self.models_dir / "bat_clf_dot.joblib")
        joblib.dump(self.clf_wicket, self.models_dir / "bat_clf_wicket.joblib")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates predictions for all deliveries:
          returns (exp_runs, prob_boundary, prob_dot, prob_wicket)
        """
        X = self.feature_builder.build_features(df)
        exp_runs = np.clip(self.reg_runs.predict(X), 0.0, 6.0)
        prob_boundary = self.clf_boundary.predict_proba(X)[:, 1]
        prob_dot = self.clf_dot.predict_proba(X)[:, 1]
        prob_wicket = self.clf_wicket.predict_proba(X)[:, 1]
        return exp_runs, prob_boundary, prob_dot, prob_wicket

    def generate_oof_predictions(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generates out-of-fold predictions for Expected Batting metrics across
        strictly chronological, match-disjoint partitions.
        Every delivery is evaluated by models that never trained on its match.
        Trains final production models on all data.
        """
        logger.info("Generating Out-of-Fold Chronological Expected Batting Predictions (%d folds)...", n_splits)
        n = len(df)
        oof_runs = np.zeros(n, dtype=np.float64)
        oof_boundary = np.zeros(n, dtype=np.float64)
        oof_dot = np.zeros(n, dtype=np.float64)
        oof_wicket = np.zeros(n, dtype=np.float64)

        X = self.feature_builder.build_features(df)
        y_runs = df["runs"].astype(float).to_numpy()
        y_boundary = (df["runs"] >= 4).astype(int).to_numpy()
        y_dot = (df["runs"] == 0).astype(int).to_numpy()
        y_wicket = df["_wicket"].astype(int).to_numpy()

        splits = get_chronological_kfold_splits(df, n_splits=n_splits)

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            logger.info("Batting OOF Fold %d/%d (train=%d, val=%d)...", fold_idx + 1, len(splits), len(tr_idx), len(val_idx))
            X_tr, y_r_tr, y_b_tr, y_d_tr, y_w_tr = X.iloc[tr_idx], y_runs[tr_idx], y_boundary[tr_idx], y_dot[tr_idx], y_wicket[tr_idx]
            X_val = X.iloc[val_idx]

            # 1. Regressor
            reg = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.06, num_leaves=31, max_depth=6, random_state=42, verbose=-1)
            reg.fit(X_tr, y_r_tr)
            oof_runs[val_idx] = np.clip(reg.predict(X_val), 0.0, 6.0)

            # 2. Boundary
            clf_b = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.06, num_leaves=31, max_depth=5, random_state=42, verbose=-1)
            clf_b.fit(X_tr, y_b_tr)
            oof_boundary[val_idx] = clf_b.predict_proba(X_val)[:, 1]

            # 3. Dot
            clf_d = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.06, num_leaves=31, max_depth=5, random_state=42, verbose=-1)
            clf_d.fit(X_tr, y_d_tr)
            oof_dot[val_idx] = clf_d.predict_proba(X_val)[:, 1]

            # 4. Wicket
            clf_w = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=25, max_depth=4, random_state=42, verbose=-1)
            clf_w.fit(X_tr, y_w_tr)
            oof_wicket[val_idx] = clf_w.predict_proba(X_val)[:, 1]

        # Train full production models
        logger.info("Training Production Batting Models on all historical data...")
        self.train(df)

        # Baseline error comparisons
        baseline_mean = float(np.mean(y_runs))
        baseline_pred = np.full(n, baseline_mean)
        base_mae = float(mean_absolute_error(y_runs, baseline_pred))
        base_rmse = float(np.sqrt(mean_squared_error(y_runs, baseline_pred)))

        oof_mae = float(mean_absolute_error(y_runs, oof_runs))
        oof_rmse = float(np.sqrt(mean_squared_error(y_runs, oof_runs)))
        oof_b_auc = float(roc_auc_score(y_boundary, oof_boundary))
        oof_d_auc = float(roc_auc_score(y_dot, oof_dot))
        oof_w_auc = float(roc_auc_score(y_wicket, oof_wicket))

        metrics = {
            "oof_runs_mae": oof_mae,
            "baseline_runs_mae": base_mae,
            "mae_improvement_pct": round((1.0 - oof_mae / base_mae) * 100.0, 2),
            "oof_runs_rmse": oof_rmse,
            "baseline_runs_rmse": base_rmse,
            "rmse_improvement_pct": round((1.0 - oof_rmse / base_rmse) * 100.0, 2),
            "oof_boundary_auc": oof_b_auc,
            "oof_dot_auc": oof_d_auc,
            "oof_wicket_auc": oof_w_auc,
        }
        logger.info("Batting OOF Metrics vs Baseline: MAE=%.4f (base=%.4f, +%.1f%%), RMSE=%.4f (base=%.4f, +%.1f%%)",
                    oof_mae, base_mae, metrics["mae_improvement_pct"],
                    oof_rmse, base_rmse, metrics["rmse_improvement_pct"])
        return oof_runs, oof_boundary, oof_dot, oof_wicket, metrics
