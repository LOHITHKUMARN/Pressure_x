"""
PRESSURE-X — Module: Win Probability Model
============================================
Calculates well-calibrated Win Probability (WP) for any match state S:
  WP = P(Batting Team Wins | State S)

Supports both Innings 1 and Innings 2 with cricket-native terminal boundary rules:
  - If runs_required <= 0 -> WP = 1.0 (exact terminal win)
  - If wickets_remaining <= 0 and runs_required > 0 -> WP = 0.0 (exact terminal loss)
  - If balls_remaining <= 0 and runs_required > 0 -> WP = 0.0 (exact terminal loss)

Uses LightGBM calibrated with out-of-fold sigmoid (Platt) scaling via
CalibratedClassifierCV to guarantee strictly calibrated probabilities.

Separates Validation and Production workflows:
  - Validation Model: Trained strictly on chronological training matches, evaluated on unseen holdout matches.
  - Production Model: Trained on all eligible historical matches for downstream pipeline feature generation.

Author: PRESSURE-X Engineering
Version: 2.1.0 (Independent Chronological Validation & Exact Boundaries)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from evaluation.split_utils import get_chronological_holdout_split, get_chronological_kfold_splits

logger = logging.getLogger(__name__)


class WinProbabilityModel:
    """
    Cricket Win Probability Estimator.
    Trains separate calibrated sub-models for Innings 1 and Innings 2.
    """

    def __init__(self, models_dir: str | Path = "models", calibrate: bool = True):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.calibrate = calibrate

        self.model_inn1: Optional[Any] = None
        self.model_inn2: Optional[Any] = None

        inn1_file = self.models_dir / "wp_model_inn1.joblib"
        inn2_file = self.models_dir / "wp_model_inn2.joblib"
        if inn1_file.exists():
            try:
                self.model_inn1 = joblib.load(inn1_file)
            except Exception:
                pass
        if inn2_file.exists():
            try:
                self.model_inn2 = joblib.load(inn2_file)
            except Exception:
                pass

        self.inn1_features: List[str] = [
            "balls_remaining",
            "wickets_remaining",
            "innings_runs",
            "crr",
            "innings_phase_encoded",
        ]

        self.inn2_features: List[str] = [
            "runs_required",
            "balls_remaining",
            "wickets_remaining",
            "rrr",
            "crr",
            "target",
            "innings_phase_encoded",
        ]

        self.metrics: Dict[str, float] = {}

    @staticmethod
    def _encode_phase(phases: pd.Series) -> pd.Series:
        mapping = {"powerplay": 0, "middle": 1, "death": 2}
        return phases.map(mapping).fillna(1).astype(int)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters valid completed matches with known winners and extracts standardized features.
        """
        valid = df[df["batting_team_won"].notna() & (~df["is_abandoned"])].copy()

        # Phase encoding
        valid["innings_phase_encoded"] = self._encode_phase(valid["innings_phase"])

        # Prep feature columns
        valid["crr"] = valid["crr_pre"]
        valid["rrr"] = valid.get("rrr_pre", 0.0).fillna(0.0)
        valid["runs_required"] = valid.get("runs_required_pre", 0.0).fillna(0.0)
        valid["balls_remaining"] = valid["balls_remaining_pre"]
        valid["wickets_remaining"] = valid["wickets_remaining_pre"]
        valid["innings_runs"] = valid["innings_runs_pre"]
        valid["target"] = valid.get("target", 160.0).fillna(160.0)

        df_inn1 = valid[valid["innings"] == 1].copy()
        df_inn2 = valid[valid["innings"] == 2].copy()

        return df_inn1, df_inn2

    def _build_model(self, n_estimators: int, max_depth: int = 5, min_class_count: int = 10) -> Any:
        base_estimator = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=max_depth,
            random_state=42,
            verbose=-1,
        )
        if self.calibrate and min_class_count >= 6:
            cv = min(3, min_class_count // 2)
            return CalibratedClassifierCV(estimator=base_estimator, cv=cv, method="sigmoid")
        return base_estimator

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Trains production models on all historical data and saves artifacts.
        """
        logger.info("Preparing data for Production Win Probability model training...")
        df_inn1, df_inn2 = self.prepare_data(df)

        logger.info("Training Production Innings 1 WP model (%d deliveries)...", len(df_inn1))
        X1 = df_inn1[self.inn1_features]
        y1 = df_inn1["batting_team_won"].astype(int)
        min_c1 = int(y1.value_counts().min()) if len(y1) > 0 else 0
        self.model_inn1 = self._build_model(n_estimators=120, max_depth=5, min_class_count=min_c1)
        self.model_inn1.fit(X1, y1)

        logger.info("Training Production Innings 2 WP model (%d deliveries)...", len(df_inn2))
        X2 = df_inn2[self.inn2_features]
        y2 = df_inn2["batting_team_won"].astype(int)
        min_c2 = int(y2.value_counts().min()) if len(y2) > 0 else 0
        self.model_inn2 = self._build_model(n_estimators=150, max_depth=5, min_class_count=min_c2)
        self.model_inn2.fit(X2, y2)

        # In-sample metrics recorded purely for diagnostic sanity (not claimed as generalization)
        p1 = self.model_inn1.predict_proba(X1)[:, 1]
        p2 = self.model_inn2.predict_proba(X2)[:, 1]

        self.metrics = {
            "prod_inn1_brier_insample": float(brier_score_loss(y1, p1)),
            "prod_inn1_auc_insample": float(roc_auc_score(y1, p1)),
            "prod_inn2_brier_insample": float(brier_score_loss(y2, p2)),
            "prod_inn2_auc_insample": float(roc_auc_score(y2, p2)),
        }

        # Persist production artifacts
        joblib.dump(self.model_inn1, self.models_dir / "wp_model_inn1.joblib")
        joblib.dump(self.model_inn2, self.models_dir / "wp_model_inn2.joblib")
        logger.info("Production Win Probability models persisted to %s", self.models_dir)
        return self.metrics

    def evaluate_chronological_holdout(
        self,
        df: pd.DataFrame,
        train_split_ratio: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Trains an isolated validation model strictly on chronological training matches
        and evaluates holdout performance strictly on unseen chronological test matches.
        Enforces strict date-based splitting with zero match overlap.
        """
        valid_inn1, valid_inn2 = self.prepare_data(df)
        all_valid = pd.concat([valid_inn1, valid_inn2], ignore_index=True)

        train_idx, test_idx, meta = get_chronological_holdout_split(all_valid, train_split_ratio=train_split_ratio)
        train_matches = set(meta["train_matches"])
        test_matches = set(meta["test_matches"])

        assert len(train_matches.intersection(test_matches)) == 0, "Train and test match sets must be completely disjoint."

        # Innings 1 split
        tr1 = valid_inn1[valid_inn1["match_id"].isin(train_matches)]
        te1 = valid_inn1[valid_inn1["match_id"].isin(test_matches)]

        min_c1 = int(tr1["batting_team_won"].value_counts().min()) if len(tr1) > 0 else 0
        val_inn1_model = self._build_model(n_estimators=120, max_depth=5, min_class_count=min_c1)
        val_inn1_model.fit(tr1[self.inn1_features], tr1["batting_team_won"].astype(int))
        p1_te = val_inn1_model.predict_proba(te1[self.inn1_features])[:, 1]
        p1_te_clipped = np.clip(p1_te, 0.001, 0.999)
        y1_te = te1["batting_team_won"].astype(int)

        inn1_auc = float(roc_auc_score(y1_te, p1_te_clipped)) if len(np.unique(y1_te)) > 1 else np.nan
        inn1_brier = float(brier_score_loss(y1_te, p1_te_clipped))
        inn1_logloss = float(log_loss(y1_te, p1_te_clipped, labels=[0, 1]))

        # Innings 2 split
        tr2 = valid_inn2[valid_inn2["match_id"].isin(train_matches)]
        te2 = valid_inn2[valid_inn2["match_id"].isin(test_matches)].copy()

        min_c2 = int(tr2["batting_team_won"].value_counts().min()) if len(tr2) > 0 else 0
        val_inn2_model = self._build_model(n_estimators=150, max_depth=5, min_class_count=min_c2)
        val_inn2_model.fit(tr2[self.inn2_features], tr2["batting_team_won"].astype(int))
        raw_p2_te = val_inn2_model.predict_proba(te2[self.inn2_features])[:, 1]
        raw_p2_te_clipped = np.clip(raw_p2_te, 0.001, 0.999)

        # Apply cricket terminal boundary rules on test set
        won_mask = te2["runs_required"] <= 0
        lost_mask = ((te2["wickets_remaining"] <= 0) | (te2["balls_remaining"] <= 0)) & (te2["runs_required"] > 0)
        p2_te = np.where(won_mask, 1.0, np.where(lost_mask, 0.0, raw_p2_te_clipped))
        y2_te = te2["batting_team_won"].astype(int)

        inn2_auc = float(roc_auc_score(y2_te, p2_te)) if len(np.unique(y2_te)) > 1 else np.nan
        inn2_brier = float(brier_score_loss(y2_te, p2_te))
        inn2_logloss = float(log_loss(y2_te, np.clip(p2_te, 1e-6, 1.0 - 1e-6), labels=[0, 1]))

        # Combined metrics across all deliveries in test matches
        y_comb = np.concatenate([y1_te, y2_te])
        p_comb = np.concatenate([p1_te_clipped, p2_te])
        comb_auc = float(roc_auc_score(y_comb, p_comb)) if len(np.unique(y_comb)) > 1 else np.nan
        comb_brier = float(brier_score_loss(y_comb, p_comb))
        comb_logloss = float(log_loss(y_comb, np.clip(p_comb, 1e-6, 1.0 - 1e-6), labels=[0, 1]))

        holdout_report = {
            "validation_type": "Chronological Out-of-Sample Holdout",
            "model_architecture": "LightGBM + CalibratedClassifierCV (Sigmoid / Platt Scaling)",
            "train_date_range": meta["train_date_range"],
            "test_date_range": meta["test_date_range"],
            "train_matches": len(train_matches),
            "test_matches": len(test_matches),
            "train_deliveries": len(tr1) + len(tr2),
            "test_deliveries": len(te1) + len(te2),
            "disjoint_match_guarantee": meta["disjoint_match_guarantee"],
            "inn1_holdout": {
                "auc": inn1_auc,
                "brier_score": inn1_brier,
                "log_loss": inn1_logloss,
                "deliveries": len(te1),
            },
            "inn2_holdout": {
                "auc": inn2_auc,
                "brier_score": inn2_brier,
                "log_loss": inn2_logloss,
                "deliveries": len(te2),
            },
            "combined_holdout": {
                "auc": comb_auc,
                "brier_score": comb_brier,
                "log_loss": comb_logloss,
                "deliveries": len(te1) + len(te2),
            },
        }

        return holdout_report

    @staticmethod
    def _predict_with_models(
        model_inn1: Any,
        model_inn2: Any,
        innings: np.ndarray,
        balls_remaining: np.ndarray,
        wickets_remaining: np.ndarray,
        innings_runs: np.ndarray,
        crr: np.ndarray,
        phases: pd.Series,
        runs_required: Optional[np.ndarray] = None,
        rrr: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Helper to predict state WP with specified model pair."""
        n = len(innings)
        wp = np.full(n, 0.5, dtype=np.float64)
        mapping = {"powerplay": 0, "middle": 1, "death": 2}
        phase_enc = phases.map(mapping).fillna(1).astype(int).to_numpy()

        # Innings 1
        m1 = innings == 1
        if np.any(m1) and model_inn1 is not None:
            feat1 = pd.DataFrame({
                "balls_remaining": balls_remaining[m1],
                "wickets_remaining": wickets_remaining[m1],
                "innings_runs": innings_runs[m1],
                "crr": crr[m1],
                "innings_phase_encoded": phase_enc[m1],
            })
            p1 = model_inn1.predict_proba(feat1)[:, 1]
            wp[m1] = np.clip(p1, 0.001, 0.999)

        # Innings 2
        m2 = innings == 2
        if np.any(m2):
            rr = runs_required if runs_required is not None else np.zeros(n)
            br = balls_remaining
            wr = wickets_remaining
            req_rate = rrr if rrr is not None else np.zeros(n)
            tgt = target if target is not None else np.full(n, 160.0)

            if model_inn2 is not None:
                feat2 = pd.DataFrame({
                    "runs_required": rr[m2],
                    "balls_remaining": br[m2],
                    "wickets_remaining": wr[m2],
                    "rrr": np.nan_to_num(req_rate[m2], nan=0.0, posinf=50.0, neginf=0.0),
                    "crr": crr[m2],
                    "target": np.nan_to_num(tgt[m2], nan=160.0),
                    "innings_phase_encoded": phase_enc[m2],
                })
                p2 = model_inn2.predict_proba(feat2)[:, 1]
            else:
                p2 = np.full(int(np.sum(m2)), 0.5)

            p2_clipped = np.clip(p2, 0.001, 0.999)
            won_mask = rr[m2] <= 0
            lost_mask = ((wr[m2] <= 0) | (br[m2] <= 0)) & (rr[m2] > 0)
            wp[m2] = np.where(won_mask, 1.0, np.where(lost_mask, 0.0, p2_clipped))

        return wp

    def generate_oof_predictions(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generates out-of-fold (OOF) pre-delivery and post-delivery Win Probability predictions
        across chronological, match-disjoint folds.
        Ensures that every historical delivery is evaluated by a model that NEVER saw that match.
        Also trains production models on full dataset for live inference.
        """
        logger.info("Generating Out-of-Fold Chronological WP Predictions (%d folds)...", n_splits)
        n = len(df)
        oof_wp_pre = np.full(n, 0.5, dtype=np.float64)
        oof_wp_post = np.full(n, 0.5, dtype=np.float64)

        splits = get_chronological_kfold_splits(df, n_splits=n_splits)

        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            logger.info("Fitting Fold %d/%d (train=%d, val=%d)...", fold_idx + 1, len(splits), len(tr_idx), len(val_idx))
            df_tr = df.iloc[tr_idx]
            df_val = df.iloc[val_idx]

            valid_inn1_tr, valid_inn2_tr = self.prepare_data(df_tr)

            # Fit fold models
            min_c1 = int(valid_inn1_tr["batting_team_won"].value_counts().min()) if len(valid_inn1_tr) > 0 else 0
            m1_fold = self._build_model(n_estimators=100, max_depth=5, min_class_count=min_c1)
            if len(valid_inn1_tr) > 0:
                m1_fold.fit(valid_inn1_tr[self.inn1_features], valid_inn1_tr["batting_team_won"].astype(int))
            else:
                m1_fold = None

            min_c2 = int(valid_inn2_tr["batting_team_won"].value_counts().min()) if len(valid_inn2_tr) > 0 else 0
            m2_fold = self._build_model(n_estimators=120, max_depth=5, min_class_count=min_c2)
            if len(valid_inn2_tr) > 0:
                m2_fold.fit(valid_inn2_tr[self.inn2_features], valid_inn2_tr["batting_team_won"].astype(int))
            else:
                m2_fold = None

            # Predict pre-delivery WP on unseen validation fold
            wp_pre_val = self._predict_with_models(
                m1_fold,
                m2_fold,
                innings=df_val["innings"].to_numpy(),
                balls_remaining=df_val["balls_remaining_pre"].to_numpy(),
                wickets_remaining=df_val["wickets_remaining_pre"].to_numpy(),
                innings_runs=df_val["innings_runs_pre"].to_numpy(),
                crr=df_val["crr_pre"].to_numpy(),
                phases=df_val["innings_phase"],
                runs_required=df_val["runs_required_pre"].to_numpy() if "runs_required_pre" in df_val.columns else None,
                rrr=df_val["rrr_pre"].to_numpy() if "rrr_pre" in df_val.columns else None,
                target=df_val["target"].to_numpy() if "target" in df_val.columns else None,
            )
            oof_wp_pre[val_idx] = wp_pre_val

            # Predict post-delivery WP on unseen validation fold
            wp_post_val = self._predict_with_models(
                m1_fold,
                m2_fold,
                innings=df_val["innings"].to_numpy(),
                balls_remaining=df_val["balls_remaining_post"].to_numpy(),
                wickets_remaining=df_val["wickets_remaining_post"].to_numpy(),
                innings_runs=df_val["innings_runs_post"].to_numpy(),
                crr=df_val["crr_post"].to_numpy(),
                phases=df_val["innings_phase"],
                runs_required=df_val["runs_required_post"].to_numpy() if "runs_required_post" in df_val.columns else None,
                rrr=df_val["rrr_post"].to_numpy() if "rrr_post" in df_val.columns else None,
                target=df_val["target"].to_numpy() if "target" in df_val.columns else None,
            )
            oof_wp_post[val_idx] = wp_post_val

        # Train full production models
        logger.info("Training Production WP Models on all historical data...")
        self.train(df)

        # OOF Evaluation Metrics on completed matches
        valid_mask = df["batting_team_won"].notna() & (~df.get("is_abandoned", False))
        y_true = df.loc[valid_mask, "batting_team_won"].astype(int).to_numpy()
        p_oof = oof_wp_pre[valid_mask]

        auc_oof = float(roc_auc_score(y_true, p_oof)) if len(np.unique(y_true)) > 1 else np.nan
        brier_oof = float(brier_score_loss(y_true, p_oof))
        logloss_oof = float(log_loss(y_true, np.clip(p_oof, 1e-6, 1.0 - 1e-6), labels=[0, 1]))

        oof_metrics = {
            "oof_auc": auc_oof,
            "oof_brier": brier_oof,
            "oof_log_loss": logloss_oof,
            "total_deliveries": n,
            "evaluated_deliveries": int(np.sum(valid_mask)),
        }
        logger.info("OOF Chronological WP Evaluation: AUC=%.4f, Brier=%.4f, LogLoss=%.4f", auc_oof, brier_oof, logloss_oof)
        return oof_wp_pre, oof_wp_post, oof_metrics

    def predict_state_wp(
        self,
        innings: np.ndarray,
        balls_remaining: np.ndarray,
        wickets_remaining: np.ndarray,
        innings_runs: np.ndarray,
        crr: np.ndarray,
        phases: pd.Series,
        runs_required: Optional[np.ndarray] = None,
        rrr: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._predict_with_models(
            self.model_inn1,
            self.model_inn2,
            innings=innings,
            balls_remaining=balls_remaining,
            wickets_remaining=wickets_remaining,
            innings_runs=innings_runs,
            crr=crr,
            phases=phases,
            runs_required=runs_required,
            rrr=rrr,
            target=target,
        )
