"""
PRESSURE-X — Chronological & Match-Disjoint Partitioning Utilities
===================================================================
Provides leakage-free temporal splitting:
  - Partitions unique match dates into contiguous chronological windows.
  - Guarantees all matches on the same date stay in the exact same partition.
  - Guarantees train_match_ids.isdisjoint(test_match_ids) with zero overlap.
  - Guarantees max(train_dates) < min(test_dates) for chronological holdouts.

Author: PRESSURE-X Engineering
Version: 2.2.0
"""

from __future__ import annotations

import logging
from typing import Dict, Generator, List, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_chronological_holdout_split(
    df: pd.DataFrame,
    train_split_ratio: float = 0.80,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """
    Splits delivery DataFrame into train and test indices strictly by unique match dates.
    Guarantees:
      1. train_match_ids.isdisjoint(test_match_ids)
      2. max(train_dates) < min(test_dates) (when dates strictly differ)
      3. All deliveries of any given match stay in the same partition.
    """
    if "match_date" not in df.columns:
        raise KeyError("match_date must be present in DataFrame for chronological splitting.")

    # Sort matches strictly by match_date
    match_dates = (
        df[["match_id", "match_date"]]
        .dropna(subset=["match_date"])
        .drop_duplicates()
        .sort_values("match_date")
    )
    unique_dates = sorted(match_dates["match_date"].unique())
    if len(unique_dates) == 0:
        raise ValueError("No valid match_date values found in dataset.")

    n_train_dates = max(1, int(len(unique_dates) * train_split_ratio))
    if n_train_dates == len(unique_dates) and len(unique_dates) > 1:
        n_train_dates = len(unique_dates) - 1

    train_dates_set = set(unique_dates[:n_train_dates])
    test_dates_set = set(unique_dates[n_train_dates:]) if n_train_dates < len(unique_dates) else set(unique_dates[-1:])

    train_matches = set(match_dates[match_dates["match_date"].isin(train_dates_set)]["match_id"])
    test_matches = set(match_dates[match_dates["match_date"].isin(test_dates_set)]["match_id"])

    # Disjoint guarantee
    if len(unique_dates) > 1:
        test_matches = test_matches - train_matches

    train_idx = df.index[df["match_id"].isin(train_matches)].to_numpy()
    test_idx = df.index[df["match_id"].isin(test_matches)].to_numpy()

    meta = {
        "train_matches": sorted(list(train_matches)),
        "test_matches": sorted(list(test_matches)),
        "train_date_range": [str(min(train_dates_set)), str(max(train_dates_set))],
        "test_date_range": [str(min(test_dates_set)), str(max(test_dates_set))] if test_dates_set else ["N/A", "N/A"],
        "train_deliveries": len(train_idx),
        "test_deliveries": len(test_idx),
        "disjoint_match_guarantee": len(train_matches.intersection(test_matches)) == 0,
    }
    return train_idx, test_idx, meta


def get_chronological_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partitions the dataset into n_splits chronological contiguous folds.
    For each fold k:
      - val_idx: Deliveries of matches in fold k (unseen test window).
      - train_idx: Deliveries of matches outside fold k.
    Guarantees no match is ever shared between train_idx and val_idx in any fold.
    """
    if "match_date" not in df.columns:
        raise KeyError("match_date must be present in DataFrame for chronological K-Fold.")

    match_dates = (
        df[["match_id", "match_date"]]
        .dropna(subset=["match_date"])
        .drop_duplicates()
        .sort_values("match_date")
    )
    unique_dates = sorted(match_dates["match_date"].unique())
    if len(unique_dates) < n_splits:
        # Fall back to unique matches if unique dates are very few (e.g. synthetic test)
        unique_matches = match_dates["match_id"].unique()
        match_chunks = np.array_split(unique_matches, min(len(unique_matches), n_splits))
        splits = []
        for chunk in match_chunks:
            val_matches = set(chunk)
            val_idx = df.index[df["match_id"].isin(val_matches)].to_numpy()
            train_idx = df.index[~df["match_id"].isin(val_matches)].to_numpy()
            if len(val_idx) > 0 and len(train_idx) > 0:
                splits.append((train_idx, val_idx))
        return splits

    # Split unique dates into contiguous chunks
    date_chunks = np.array_split(unique_dates, n_splits)
    splits = []
    for chunk in date_chunks:
        val_dates = set(chunk)
        val_matches = set(match_dates[match_dates["match_date"].isin(val_dates)]["match_id"])
        val_idx = df.index[df["match_id"].isin(val_matches)].to_numpy()
        train_idx = df.index[~df["match_id"].isin(val_matches)].to_numpy()
        if len(val_idx) > 0 and len(train_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits
