"""
drift_detector.py - Customer blend vectors and behavioral drift scoring.

This module computes:
  - Pattern blend vectors: how much of each pattern a customer expresses
  - Drift scores: how much a customer's blend has changed between cycles
  - Risk levels: data-driven thresholds derived from the drift distribution
  - Pattern trajectories: trend direction across multiple scoring cycles

Dependencies: numpy, scipy, pandas

All numeric thresholds are exposed as named function parameters.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_distance
from scipy import stats

logger = logging.getLogger("nmf_prototype")


def compute_blends(
    W_matrix: np.ndarray,
    customer_ids: List[str],
    pattern_labels: List[str],
) -> pd.DataFrame:
    """
    Normalise customer-pattern weights to produce blend vectors.

    Each row of W represents a customer's raw weights across K patterns.
    This function normalises each row so weights sum to 1, producing a
    percentage blend (e.g., 40% Engagement, 35% Revenue, 25% Risk).

    Args:
        W_matrix: Customer x pattern weight matrix from NMF (n x k).
        customer_ids: List of customer IDs matching W rows.
        pattern_labels: List of pattern label strings matching W columns.

    Returns:
        DataFrame with customer_id as index, pattern labels as columns.
        Each cell is a float 0-1 representing the fraction of that pattern
        the customer expresses.
    """
    logger.info("Computing blend vectors for %d customers ...", len(customer_ids))

    # Normalise rows so they sum to 1
    row_sums = W_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for customers with zero-weight rows
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    normalised = W_matrix / row_sums

    blends_df = pd.DataFrame(
        normalised,
        index=customer_ids,
        columns=pattern_labels,
    )
    blends_df.index.name = "customer_id"

    logger.info("  [OK] Blend vectors computed. Shape: %d x %d", *blends_df.shape)
    return blends_df


def compute_drift_scores(
    current_blends: pd.DataFrame,
    previous_blends: Optional[pd.DataFrame] = None,
    high_risk_std_multiplier: float = 1.5,
    medium_risk_std_multiplier: float = 0.5,
) -> pd.DataFrame:
    """
    Compute behavioral drift scores between two scoring cycles.

    For each customer present in both cycles, computes cosine distance
    between their current and previous blend vectors.

    Risk thresholds are derived from the actual distribution of drift scores
    (data-driven, not hardcoded):
      - HIGH RISK: drift_score > mean + high_risk_std_multiplier x std
      - MEDIUM RISK: mean + medium_risk_std_multiplier x std < drift <= HIGH
      - LOW RISK: everything else

    Args:
        current_blends: DataFrame from compute_blends() for the current cycle.
        previous_blends: DataFrame from a previous cycle, or None for first run.
        high_risk_std_multiplier: Number of standard deviations above the mean
            that defines the HIGH RISK threshold. Default 1.5 - this means
            roughly the top 7% of drifters (assuming normal distribution).
            Chosen to flag meaningful outliers without being too aggressive.
        medium_risk_std_multiplier: Number of standard deviations above the
            mean that defines the MEDIUM RISK lower bound. Default 0.5 -
            this captures approximately the top 30% of drifters.

    Returns:
        DataFrame with columns: customer_id, drift_score, risk_level,
        pattern_moved_from, pattern_moved_toward, threshold_high,
        threshold_medium, has_previous_state.

    PASS result (for validation): Mean drift of churned > mean drift of retained.
    """
    if previous_blends is None:
        logger.info("No previous blends - returning zero drift (first run baseline)")
        result = pd.DataFrame({
            "customer_id": current_blends.index,
            "drift_score": 0.0,
            "risk_level": "LOW",
            "pattern_moved_from": "N/A",
            "pattern_moved_toward": "N/A",
            "threshold_high": np.nan,
            "threshold_medium": np.nan,
            "has_previous_state": False,
            "is_cold_start": True,
            "scoring_mode": "COLD_START_BASELINE",
        })
        result.set_index("customer_id", inplace=True)
        return result

    logger.info("Computing drift scores between cycles ...")

    # Find customers present in both cycles
    common_ids = current_blends.index.intersection(previous_blends.index)
    # Align columns (patterns) - use current columns; missing ones in previous = 0
    all_cols = current_blends.columns.tolist()

    records = []
    for cust_id in common_ids:
        curr_vec = current_blends.loc[cust_id, all_cols].values.astype(float)
        prev_vec = np.zeros(len(all_cols))
        for i, col in enumerate(all_cols):
            if col in previous_blends.columns:
                prev_vec[i] = previous_blends.loc[cust_id, col]

        # Cosine distance: 1 - cosine_similarity (0 = identical, 2 = opposite)
        if np.linalg.norm(curr_vec) == 0 or np.linalg.norm(prev_vec) == 0:
            drift = 0.0
        else:
            drift = float(cosine_distance(curr_vec, prev_vec))

        # Determine which pattern decreased most and which increased most
        diff = curr_vec - prev_vec
        moved_from = all_cols[int(np.argmin(diff))] if len(diff) > 0 else "N/A"
        moved_toward = all_cols[int(np.argmax(diff))] if len(diff) > 0 else "N/A"

        records.append({
            "customer_id": cust_id,
            "drift_score": drift,
            "pattern_moved_from": moved_from,
            "pattern_moved_toward": moved_toward,
            "has_previous_state": True,
            "is_cold_start": False,
            "scoring_mode": "NMF_DRIFT",
        })

    # Add customers only in current (new customers - Cold Start)
    new_ids = current_blends.index.difference(previous_blends.index)
    for cust_id in new_ids:
        records.append({
            "customer_id": cust_id,
            "drift_score": 0.0,
            "pattern_moved_from": "N/A (new customer)",
            "pattern_moved_toward": "N/A (new customer)",
            "has_previous_state": False,
            "is_cold_start": True,
            "scoring_mode": "COLD_START_BASELINE",
        })

    drift_df = pd.DataFrame(records)
    drift_df.set_index("customer_id", inplace=True)

    # Derive risk thresholds from the actual distribution
    scores = drift_df.loc[drift_df["has_previous_state"], "drift_score"]

    if len(scores) > 0 and scores.std() > 0:
        mean_drift = float(scores.mean())
        std_drift = float(scores.std())
        threshold_high = mean_drift + high_risk_std_multiplier * std_drift
        threshold_medium = mean_drift + medium_risk_std_multiplier * std_drift
    else:
        mean_drift = 0.0
        std_drift = 0.0
        threshold_high = 1.0  # Effectively no one flagged
        threshold_medium = 0.5

    drift_df["threshold_high"] = threshold_high
    drift_df["threshold_medium"] = threshold_medium

    # Assign risk levels
    drift_df["risk_level"] = "LOW"
    drift_df.loc[drift_df["drift_score"] > threshold_medium, "risk_level"] = "MEDIUM"
    drift_df.loc[drift_df["drift_score"] > threshold_high, "risk_level"] = "HIGH"

    # Report
    risk_counts = drift_df["risk_level"].value_counts()
    logger.info("  Drift score distribution:")
    logger.info("    Mean: %.4f | Std: %.4f", mean_drift, std_drift)
    logger.info("    HIGH threshold: %.4f | MEDIUM threshold: %.4f",
                threshold_high, threshold_medium)
    logger.info("    Risk breakdown: %s",
                {level: int(count) for level, count in risk_counts.items()})

    return drift_df


def identify_pattern_trajectory(
    blends_over_time: Dict[str, pd.DataFrame],
    slope_threshold: float = 0.01,
) -> Dict[str, Dict[str, str]]:
    """
    Identify trend direction for each customer in each pattern over time.

    Uses linear regression on the time series of each pattern's weight
    to determine if the customer is increasing, decreasing, or stable
    in that pattern.

    Args:
        blends_over_time: Dict of {date_string: blend_DataFrame} across
            multiple scoring cycles. Must have at least 2 time points.
        slope_threshold: Minimum absolute slope of the linear regression
            to consider a trend "increasing" or "decreasing". Below this,
            the trend is "stable". Default 0.01 - a 1 percentage-point
            change per time step is considered significant enough to flag
            as a directional trend.

    Returns:
        Dict of {customer_id: {pattern_label: trend_direction}} where
        trend_direction is "increasing", "decreasing", or "stable".
    """
    logger.info("Computing pattern trajectories across %d time points ...",
                len(blends_over_time))

    if len(blends_over_time) < 2:
        logger.warning("  Need at least 2 time points for trajectory analysis.")
        return {}

    sorted_dates = sorted(blends_over_time.keys())
    time_indices = list(range(len(sorted_dates)))

    # Collect all customer IDs and pattern labels
    all_customers = set()
    pattern_labels = None
    for date_key in sorted_dates:
        df = blends_over_time[date_key]
        all_customers.update(df.index.tolist())
        if pattern_labels is None:
            pattern_labels = df.columns.tolist()

    if not pattern_labels:
        return {}

    trajectories = {}

    for cust_id in all_customers:
        cust_trajectories = {}

        for pattern in pattern_labels:
            values = []
            times = []

            for t_idx, date_key in enumerate(sorted_dates):
                df = blends_over_time[date_key]
                if cust_id in df.index and pattern in df.columns:
                    values.append(float(df.loc[cust_id, pattern]))
                    times.append(t_idx)

            if len(values) < 2:
                cust_trajectories[pattern] = "stable"
                continue

            # Linear regression
            slope, _, _, _, _ = stats.linregress(times, values)

            if slope > slope_threshold:
                cust_trajectories[pattern] = "increasing"
            elif slope < -slope_threshold:
                cust_trajectories[pattern] = "decreasing"
            else:
                cust_trajectories[pattern] = "stable"

        trajectories[str(cust_id)] = cust_trajectories

    logger.info("  [OK] Trajectories computed for %d customers", len(trajectories))
    return trajectories
