"""
data_loader.py - Load and prepare feature data from the existing database.

This module reads from four existing tables:
  - ml_db.tenant_feature_config  (active tenants and their selected features)
  - ml_db.ml_feature_store       (pre-calculated feature matrix per customer)
  - ml_db.ml_feedback            (churn outcome labels)
  - ml_db.feature_registry       (feature metadata and categories)

It never writes to any production table. All database queries use parameterised
placeholders to prevent SQL injection.

Key design decisions:
  - ml_feature_history does NOT exist in this database, so ml_feature_store
    is the sole data source for the feature matrix.
  - Features listed in tenant_feature_config.selected_features that are NOT
    columns in ml_feature_store are looked up in the custom_features JSON column.
  - Feature names are normalised to lowercase for consistent matching.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import get_db_connection

logger = logging.getLogger("nmf_prototype")

# ---------------------------------------------------------------------------
# Columns present as first-class columns in ml_feature_store.
# Derived from the actual DESCRIBE output of the table.
# ---------------------------------------------------------------------------
FEATURE_STORE_COLUMNS = [
    "customer_age_days",
    "tickets_last_30_days",
    "days_since_last_ticket",
    "avg_reopen_count",
    "sla_breaches",
    "days_since_last_purchase",
    "total_revenue",
    "active_subscriptions",
    "avg_product_amount",
    "days_since_last_meeting",
    "meetings_last_90_days",
    "lost_opportunities",
    "red_flags_count",
    "primary_contacts",
    "engagement_activities_last_60_days",
    "account_notes_last_30_days",
    "key_account",
]

# Map of common aliases / case variations to the canonical column name
_FEATURE_ALIASES = {
    "keyaccount": "key_account",
    "key_account": "key_account",
}


def _normalise_feature_name(name: str) -> str:
    """
    Normalise a feature name to lowercase with underscores.

    Handles the mismatch between tenant_feature_config entries like
    'KeyAccount' and the actual column name 'key_account'.
    """
    lower = name.strip().lower()
    return _FEATURE_ALIASES.get(lower, lower)


# ===================================================================
# Public API
# ===================================================================

def get_active_tenants() -> List[str]:
    """
    Read tenant_feature_config where is_active = 1.

    Returns:
        List of tenant_id strings for all active tenants.
    """
    logger.info("Querying active tenants from tenant_feature_config ...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT tenant_id FROM tenant_feature_config WHERE is_active = %s",
        (1,),
    )
    tenants = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    logger.info("  Found %d active tenant(s): %s", len(tenants), tenants)
    return tenants


def get_active_features_for_tenant(tenant_id: str) -> List[str]:
    """
    Read the selected_features JSON from tenant_feature_config for a tenant.

    Returns:
        List of feature name strings (normalised to lowercase).

    Raises:
        ValueError: If no config row exists for the tenant.
    """
    logger.info("Loading feature config for tenant '%s' ...", tenant_id)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT selected_features FROM tenant_feature_config "
        "WHERE tenant_id = %s AND is_active = %s",
        (tenant_id, 1),
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        raise ValueError(
            f"No active tenant_feature_config row for tenant_id='{tenant_id}'"
        )

    raw = row[0]
    if isinstance(raw, str):
        features = json.loads(raw)
    elif isinstance(raw, (list, tuple)):
        features = list(raw)
    else:
        # MySQL connector may return a JSON column as a Python object
        features = list(raw)

    normalised = [_normalise_feature_name(f) for f in features]
    logger.info("  Selected features (%d): %s", len(normalised), normalised)
    return normalised


def load_feature_matrix(
    tenant_id: str,
    null_warning_threshold: float = 0.3,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build a customer x feature matrix from ml_feature_store.

    Steps:
      1. Read selected features for the tenant
      2. Query ml_feature_store for all customers of that tenant
      3. For features not in first-class columns, extract from custom_features JSON
      4. Warn about high null rates (above null_warning_threshold)
      5. Replace nulls with 0 (after the warning)
      6. Ensure all values are non-negative (abs + warn for negatives)

    Args:
        tenant_id: The tenant to load data for.
        null_warning_threshold: Fraction (0-1) above which a null-rate
            warning is printed. Named parameter, not a buried literal.
            0.3 = 30% chosen as a reasonable default for sparse feature data.

    Returns:
        Tuple of:
          - DataFrame with customer_id as index, features as columns
          - data_quality_report dict with: total_customers, features_available,
            null_rates (per feature), features_from_json (list),
            negative_features_corrected (list)
    """
    logger.info("=" * 60)
    logger.info("Loading feature matrix for tenant '%s' ...", tenant_id)
    logger.info("=" * 60)

    selected = get_active_features_for_tenant(tenant_id)

    # Separate features into first-class columns vs. custom_features JSON
    sql_columns = []
    json_features = []
    for feat in selected:
        if feat in FEATURE_STORE_COLUMNS:
            sql_columns.append(feat)
        else:
            json_features.append(feat)

    if json_features:
        logger.info(
            "  Features from custom_features JSON: %s", json_features
        )

    # -- Query first-class columns ----------------------------------------
    cols_clause = ", ".join([f"`{c}`" for c in sql_columns])
    extra_cols = ", `custom_features`" if json_features else ""
    query = (
        f"SELECT `customer_id`, {cols_clause}{extra_cols} "
        f"FROM `ml_feature_store` WHERE `tenant_id` = %s"
    )

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, (tenant_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        logger.warning("  No customers found in ml_feature_store for '%s'", tenant_id)
        empty_df = pd.DataFrame(columns=selected)
        empty_df.index.name = "customer_id"
        return empty_df, {
            "total_customers": 0,
            "features_available": selected,
            "null_rates": {},
            "features_from_json": json_features,
            "negative_features_corrected": [],
        }

    df = pd.DataFrame(rows)
    df.set_index("customer_id", inplace=True)

    # -- Extract custom_features JSON columns ------------------------------
    if json_features and "custom_features" in df.columns:
        for feat in json_features:
            df[feat] = df["custom_features"].apply(
                lambda x: _extract_json_feature(x, feat)
            )
        df.drop(columns=["custom_features"], inplace=True, errors="ignore")
    elif "custom_features" in df.columns:
        df.drop(columns=["custom_features"], inplace=True, errors="ignore")

    # Keep only selected features in the correct order
    available = [f for f in selected if f in df.columns]
    missing = [f for f in selected if f not in df.columns]
    if missing:
        logger.warning(
            "  [!] Features configured but NOT found in data: %s - skipping them",
            missing,
        )
    df = df[available]

    # Convert to float (handles Decimal types from MySQL)
    df = df.astype(float)

    logger.info("  Raw matrix: %d customers x %d features", *df.shape)

    # -- Null analysis and handling ----------------------------------------
    null_rates = df.isnull().mean()
    high_nulls = null_rates[null_rates > null_warning_threshold]

    if not high_nulls.empty:
        logger.warning("  [!] HIGH NULL RATES (above %.0f%%):", null_warning_threshold * 100)
        for feat_name, rate in high_nulls.items():
            logger.warning("      %s: %.1f%% null", feat_name, rate * 100)

    # Replace NaN with 0 AFTER warning
    null_count_before = df.isnull().sum().sum()
    df.fillna(0.0, inplace=True)
    if null_count_before > 0:
        logger.info("  Replaced %d null values with 0", null_count_before)

    # -- Non-negativity enforcement ----------------------------------------
    negative_features = []
    for col in df.columns:
        neg_mask = df[col] < 0
        if neg_mask.any():
            neg_count = neg_mask.sum()
            negative_features.append(col)
            logger.warning(
                "  [!] Feature '%s' has %d negative values - applying abs()",
                col, neg_count,
            )
            df[col] = df[col].abs()

    # -- Feature Scaling to [0, 1] (Max-Scaling) --------------------------
    # Crucial for NMF to prevent large-magnitude features (e.g. revenue)
    # from dominating features with smaller scales (e.g. support tickets).
    # Max-Scaling is used instead of Min-Max Scaling to preserve sparsity (0s remain 0s).
    logger.info("  Scaling features to [0, 1] using Max-Scaling...")
    scaled_df = df.copy()
    for col in scaled_df.columns:
        max_val = scaled_df[col].max()
        if max_val > 0:
            scaled_df[col] = scaled_df[col] / max_val
        else:
            scaled_df[col] = 0.0
    df = scaled_df

    # -- Quality report ----------------------------------------------------
    quality_report = {
        "total_customers": len(df),
        "features_available": available,
        "features_missing": missing,
        "null_rates": null_rates.to_dict(),
        "features_from_json": json_features,
        "negative_features_corrected": negative_features,
    }

    logger.info(
        "  [OK] Final scaled matrix: %d customers x %d features", *df.shape
    )
    return df, quality_report


def load_feedback_data(
    tenant_id: str,
    min_feedback_warning: int = 5,
) -> pd.DataFrame:
    """
    Read ml_feedback for a tenant.

    Args:
        tenant_id: The tenant to load feedback for.
        min_feedback_warning: If fewer than this many rows exist, print a
            warning. The prototype still runs - it just cannot validate
            against outcomes. Default 5 per the spec; named parameter.

    Returns:
        DataFrame with columns: customer_id, actual_churned, feedback_date.
        Empty DataFrame if no feedback exists.
    """
    logger.info("Loading feedback data for tenant '%s' ...", tenant_id)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT customer_id, actual_churned, feedback_date "
        "FROM ml_feedback WHERE tenant_id = %s "
        "ORDER BY feedback_date DESC",
        (tenant_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        logger.warning(
            "  [!] No feedback data found for tenant '%s'. "
            "Outcome validation will be skipped.",
            tenant_id,
        )
        return pd.DataFrame(columns=["customer_id", "actual_churned", "feedback_date"])

    df = pd.DataFrame(rows)

    if len(df) < min_feedback_warning:
        logger.warning(
            "  [!] Only %d feedback rows for tenant '%s' (minimum recommended: %d). "
            "Outcome validation results may be unreliable.",
            len(df), tenant_id, min_feedback_warning,
        )

    churned_count = (df["actual_churned"] == 1).sum()
    retained_count = (df["actual_churned"] == 0).sum()
    logger.info(
        "  Feedback loaded: %d rows (%d churned, %d retained)",
        len(df), churned_count, retained_count,
    )
    return df


def load_feature_registry() -> pd.DataFrame:
    """
    Read feature_registry for category metadata.

    Used by pattern_engine.auto_label_patterns() to enrich pattern labels
    with known feature categories (engagement, revenue, support, risk, etc.).

    Returns:
        DataFrame with columns: feature_name, feature_category.
    """
    logger.info("Loading feature registry ...")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT feature_name, feature_category FROM feature_registry "
        "WHERE is_active = %s",
        (1,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        logger.warning("  No feature_registry entries found.")
        return pd.DataFrame(columns=["feature_name", "feature_category"])

    df = pd.DataFrame(rows)
    logger.info("  Loaded %d registered features", len(df))
    return df


def load_feature_snapshots_by_date(
    tenant_id: str,
    before_date: Optional[str] = None,
    after_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load feature matrices split by last_calculated_at date.

    Since ml_feature_history does not exist, this function uses the
    last_calculated_at timestamp in ml_feature_store to split the data
    into two temporal windows. This supports the alignment validation
    (Question 4) which needs two separate feature matrices.

    Args:
        tenant_id: The tenant to load data for.
        before_date: ISO date string (YYYY-MM-DD). Customers calculated
            before this date go into the 'before' set.
        after_date: ISO date string (YYYY-MM-DD). Customers calculated
            on or after this date go into the 'after' set.
            If None, defaults to same as before_date.

    Returns:
        Tuple of (before_df, after_df) - each a customerxfeature DataFrame.
        Either can be empty if no data falls in that window.
    """
    if after_date is None:
        after_date = before_date

    logger.info(
        "Loading feature snapshots split at date '%s' for tenant '%s' ...",
        before_date, tenant_id,
    )

    selected = get_active_features_for_tenant(tenant_id)
    sql_columns = [f for f in selected if f in FEATURE_STORE_COLUMNS]
    json_features = [f for f in selected if f not in FEATURE_STORE_COLUMNS]

    cols_clause = ", ".join([f"`{c}`" for c in sql_columns])
    extra_cols = ", `custom_features`" if json_features else ""

    conn = get_db_connection()

    # Before-window
    cursor_before = conn.cursor(dictionary=True)
    cursor_before.execute(
        f"SELECT `customer_id`, {cols_clause}{extra_cols} "
        f"FROM `ml_feature_store` "
        f"WHERE `tenant_id` = %s AND `last_calculated_at` < %s",
        (tenant_id, before_date),
    )
    rows_before = cursor_before.fetchall()
    cursor_before.close()

    # After-window
    cursor_after = conn.cursor(dictionary=True)
    cursor_after.execute(
        f"SELECT `customer_id`, {cols_clause}{extra_cols} "
        f"FROM `ml_feature_store` "
        f"WHERE `tenant_id` = %s AND `last_calculated_at` >= %s",
        (tenant_id, after_date),
    )
    rows_after = cursor_after.fetchall()
    cursor_after.close()
    conn.close()

    before_df = _rows_to_feature_df(rows_before, selected, json_features)
    after_df = _rows_to_feature_df(rows_after, selected, json_features)

    logger.info(
        "  Before '%s': %d customers | After: %d customers",
        before_date, len(before_df), len(after_df),
    )
    return before_df, after_df


# ===================================================================
# Private helpers
# ===================================================================

def _extract_json_feature(json_val, feature_name: str):
    """
    Extract a feature value from the custom_features JSON column.

    Args:
        json_val: The raw value from the custom_features column
            (may be str, dict, or None).
        feature_name: The feature key to extract.

    Returns:
        The feature value as a float, or np.nan if not found.
    """
    if json_val is None:
        return np.nan

    if isinstance(json_val, str):
        try:
            data = json.loads(json_val)
        except (json.JSONDecodeError, TypeError):
            return np.nan
    elif isinstance(json_val, dict):
        data = json_val
    else:
        return np.nan

    val = data.get(feature_name, np.nan)
    if val is None:
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _rows_to_feature_df(
    rows: list,
    selected_features: List[str],
    json_features: List[str],
) -> pd.DataFrame:
    """
    Convert raw DB rows to a clean customer x feature DataFrame.

    Handles JSON extraction, null filling, and non-negativity.
    """
    if not rows:
        empty = pd.DataFrame(columns=selected_features)
        empty.index.name = "customer_id"
        return empty

    df = pd.DataFrame(rows)
    df.set_index("customer_id", inplace=True)

    # Extract JSON features
    if json_features and "custom_features" in df.columns:
        for feat in json_features:
            df[feat] = df["custom_features"].apply(
                lambda x: _extract_json_feature(x, feat)
            )
    if "custom_features" in df.columns:
        df.drop(columns=["custom_features"], inplace=True, errors="ignore")

    available = [f for f in selected_features if f in df.columns]
    df = df[available].astype(float)
    df.fillna(0.0, inplace=True)

    # Enforce non-negativity
    for col in df.columns:
        neg_mask = df[col] < 0
        if neg_mask.any():
            df[col] = df[col].abs()

    # Scale features using Max-Scaling
    scaled_df = df.copy()
    for col in scaled_df.columns:
        max_val = scaled_df[col].max()
        if max_val > 0:
            scaled_df[col] = scaled_df[col] / max_val
        else:
            scaled_df[col] = 0.0
    df = scaled_df

    return df


def use_population_baseline_model(
    feature_df: pd.DataFrame,
    min_customers_threshold: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Challenge #1: Data Sufficiency (Small Tenant Problem).

    When a tenant has fewer customers than min_customers_threshold (< 5), NMF
    cannot discover stable behavioral clusters. This fallback uses population-level
    mean feature weights to produce a baseline scoring model until the tenant
    accumulates sufficient customer history.

    Args:
        feature_df: Customer x feature DataFrame.
        min_customers_threshold: Minimum customer count threshold (default 5).

    Returns:
        Tuple of:
          - H_baseline (1 x features): Single population baseline pattern.
          - W_baseline (customers x 1): Equal baseline blend weights (1.0).
          - pattern_labels: ['Population Baseline Pattern']
    """
    logger.info(
        "  [FALLBACK] Tenant customer count (%d) < threshold (%d). "
        "Using population-based baseline model.",
        len(feature_df), min_customers_threshold,
    )
    feature_names = feature_df.columns.tolist()
    customer_ids = feature_df.index.tolist()

    # Single pattern = column means
    mean_vec = feature_df.mean(axis=0).values.reshape(1, -1)
    H_baseline = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)

    W_baseline = np.ones((len(customer_ids), 1))
    pattern_labels = ["Population Baseline Pattern"]

    return H_baseline, W_baseline, pattern_labels

