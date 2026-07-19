"""
results.py - Save and display all prototype results clearly.

This module handles:
  - ASCII table printing for patterns, risks, and validation summaries
  - CSV/JSON file output to the output/ directory
  - Cross-tenant comparison summaries for multi-tenant runs

All output is transparent - the user sees exactly what was computed
and can inspect every intermediate result.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import PROTOTYPE_OUTPUT_DIR

logger = logging.getLogger("nmf_prototype")


def print_pattern_summary(
    H_matrix: np.ndarray,
    feature_names: List[str],
    pattern_labels: List[str],
    blends_df: pd.DataFrame,
    dominant_blend_threshold: float = 0.5,
) -> None:
    """
    Print a clear ASCII table showing discovered patterns.

    Displays:
      - Each pattern name
      - Its top 3 contributing features with loading values
      - How many customers primarily express this pattern (blend > threshold)
      - What percentage of total customers that represents

    Args:
        H_matrix: Pattern matrix (k x features).
        feature_names: Feature column names.
        pattern_labels: Pattern label strings.
        blends_df: Customer blend DataFrame.
        dominant_blend_threshold: Blend weight above which a customer is
            considered to "primarily express" a pattern. Default 0.5 (50%).
    """
    print("\n" + "=" * 80)
    print("PATTERN SUMMARY")
    print("=" * 80)

    total_customers = len(blends_df)

    for p_idx, label in enumerate(pattern_labels):
        print(f"\n  Pattern {p_idx}: {label}")
        print("  " + "-" * 50)

        # Top features
        pattern_vec = H_matrix[p_idx]
        top_indices = np.argsort(pattern_vec)[::-1][:3]
        for rank, feat_idx in enumerate(top_indices, 1):
            if feat_idx < len(feature_names):
                print(
                    f"    {rank}. {feature_names[feat_idx]:<40s} loading: {pattern_vec[feat_idx]:.4f}"
                )

        # Customer count
        if label in blends_df.columns:
            dominant_count = (blends_df[label] > dominant_blend_threshold).sum()
            pct = (dominant_count / total_customers * 100) if total_customers > 0 else 0.0
            print(f"    Customers (blend > {dominant_blend_threshold}): "
                  f"{dominant_count} ({pct:.1f}%)")

    print("\n" + "=" * 80)


def print_customer_risk_table(
    blends_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    feedback_df: Optional[pd.DataFrame] = None,
    top_n: int = 20,
) -> None:
    """
    Print the top N customers by drift score.

    Shows: customer_id, previous dominant pattern, current dominant pattern,
    drift score, risk level, and churn status (if feedback available).

    Args:
        blends_df: Current blend DataFrame.
        drift_df: Drift score DataFrame.
        feedback_df: Optional feedback DataFrame for churn labels.
        top_n: Number of top customers to display. Default 20.
    """
    print("\n" + "=" * 80)
    print(f"TOP {top_n} CUSTOMERS BY DRIFT SCORE")
    print("=" * 80)

    # Sort by drift score descending
    sorted_drift = drift_df.sort_values("drift_score", ascending=False).head(top_n)

    # Determine dominant pattern for each customer
    dominant_patterns = {}
    for cust_id in sorted_drift.index:
        if cust_id in blends_df.index:
            row = blends_df.loc[cust_id]
            dominant_patterns[cust_id] = row.idxmax()
        else:
            dominant_patterns[cust_id] = "N/A"

    # Build feedback lookup
    churn_lookup = {}
    if feedback_df is not None and not feedback_df.empty:
        if "customer_id" in feedback_df.columns:
            for _, row in feedback_df.iterrows():
                churn_lookup[row["customer_id"]] = int(row["actual_churned"])
        elif feedback_df.index.name == "customer_id":
            for cust_id in feedback_df.index:
                churn_lookup[cust_id] = int(feedback_df.loc[cust_id, "actual_churned"])

    # Print header
    print(f"\n  {'Customer ID':<30s} {'Dominant Pattern':<28s} "
          f"{'Drift':>8s} {'Risk':>8s} {'Churned':>8s}")
    print("  " + "-" * 86)

    for cust_id in sorted_drift.index:
        row = sorted_drift.loc[cust_id]
        dominant = dominant_patterns.get(cust_id, "N/A")
        # Truncate long pattern names
        if len(str(dominant)) > 26:
            dominant = str(dominant)[:23] + "..."
        drift = row["drift_score"]
        risk = row["risk_level"]
        churned = churn_lookup.get(cust_id, "-")

        # Truncate long customer IDs
        cust_display = str(cust_id)
        if len(cust_display) > 28:
            cust_display = cust_display[:25] + "..."

        print(f"  {cust_display:<30s} {str(dominant):<28s} "
              f"{drift:>8.4f} {risk:>8s} {str(churned):>8s}")

    print()


def print_pattern_lineage_table(alignment_report: Dict) -> None:
    """
    Print the pattern lineage mapping showing the Hungarian alignment.
    """
    if not alignment_report or alignment_report.get("verdict") == "INSUFFICIENT_DATA":
        return

    lineage = alignment_report.get("lineage", [])
    if not lineage:
        return

    print("\n" + "=" * 80)
    print("PATTERN LINEAGE (HUNGARIAN ALIGNMENT)")
    print("=" * 80)

    print(f"\n  {'New Pattern (Retrained)':<34s}   {'Matched Old Pattern':<34s} {'Cosine Sim':>10s}")
    print("  " + "-" * 81)

    for item in lineage:
        new_label = item["new_label"]
        old_label = item["old_label"]
        sim = item["similarity"]

        # Truncate label strings to avoid overlapping
        if len(new_label) > 32:
            new_label = new_label[:29] + "..."
        if len(old_label) > 32:
            old_label = old_label[:29] + "..."

        print(f"  {new_label:<34s} -> {old_label:<34s} {sim:>10.4f}")

    print("\n" + "=" * 80)


def print_validation_summary(
    quality_report: Dict,
    stability_report: Dict,
    outcome_report: Dict,
    alignment_report: Dict,
) -> None:
    """
    Print the final validation summary with PASS / MARGINAL / FAIL verdicts.

    Includes the key numbers that led to each verdict and ends with a
    plain-English recommendation.

    Args:
        quality_report: From validate_pattern_quality().
        stability_report: From validate_k_stability().
        outcome_report: From validate_against_outcomes().
        alignment_report: From validate_pattern_alignment().
    """
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    verdicts = []

    # Q1: Pattern Quality
    q1_verdict = quality_report.get("verdict", "N/A")
    verdicts.append(q1_verdict)
    q1_explained = quality_report.get("reconstruction_explained", 0)
    q1_distance = quality_report.get("inter_pattern_distance", 0)
    print(f"\n  Q1. Pattern Quality:        [{q1_verdict:^12s}]")
    print(f"       Variance explained: {q1_explained:.1%}")
    print(f"       Inter-pattern distance: {q1_distance:.4f}")
    print(f"       Checks passed: {quality_report.get('checks_passed', 0)}"
          f"/{quality_report.get('total_checks', 3)}")

    # Q2: K Stability
    q2_verdict = stability_report.get("verdict", "N/A")
    verdicts.append(q2_verdict)
    q2_sim = stability_report.get("mean_similarity", 0)
    print(f"\n  Q2. K Stability (K={stability_report.get('optimal_k', '?')}): "
          f"[{q2_verdict:^12s}]")
    print(f"       Mean cross-run similarity: {q2_sim:.4f}")
    print(f"       Range: [{stability_report.get('min_similarity', 0):.4f}, "
          f"{stability_report.get('max_similarity', 0):.4f}]")

    # Q3: Outcome Validation
    q3_verdict = outcome_report.get("verdict", "N/A")
    verdicts.append(q3_verdict)
    print(f"\n  Q3. Outcome Validation:     [{q3_verdict:^12s}]")
    if outcome_report.get("sufficient_data", False):
        print(f"       Precision: {outcome_report.get('precision', 0):.4f}")
        print(f"       Recall: {outcome_report.get('recall', 0):.4f}")
        print(f"       Mean drift (churned): {outcome_report.get('mean_drift_churned', 0):.4f}")
        print(f"       Mean drift (retained): {outcome_report.get('mean_drift_retained', 0):.4f}")
        print(f"       Sample size: {outcome_report.get('sample_size', 0)}")
    else:
        print(f"       [!] Insufficient feedback data (n={outcome_report.get('sample_size', 0)})")

    # Q4: Pattern Alignment
    q4_verdict = alignment_report.get("verdict", "N/A")
    verdicts.append(q4_verdict)
    q4_quality = alignment_report.get("alignment_quality")
    print(f"\n  Q4. Pattern Alignment:      [{q4_verdict:^12s}]")
    if q4_quality is not None:
        print(f"       Alignment quality: {q4_quality:.4f}")
        print(f"       New patterns emerged: {alignment_report.get('new_patterns_count', 0)}")
    else:
        print(f"       [!] Insufficient data for alignment validation")

    # Final recommendation
    print("\n" + "-" * 80)

    pass_count = sum(1 for v in verdicts if v == "PASS")
    marginal_count = sum(1 for v in verdicts if v == "MARGINAL")
    fail_count = sum(1 for v in verdicts if v == "FAIL")
    insufficient_count = sum(1 for v in verdicts if v in ("INSUFFICIENT_DATA", "N/A"))

    # Exclude insufficient data from fail assessment
    assessed = pass_count + marginal_count + fail_count

    if fail_count == 0 and pass_count >= assessed - 1 and assessed > 0:
        if pass_count == assessed:
            print("\n  [PASS] RECOMMENDATION: NMF pattern decomposition is validated on")
            print("     this tenant's data. Recommend proceeding to production design.")
        else:
            print("\n  [WARN] RECOMMENDATION: Approach is viable with noted limitation.")
            print("     Review the marginal area(s) before proceeding.")
    elif fail_count > 0:
        print("\n  [FAIL] RECOMMENDATION: Approach has a fundamental issue with this")
        print("     tenant's data. Details above. Recommend review before proceeding.")
    else:
        print("\n  [INFO]  RECOMMENDATION: Unable to fully assess - insufficient data for")
        print("     some validations. Gather more data and re-run.")

    if insufficient_count > 0:
        print(f"\n     Note: {insufficient_count} validation(s) could not be assessed "
              f"due to insufficient data.")

    print("\n" + "=" * 80)


def print_cross_tenant_summary(
    all_tenant_reports: Dict[str, Dict],
) -> None:
    """
    Print a cross-tenant comparison table for multi-tenant runs.

    Args:
        all_tenant_reports: Dict of {tenant_id: {q1: report, q2: report, ...}}.
    """
    print("\n" + "=" * 80)
    print("CROSS-TENANT SUMMARY")
    print("=" * 80)

    print(f"\n  {'Tenant':<20s} {'Q1 Quality':^14s} {'Q2 Stability':^14s} "
          f"{'Q3 Outcomes':^14s} {'Q4 Alignment':^14s}")
    print("  " + "-" * 76)

    for tenant_id, reports in all_tenant_reports.items():
        q1 = reports.get("quality", {}).get("verdict", "-")
        q2 = reports.get("stability", {}).get("verdict", "-")
        q3 = reports.get("outcome", {}).get("verdict", "-")
        q4 = reports.get("alignment", {}).get("verdict", "-")

        print(f"  {tenant_id:<20s} {q1:^14s} {q2:^14s} {q3:^14s} {q4:^14s}")

    print("\n" + "=" * 80)


def save_results_to_files(
    tenant_id: str,
    pattern_labels: List[str],
    H_matrix: np.ndarray,
    feature_names: List[str],
    blends_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    all_reports: Dict,
    data_quality_report: Dict,
    optimal_k: int,
) -> None:
    """
    Save all results to the output/ directory.

    Files created:
      - patterns_{tenant_id}.csv - H matrix with feature names as columns
      - blends_{tenant_id}.csv - customer blends
      - drift_scores_{tenant_id}.csv - customer drift scores and risk levels
      - validation_report_{tenant_id}.json - all four validation reports
      - run_metadata_{tenant_id}.json - timestamp, K, features, quality report

    Args:
        tenant_id: Tenant identifier for filename.
        pattern_labels: Pattern label strings.
        H_matrix: Pattern matrix.
        feature_names: Feature column names.
        blends_df: Customer blend DataFrame.
        drift_df: Drift score DataFrame.
        all_reports: Dict containing all four validation reports.
        data_quality_report: Quality report from load_feature_matrix().
        optimal_k: The selected K value.
    """
    os.makedirs(PROTOTYPE_OUTPUT_DIR, exist_ok=True)

    # 1. Patterns CSV
    patterns_path = os.path.join(PROTOTYPE_OUTPUT_DIR, f"patterns_{tenant_id}.csv")
    patterns_df = pd.DataFrame(H_matrix, columns=feature_names, index=pattern_labels)
    patterns_df.index.name = "pattern"
    patterns_df.to_csv(patterns_path)
    print(f"  Saved: {patterns_path}")

    # 2. Blends CSV
    blends_path = os.path.join(PROTOTYPE_OUTPUT_DIR, f"blends_{tenant_id}.csv")
    blends_df.to_csv(blends_path)
    print(f"  Saved: {blends_path}")

    # 3. Drift scores CSV
    drift_path = os.path.join(PROTOTYPE_OUTPUT_DIR, f"drift_scores_{tenant_id}.csv")
    drift_df.to_csv(drift_path)
    print(f"  Saved: {drift_path}")

    # 4. Validation report JSON
    report_path = os.path.join(PROTOTYPE_OUTPUT_DIR, f"validation_report_{tenant_id}.json")
    serialisable_reports = _make_json_serialisable(all_reports)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serialisable_reports, f, indent=2, default=str)
    print(f"  Saved: {report_path}")

    # 5. Run metadata JSON
    metadata_path = os.path.join(PROTOTYPE_OUTPUT_DIR, f"run_metadata_{tenant_id}.json")
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "tenant_id": tenant_id,
        "optimal_k": optimal_k,
        "pattern_labels": pattern_labels,
        "feature_names": feature_names,
        "data_quality_report": _make_json_serialisable(data_quality_report),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved: {metadata_path}")


# ===================================================================
# Private helpers
# ===================================================================

def _make_json_serialisable(obj):
    """
    Recursively convert numpy types and other non-serialisable objects
    to standard Python types for JSON output.
    """
    if isinstance(obj, dict):
        return {str(k): _make_json_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serialisable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj
