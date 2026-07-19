"""
run_prototype.py - Single entry point for the NMF churn pattern prototype.

Usage:
    python run_prototype.py --tenant sciqusams
    python run_prototype.py --tenant sciqusams --split-date 2026-06-01
    python run_prototype.py --tenant all
    python run_prototype.py  (interactive tenant selection)

This script orchestrates the full validation pipeline:
  1. Print available tenants and select one (or all)
  2. Load feature data and print quality report
  3. Select optimal K via elbow detection
  4. Run NMF and discover patterns
  5. Label patterns automatically
  6. Compute blend vectors and drift scores
  7. Run all four validations
  8. Print full validation summary
  9. Save results to output/
  10. Print final plain-English summary
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional

import numpy as np

from config import validate_db_connection, logger
from data_loader import (
    get_active_tenants,
    get_active_features_for_tenant,
    load_feature_matrix,
    load_feedback_data,
    load_feature_registry,
    load_feature_snapshots_by_date,
)
from pattern_engine import (
    select_optimal_k,
    run_nmf,
    auto_label_patterns,
)
from drift_detector import (
    compute_blends,
    compute_drift_scores,
)
from validator import (
    validate_pattern_quality,
    validate_k_stability,
    validate_against_outcomes,
    validate_pattern_alignment,
    validate_pattern_alignment_temporal,
)
from results import (
    print_pattern_summary,
    print_customer_risk_table,
    print_pattern_lineage_table,
    print_validation_summary,
    print_cross_tenant_summary,
    save_results_to_files,
)


def run_single_tenant(
    tenant_id: str,
    split_date: Optional[str] = None,
) -> Dict:
    """
    Execute the full prototype pipeline for a single tenant.

    Args:
        tenant_id: The tenant to analyse.
        split_date: Optional ISO date for temporal alignment validation.

    Returns:
        Dict containing all validation reports for this tenant.
    """
    print("\n" + "#" * 80)
    print(f"  RUNNING NMF PROTOTYPE FOR TENANT: {tenant_id}")
    print("#" * 80)

    # -- Step 1: Load data --------------------------------------------------
    logger.info("Step 1/10: Loading feature data ...")
    feature_df, quality_report = load_feature_matrix(tenant_id)

    if feature_df.empty or len(feature_df) < 3:
        logger.error(
            "Cannot proceed: only %d customers found for tenant '%s'. "
            "Need at least 3 customers for NMF.",
            len(feature_df), tenant_id,
        )
        return {"error": f"Insufficient data ({len(feature_df)} customers)"}

    print(f"\n  Data Quality Report for '{tenant_id}':")
    print(f"    Total customers: {quality_report['total_customers']}")
    print(f"    Features available: {quality_report['features_available']}")
    if quality_report.get("features_missing"):
        print(f"    Features MISSING: {quality_report['features_missing']}")
    if quality_report.get("features_from_json"):
        print(f"    Features from JSON: {quality_report['features_from_json']}")
    if quality_report.get("negative_features_corrected"):
        print(f"    Negative features corrected: {quality_report['negative_features_corrected']}")

    # Null rate summary
    null_rates = quality_report.get("null_rates", {})
    high_null_features = {k: v for k, v in null_rates.items() if v > 0.1}
    if high_null_features:
        print(f"    High null-rate features: {high_null_features}")

    feature_matrix = feature_df.values
    feature_names = feature_df.columns.tolist()
    customer_ids = feature_df.index.tolist()

    # -- Step 2: Load feedback ----------------------------------------------
    logger.info("Step 2/10: Loading feedback data ...")
    feedback_df = load_feedback_data(tenant_id)

    # -- Step 3: Load feature registry (for category metadata) -------------
    logger.info("Step 3/10: Loading feature registry ...")
    registry_df = load_feature_registry()
    feature_categories = {}
    if not registry_df.empty:
        feature_categories = dict(
            zip(registry_df["feature_name"], registry_df["feature_category"])
        )

    # -- Step 4: Select optimal K ------------------------------------------
    logger.info("Step 4/10: Selecting optimal K ...")
    optimal_k, k_selection_data, k_stability = select_optimal_k(feature_matrix)

    # -- Step 5: Run NMF ---------------------------------------------------
    logger.info("Step 5/10: Running NMF with K=%d ...", optimal_k)
    H_matrix, W_matrix, recon_error, converged = run_nmf(feature_matrix, optimal_k)

    # -- Step 6: Label patterns --------------------------------------------
    logger.info("Step 6/10: Auto-labeling patterns ...")
    pattern_labels, feature_importance = auto_label_patterns(
        H_matrix, feature_names, feature_categories=feature_categories,
    )

    # -- Step 7: Compute blends and drift ----------------------------------
    logger.info("Step 7/10: Computing blends and drift scores ...")
    blends_df = compute_blends(W_matrix, customer_ids, pattern_labels)
    # First run - no previous blends, so drift will be zero
    drift_df = compute_drift_scores(blends_df, previous_blends=None)

    # Print intermediate results
    print_pattern_summary(H_matrix, feature_names, pattern_labels, blends_df)
    print_customer_risk_table(blends_df, drift_df, feedback_df=feedback_df)

    # -- Step 8: Validate --------------------------------------------------
    logger.info("Step 8/10: Running all four validations ...")

    # Q1: Pattern quality
    quality_report_v = validate_pattern_quality(
        H_matrix, feature_names, feature_matrix, W_matrix, recon_error,
    )

    # Q2: K stability
    stability_report = validate_k_stability(feature_matrix, optimal_k)

    # Q3: Outcome validation
    outcome_report = validate_against_outcomes(blends_df, drift_df, feedback_df)

    # Q4: Pattern alignment
    if split_date:
        logger.info("Using temporal split at %s for alignment validation ...", split_date)
        before_df, after_df = load_feature_snapshots_by_date(
            tenant_id, before_date=split_date,
        )
        if not before_df.empty and not after_df.empty:
            alignment_report = validate_pattern_alignment_temporal(
                before_df.values, after_df.values, optimal_k,
                feature_names=feature_names, feature_categories=feature_categories
            )
        else:
            logger.warning(
                "Temporal split produced empty matrices. "
                "Falling back to random split."
            )
            alignment_report = validate_pattern_alignment(
                feature_matrix, optimal_k,
                feature_names=feature_names, feature_categories=feature_categories
            )
    else:
        alignment_report = validate_pattern_alignment(
            feature_matrix, optimal_k,
            feature_names=feature_names, feature_categories=feature_categories
        )

    # Print Hungarian pattern lineage table
    print_pattern_lineage_table(alignment_report)

    # -- Step 9: Print validation summary ----------------------------------
    logger.info("Step 9/10: Printing validation summary ...")
    print_validation_summary(
        quality_report_v, stability_report, outcome_report, alignment_report,
    )

    # -- Step 10: Save results ---------------------------------------------
    logger.info("Step 10/10: Saving results to output/ ...")
    all_reports = {
        "quality": quality_report_v,
        "stability": stability_report,
        "outcome": outcome_report,
        "alignment": alignment_report,
    }

    save_results_to_files(
        tenant_id=tenant_id,
        pattern_labels=pattern_labels,
        H_matrix=H_matrix,
        feature_names=feature_names,
        blends_df=blends_df,
        drift_df=drift_df,
        all_reports=all_reports,
        data_quality_report=quality_report,
        optimal_k=optimal_k,
    )

    # -- Final summary paragraph -------------------------------------------
    _print_final_summary(tenant_id, optimal_k, pattern_labels, quality_report_v,
                         stability_report, outcome_report, alignment_report)

    return all_reports


def run_all_tenants(
    split_date: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Execute the prototype for all active tenants and produce a cross-tenant summary.

    Args:
        split_date: Optional ISO date for temporal alignment validation.

    Returns:
        Dict of {tenant_id: all_reports} for every tenant.
    """
    tenants = get_active_tenants()
    if not tenants:
        logger.error("No active tenants found. Exiting.")
        sys.exit(1)

    print(f"\n  Multi-tenant mode: running across {len(tenants)} tenants")
    print(f"  Tenants: {tenants}\n")

    all_tenant_reports = {}

    for tenant_id in tenants:
        try:
            reports = run_single_tenant(tenant_id, split_date=split_date)
            all_tenant_reports[tenant_id] = reports
        except Exception as exc:
            logger.error(
                "Failed to process tenant '%s': %s", tenant_id, exc,
                exc_info=True,
            )
            all_tenant_reports[tenant_id] = {"error": str(exc)}

    # Cross-tenant summary
    print_cross_tenant_summary(all_tenant_reports)

    return all_tenant_reports


def _print_final_summary(
    tenant_id: str,
    optimal_k: int,
    pattern_labels: List[str],
    quality_report: Dict,
    stability_report: Dict,
    outcome_report: Dict,
    alignment_report: Dict,
) -> None:
    """
    Print a final one-paragraph plain-English summary of findings.
    """
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    q1 = quality_report.get("verdict", "N/A")
    q2 = stability_report.get("verdict", "N/A")
    q3 = outcome_report.get("verdict", "N/A")
    q4 = alignment_report.get("verdict", "N/A")

    verdicts = [q1, q2, q3, q4]
    pass_count = sum(1 for v in verdicts if v == "PASS")
    marginal_count = sum(1 for v in verdicts if v == "MARGINAL")

    print(f"""
  NMF Behavioral Pattern Decomposition was run on tenant '{tenant_id}'
  with {optimal_k} patterns discovered: {', '.join(pattern_labels)}.

  Validation results:
    Q1 Pattern Quality:    {q1}
    Q2 K Stability:        {q2}
    Q3 Outcome Prediction: {q3}
    Q4 Pattern Alignment:  {q4}

  Overall: {pass_count} PASS, {marginal_count} MARGINAL, \
{sum(1 for v in verdicts if v == 'FAIL')} FAIL, \
{sum(1 for v in verdicts if v in ('INSUFFICIENT_DATA', 'N/A'))} INSUFFICIENT DATA.

  Results saved to output/ directory.
""")
    print("=" * 80)


def main():
    """
    Main entry point. Parse arguments and run the prototype.
    """
    parser = argparse.ArgumentParser(
        description="NMF Behavioral Pattern Decomposition - Churn Risk Prototype",
    )
    parser.add_argument(
        "--tenant",
        type=str,
        default=None,
        help="Tenant ID to analyse (e.g., 'sciqusams'). "
             "Use 'all' to run across all active tenants. "
             "If not provided, interactive selection.",
    )
    parser.add_argument(
        "--split-date",
        type=str,
        default=None,
        help="ISO date (YYYY-MM-DD) for temporal alignment validation "
             "(Question 4). If not provided, uses random split.",
    )
    args = parser.parse_args()

    # Banner
    print("\n" + "=" * 80)
    print("  NMF BEHAVIORAL PATTERN DECOMPOSITION - CHURN RISK PROTOTYPE")
    print("  Scientific Validation - Not a Production System")
    print("=" * 80)

    # Test DB connection
    if not validate_db_connection():
        sys.exit(1)

    # Determine tenant
    tenant_id = args.tenant

    if tenant_id is None:
        # Interactive mode
        tenants = get_active_tenants()
        if not tenants:
            logger.error("No active tenants found. Exiting.")
            sys.exit(1)

        print("\n  Available tenants:")
        for i, t in enumerate(tenants, 1):
            print(f"    {i}. {t}")
        print(f"    {len(tenants) + 1}. ALL (run across all tenants)")

        try:
            choice = input("\n  Select tenant number: ").strip()
            idx = int(choice) - 1
            if idx == len(tenants):
                tenant_id = "all"
            elif 0 <= idx < len(tenants):
                tenant_id = tenants[idx]
            else:
                print("  Invalid selection. Exiting.")
                sys.exit(1)
        except (ValueError, EOFError):
            print("  Invalid input. Exiting.")
            sys.exit(1)

    # Run
    if tenant_id.lower() == "all":
        run_all_tenants(split_date=args.split_date)
    else:
        run_single_tenant(tenant_id, split_date=args.split_date)


if __name__ == "__main__":
    main()
