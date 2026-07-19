"""
validator.py - Validation logic answering the four prototype questions.

This is the most important file in the prototype. It determines whether
NMF-based Behavioral Pattern Decomposition is viable on real tenant data.

The four questions:
  Q1: Does NMF discover meaningful, interpretable behavioral patterns?
  Q2: Does the K selection logic produce a stable, sensible number of patterns?
  Q3: Do customer blend shifts predict confirmed churn events?
  Q4: Do patterns hold consistent across simulated retraining windows?

All thresholds are named function parameters with docstrings explaining
their origin and what constitutes a PASS result.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

from pattern_engine import run_nmf, align_patterns, auto_label_patterns

logger = logging.getLogger("nmf_prototype")


def validate_pattern_quality(
    H_matrix: np.ndarray,
    feature_names: List[str],
    feature_matrix: np.ndarray,
    W_matrix: np.ndarray,
    reconstruction_error: float,
    min_inter_pattern_distance: float = 0.3,
    min_feature_concentration: float = 0.4,
) -> Dict:
    """
    Question 1: Are the discovered patterns meaningful?

    Evaluates three criteria:
      1. Reconstruction quality - what fraction of the original data
         variance does the NMF model explain?
      2. Inter-pattern distinctiveness - are patterns sufficiently different
         from each other? Measured by average pairwise cosine distance.
      3. Feature concentration - does each pattern have clear dominant
         features, or are loadings spread evenly (no clear meaning)?

    Args:
        H_matrix: Pattern matrix (k x features).
        feature_names: Feature column names.
        feature_matrix: Original data matrix (customers x features).
        W_matrix: Customer-pattern weight matrix (customers x k).
        reconstruction_error: Error from NMF run.
        min_inter_pattern_distance: Minimum acceptable average cosine
            distance between pattern vectors. Below this, patterns are
            too similar to be useful. Default 0.3 based on empirical
            observation that distance < 0.3 means patterns overlap
            more than 70% and are not distinct.
        min_feature_concentration: Minimum acceptable concentration ratio
            (max_loading / sum_loading) for a pattern to have clear meaning.
            Default 0.4 - at least 40% of a pattern's total loading should
            come from its single strongest feature. A uniform distribution
            across N features would give 1/N ~ 0.14 for 7 features.

    Returns:
        Dict with keys: verdict ("PASS", "MARGINAL", "FAIL"),
        reconstruction_explained, inter_pattern_distance,
        concentration_scores, pattern_details (list of per-pattern reports).

    PASS: All three checks pass their thresholds.
    MARGINAL: One check fails but others pass.
    FAIL: Two or more checks fail.
    """
    logger.info("=" * 60)
    logger.info("VALIDATION Q1: Pattern Quality")
    logger.info("=" * 60)

    n_patterns = H_matrix.shape[0]
    checks_passed = 0
    total_checks = 3
    report = {"pattern_details": []}

    # -- Check 1: Reconstruction quality -----------------------------------
    original_variance = np.sum(feature_matrix ** 2)
    if original_variance > 0:
        reconstruction_explained = 1.0 - (reconstruction_error ** 2) / original_variance
    else:
        reconstruction_explained = 0.0
    reconstruction_explained = max(0.0, reconstruction_explained)

    report["reconstruction_explained"] = float(reconstruction_explained)
    recon_pass = reconstruction_explained > 0.5
    if recon_pass:
        checks_passed += 1
    logger.info(
        "  Reconstruction: %.1f%% variance explained %s",
        reconstruction_explained * 100,
        "[OK]" if recon_pass else "[X]",
    )

    # -- Check 2: Inter-pattern distinctiveness ----------------------------
    if n_patterns >= 2:
        sim_matrix = cosine_similarity(H_matrix)
        # Average off-diagonal distance
        distances = []
        for i in range(n_patterns):
            for j in range(i + 1, n_patterns):
                distances.append(1.0 - sim_matrix[i, j])
        avg_distance = float(np.mean(distances))
    else:
        avg_distance = 1.0  # Single pattern is maximally distinct

    report["inter_pattern_distance"] = avg_distance
    distance_pass = avg_distance >= min_inter_pattern_distance
    if distance_pass:
        checks_passed += 1
    logger.info(
        "  Inter-pattern distance: %.4f (threshold: %.2f) %s",
        avg_distance, min_inter_pattern_distance,
        "[OK]" if distance_pass else "[X]",
    )

    # -- Check 3: Feature concentration ------------------------------------
    concentration_scores = {}
    concentration_pass_count = 0

    for p_idx in range(n_patterns):
        pattern_vec = H_matrix[p_idx]
        total_loading = pattern_vec.sum()
        if total_loading > 0:
            max_loading = pattern_vec.max()
            concentration = max_loading / total_loading
        else:
            concentration = 0.0

        top_idx = np.argmax(pattern_vec)
        top_feature = feature_names[top_idx] if top_idx < len(feature_names) else "unknown"

        concentration_scores[p_idx] = float(concentration)

        if concentration >= min_feature_concentration:
            concentration_pass_count += 1

        # Detailed per-pattern report
        top_3_indices = np.argsort(pattern_vec)[::-1][:3]
        top_features_detail = [
            {"feature": feature_names[i], "loading": float(pattern_vec[i])}
            for i in top_3_indices if i < len(feature_names)
        ]

        meaningful = concentration >= min_feature_concentration
        report["pattern_details"].append({
            "pattern_index": p_idx,
            "top_feature": top_feature,
            "concentration": float(concentration),
            "meaningful": meaningful,
            "top_features": top_features_detail,
        })

        logger.info(
            "  Pattern %d: concentration=%.4f, top feature='%s' %s",
            p_idx, concentration, top_feature,
            "[OK]" if meaningful else "[X]",
        )

    report["concentration_scores"] = concentration_scores

    # Concentration passes if majority of patterns are concentrated
    overall_concentration_pass = concentration_pass_count > n_patterns / 2
    if overall_concentration_pass:
        checks_passed += 1

    # -- Final verdict -----------------------------------------------------
    if checks_passed == total_checks:
        verdict = "PASS"
    elif checks_passed >= total_checks - 1:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    report["verdict"] = verdict
    report["checks_passed"] = checks_passed
    report["total_checks"] = total_checks

    logger.info("")
    logger.info("  Q1 Verdict: %s (%d/%d checks passed)", verdict, checks_passed, total_checks)
    return report


def validate_k_stability(
    feature_matrix: np.ndarray,
    optimal_k: int,
    n_validation_runs: int = 10,
    pass_threshold: float = 0.8,
    marginal_threshold: float = 0.6,
    max_iter: int = 500,
) -> Dict:
    """
    Question 2: Is K stable?

    Runs NMF n_validation_runs times at the selected K and measures
    how consistent the discovered patterns are across runs.

    Uses the Hungarian algorithm to align H matrices from different runs,
    then computes cosine similarity of matched patterns.

    Args:
        feature_matrix: Original data matrix (customers x features).
        optimal_k: The selected number of patterns.
        n_validation_runs: Number of independent runs. Default 10.
        pass_threshold: Mean pairwise similarity must exceed this for PASS.
            Default 0.8 - patterns should be at least 80% similar across
            runs to be considered stable. Standard in NMF stability analysis
            (Brunet et al. 2004, Kim & Park 2007).
        marginal_threshold: Below pass but above this is MARGINAL.
            Default 0.6 - patterns are somewhat reproducible but may shift.
        max_iter: NMF max iterations. Default 500.

    Returns:
        Dict with: verdict, mean_similarity, min_similarity,
        max_similarity, all_similarities.

    PASS: mean_similarity > pass_threshold (0.8).
    MARGINAL: marginal_threshold < mean_similarity <= pass_threshold.
    FAIL: mean_similarity <= marginal_threshold.
    """
    logger.info("=" * 60)
    logger.info("VALIDATION Q2: K Stability (K=%d, %d runs)", optimal_k, n_validation_runs)
    logger.info("=" * 60)

    h_matrices = []
    for run_idx in range(n_validation_runs):
        model = NMF(
            n_components=optimal_k,
            init="nndsvda",
            random_state=run_idx * 37 + 7,
            max_iter=max_iter,
        )
        model.fit_transform(feature_matrix)
        h_matrices.append(model.components_)

    # Compute pairwise aligned cosine similarities
    all_sims = []
    for i in range(len(h_matrices)):
        for j in range(i + 1, len(h_matrices)):
            sim = _aligned_similarity(h_matrices[i], h_matrices[j])
            all_sims.append(sim)

    mean_sim = float(np.mean(all_sims)) if all_sims else 0.0
    min_sim = float(np.min(all_sims)) if all_sims else 0.0
    max_sim = float(np.max(all_sims)) if all_sims else 0.0

    if mean_sim > pass_threshold:
        verdict = "PASS"
    elif mean_sim > marginal_threshold:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    logger.info("  Mean similarity: %.4f", mean_sim)
    logger.info("  Min similarity: %.4f | Max: %.4f", min_sim, max_sim)
    logger.info("  Q2 Verdict: %s", verdict)

    return {
        "verdict": verdict,
        "mean_similarity": mean_sim,
        "min_similarity": min_sim,
        "max_similarity": max_sim,
        "n_runs": n_validation_runs,
        "optimal_k": optimal_k,
        "all_similarities": [float(s) for s in all_sims],
    }


def validate_against_outcomes(
    current_blends: pd.DataFrame,
    drift_scores: pd.DataFrame,
    feedback_df: pd.DataFrame,
    min_feedback_for_validation: int = 5,
) -> Dict:
    """
    Question 3: Do drift scores predict real churn?

    Joins drift scores with the feedback table and compares:
      - Mean drift of confirmed churned vs confirmed retained customers
      - Precision: what % of HIGH-risk flagged customers actually churned
      - Recall: what % of churned customers were flagged HIGH or MEDIUM

    Args:
        current_blends: Blend DataFrame (for dominant pattern info).
        drift_scores: DataFrame from compute_drift_scores().
        feedback_df: DataFrame from load_feedback_data().
        min_feedback_for_validation: Minimum number of confirmed outcomes
            needed for the validation to be meaningful. Default 5.
            Below this, results are flagged as insufficient_data.

    Returns:
        Dict with: verdict, precision, recall, mean_drift_churned,
        mean_drift_retained, sample_size, sufficient_data.

    PASS: precision > 0.5 AND recall > 0.3 AND mean_drift_churned > mean_drift_retained.
    MARGINAL: Some positive signal but weak numbers.
    FAIL: No predictive signal, or insufficient data.
    """
    logger.info("=" * 60)
    logger.info("VALIDATION Q3: Outcome Validation")
    logger.info("=" * 60)

    if feedback_df.empty or len(feedback_df) < min_feedback_for_validation:
        logger.warning(
            "  [!] Insufficient feedback data (%d rows, minimum: %d). "
            "Cannot validate against outcomes.",
            len(feedback_df), min_feedback_for_validation,
        )
        return {
            "verdict": "INSUFFICIENT_DATA",
            "precision": None,
            "recall": None,
            "mean_drift_churned": None,
            "mean_drift_retained": None,
            "sample_size": len(feedback_df),
            "sufficient_data": False,
        }

    # Join drift scores with feedback
    if "customer_id" in feedback_df.columns:
        feedback_cleaned = feedback_df.drop_duplicates(subset=["customer_id"], keep="first")
        feedback_indexed = feedback_cleaned.set_index("customer_id")
    else:
        feedback_cleaned = feedback_df.reset_index().drop_duplicates(subset=["customer_id"], keep="first")
        feedback_indexed = feedback_cleaned.set_index("customer_id")

    # Find customers in both datasets
    common_ids = drift_scores.index.intersection(feedback_indexed.index)
    logger.info("  Customers with both drift scores and feedback: %d", len(common_ids))

    if len(common_ids) < min_feedback_for_validation:
        logger.warning(
            "  [!] Only %d matching customers (minimum: %d). "
            "Insufficient for validation.",
            len(common_ids), min_feedback_for_validation,
        )
        return {
            "verdict": "INSUFFICIENT_DATA",
            "precision": None,
            "recall": None,
            "mean_drift_churned": None,
            "mean_drift_retained": None,
            "sample_size": len(common_ids),
            "sufficient_data": False,
        }

    joined = drift_scores.loc[common_ids].copy()
    joined["actual_churned"] = feedback_indexed.loc[common_ids, "actual_churned"].astype(int)

    churned = joined[joined["actual_churned"] == 1]
    retained = joined[joined["actual_churned"] == 0]

    mean_drift_churned = float(churned["drift_score"].mean()) if len(churned) > 0 else 0.0
    mean_drift_retained = float(retained["drift_score"].mean()) if len(retained) > 0 else 0.0

    logger.info("  Mean drift (churned): %.4f (%d customers)", mean_drift_churned, len(churned))
    logger.info("  Mean drift (retained): %.4f (%d customers)", mean_drift_retained, len(retained))

    # Precision: of those flagged HIGH risk, how many actually churned?
    flagged_high = joined[joined["risk_level"] == "HIGH"]
    if len(flagged_high) > 0:
        precision = float((flagged_high["actual_churned"] == 1).sum() / len(flagged_high))
    else:
        precision = 0.0

    # Recall: of those who churned, how many were flagged HIGH or MEDIUM?
    if len(churned) > 0:
        flagged_churned = churned[churned["risk_level"].isin(["HIGH", "MEDIUM"])]
        recall = float(len(flagged_churned) / len(churned))
    else:
        recall = 0.0

    logger.info("  Precision (HIGH risk -> actual churn): %.4f", precision)
    logger.info("  Recall (churned -> flagged HIGH/MEDIUM): %.4f", recall)

    # Verdict
    drift_signal = mean_drift_churned > mean_drift_retained
    if drift_signal and precision > 0.5 and recall > 0.3:
        verdict = "PASS"
    elif drift_signal or precision > 0.3 or recall > 0.2:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    logger.info("  Q3 Verdict: %s", verdict)

    return {
        "verdict": verdict,
        "precision": precision,
        "recall": recall,
        "mean_drift_churned": mean_drift_churned,
        "mean_drift_retained": mean_drift_retained,
        "sample_size": len(common_ids),
        "churned_count": len(churned),
        "retained_count": len(retained),
        "sufficient_data": True,
    }


def validate_pattern_alignment(
    feature_matrix: np.ndarray,
    optimal_k: int,
    split_method: str = "random",
    pass_threshold: float = 0.75,
    marginal_threshold: float = 0.50,
    n_runs: int = 10,
    max_iter: int = 500,
    similarity_threshold: float = 0.6,
    feature_names: Optional[List[str]] = None,
    feature_categories: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Question 4: Do patterns stay consistent across retraining windows?

    Splits the feature matrix into two halves and runs NMF independently
    on each half, then aligns the discovered patterns using the Hungarian
    algorithm.

    Since ml_feature_history does not exist in this database, we simulate
    two retraining windows by splitting customers into two random halves.

    Args:
        feature_matrix: Original data matrix (customers x features).
        optimal_k: Number of patterns to use.
        split_method: How to split the data. "random" = random 50/50 split.
            Default "random".
        pass_threshold: Alignment quality must exceed this for PASS.
            Default 0.75 - patterns should be at least 75% similar across
            windows to be considered stable across retrains.
        marginal_threshold: Below pass but above this is MARGINAL.
            Default 0.50.
        n_runs: Number of NMF runs per split. Default 10.
        max_iter: NMF max iterations. Default 500.
        similarity_threshold: Minimum similarity for a pattern match.
            Passed through to align_patterns(). Default 0.6.
        feature_names: List of feature names for auto-labeling.
        feature_categories: Metadata categories for features.

    Returns:
        Dict with: verdict, alignment_quality, alignment_map,
        similarity_scores, split_sizes, lineage (list of maps).

    PASS: alignment_quality > pass_threshold (0.75).
    MARGINAL: marginal_threshold < alignment_quality <= pass_threshold.
    FAIL: alignment_quality <= marginal_threshold.
    """
    logger.info("=" * 60)
    logger.info("VALIDATION Q4: Pattern Alignment Across Retraining Windows")
    logger.info("=" * 60)

    n_customers = feature_matrix.shape[0]

    if n_customers < 2 * optimal_k:
        logger.warning(
            "  [!] Not enough customers (%d) to split into two groups for K=%d. "
            "Skipping alignment validation.",
            n_customers, optimal_k,
        )
        return {
            "verdict": "INSUFFICIENT_DATA",
            "alignment_quality": None,
            "alignment_map": {},
            "similarity_scores": {},
            "split_sizes": (n_customers, 0),
            "lineage": [],
        }

    # Split customers into two halves
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_customers)
    split_point = n_customers // 2

    matrix_a = feature_matrix[indices[:split_point]]
    matrix_b = feature_matrix[indices[split_point:]]

    logger.info("  Split: Group A = %d customers, Group B = %d customers",
                len(matrix_a), len(matrix_b))

    # Run NMF on each half
    H_a, W_a, err_a, conv_a = run_nmf(matrix_a, optimal_k, n_runs=n_runs, max_iter=max_iter)
    H_b, W_b, err_b, conv_b = run_nmf(matrix_b, optimal_k, n_runs=n_runs, max_iter=max_iter)

    # Align patterns
    alignment_map, similarity_scores, alignment_quality = align_patterns(
        H_b, H_a, similarity_threshold=similarity_threshold,
    )

    if alignment_quality > pass_threshold:
        verdict = "PASS"
    elif alignment_quality > marginal_threshold:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    logger.info("  Alignment quality: %.4f", alignment_quality)
    logger.info("  Q4 Verdict: %s", verdict)

    new_patterns = [k for k, v in alignment_map.items() if v == "NEW_PATTERN"]
    if new_patterns:
        logger.info("  New patterns detected (no match in other group): %s", new_patterns)

    # Calculate pattern lineage if feature names are provided
    lineage = []
    if feature_names:
        labels_a, _ = auto_label_patterns(H_a, feature_names, feature_categories=feature_categories)
        labels_b, _ = auto_label_patterns(H_b, feature_names, feature_categories=feature_categories)
        for new_idx, old_idx in alignment_map.items():
            sim = similarity_scores.get(new_idx, 0.0)
            label_new = labels_b[new_idx]
            if old_idx == "NEW_PATTERN":
                label_old = "[NEW PATTERN]"
            else:
                label_old = labels_a[old_idx]
            lineage.append({
                "new_index": new_idx,
                "old_index": old_idx,
                "new_label": label_new,
                "old_label": label_old,
                "similarity": sim
            })

    return {
        "verdict": verdict,
        "alignment_quality": alignment_quality,
        "alignment_map": alignment_map,
        "similarity_scores": similarity_scores,
        "split_sizes": (len(matrix_a), len(matrix_b)),
        "new_patterns_count": len(new_patterns),
        "lineage": lineage,
    }


def validate_pattern_alignment_temporal(
    before_matrix: np.ndarray,
    after_matrix: np.ndarray,
    optimal_k: int,
    pass_threshold: float = 0.75,
    marginal_threshold: float = 0.50,
    n_runs: int = 10,
    max_iter: int = 500,
    similarity_threshold: float = 0.6,
    feature_names: Optional[List[str]] = None,
    feature_categories: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Question 4 (temporal variant): Alignment using actual time-split data.

    Used when load_feature_snapshots_by_date() can produce two non-empty
    matrices. Falls back to validate_pattern_alignment() (random split)
    if temporal data is insufficient.

    Args:
        before_matrix: Feature matrix for the earlier time window.
        after_matrix: Feature matrix for the later time window.
        optimal_k: Number of patterns.
        pass_threshold: PASS threshold for alignment quality. Default 0.75.
        marginal_threshold: MARGINAL threshold. Default 0.50.
        n_runs: NMF runs per window. Default 10.
        max_iter: NMF iterations. Default 500.
        similarity_threshold: Match threshold. Default 0.6.
        feature_names: List of feature names for auto-labeling.
        feature_categories: Metadata categories for features.

    Returns:
        Same structure as validate_pattern_alignment().
    """
    logger.info("=" * 60)
    logger.info("VALIDATION Q4: Temporal Pattern Alignment")
    logger.info("=" * 60)

    if before_matrix.shape[0] < optimal_k or after_matrix.shape[0] < optimal_k:
        logger.warning(
            "  [!] Insufficient data in one or both time windows "
            "(before: %d, after: %d, need at least K=%d each). "
            "Falling back to random split.",
            before_matrix.shape[0], after_matrix.shape[0], optimal_k,
        )
        return {
            "verdict": "INSUFFICIENT_DATA",
            "alignment_quality": None,
            "method": "temporal",
            "lineage": [],
        }

    logger.info("  Before window: %d customers | After window: %d customers",
                before_matrix.shape[0], after_matrix.shape[0])

    H_before, _, _, _ = run_nmf(before_matrix, optimal_k, n_runs=n_runs, max_iter=max_iter)
    H_after, _, _, _ = run_nmf(after_matrix, optimal_k, n_runs=n_runs, max_iter=max_iter)

    alignment_map, similarity_scores, alignment_quality = align_patterns(
        H_after, H_before, similarity_threshold=similarity_threshold,
    )

    if alignment_quality > pass_threshold:
        verdict = "PASS"
    elif alignment_quality > marginal_threshold:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    logger.info("  Alignment quality: %.4f", alignment_quality)
    logger.info("  Q4 Verdict: %s", verdict)

    # Calculate pattern lineage if feature names are provided
    lineage = []
    if feature_names:
        labels_before, _ = auto_label_patterns(H_before, feature_names, feature_categories=feature_categories)
        labels_after, _ = auto_label_patterns(H_after, feature_names, feature_categories=feature_categories)
        for new_idx, old_idx in alignment_map.items():
            sim = similarity_scores.get(new_idx, 0.0)
            label_new = labels_after[new_idx]
            if old_idx == "NEW_PATTERN":
                label_old = "[NEW PATTERN]"
            else:
                label_old = labels_before[old_idx]
            lineage.append({
                "new_index": new_idx,
                "old_index": old_idx,
                "new_label": label_new,
                "old_label": label_old,
                "similarity": sim
            })

    return {
        "verdict": verdict,
        "alignment_quality": alignment_quality,
        "alignment_map": alignment_map,
        "similarity_scores": similarity_scores,
        "method": "temporal",
        "split_sizes": (before_matrix.shape[0], after_matrix.shape[0]),
        "new_patterns_count": sum(1 for v in alignment_map.values() if v == "NEW_PATTERN"),
        "lineage": lineage,
    }


# ===================================================================
# Private helpers
# ===================================================================

def _aligned_similarity(H1: np.ndarray, H2: np.ndarray) -> float:
    """
    Compute aligned cosine similarity between two H matrices using
    the Hungarian algorithm.
    """
    sim_matrix = cosine_similarity(H1, H2)
    cost_matrix = 1.0 - sim_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    similarities = [sim_matrix[r, c] for r, c in zip(row_indices, col_indices)]
    return float(np.mean(similarities)) if similarities else 0.0
