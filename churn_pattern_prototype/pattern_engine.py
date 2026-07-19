"""
pattern_engine.py - NMF pattern discovery, K selection, labeling, and alignment.

This module implements all Non-Negative Matrix Factorization logic:
  - Automatic K selection via elbow detection with stability scoring
  - Best-of-N NMF runs with convergence monitoring
  - Automatic pattern labeling based on dominant feature categories
  - Cross-retrain pattern alignment using the Hungarian algorithm

Dependencies: scikit-learn, numpy, scipy

All numeric thresholds are exposed as named function parameters with
docstrings explaining their origin - none are buried inline as literals.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

from config import FEATURE_CATEGORY_KEYWORDS

logger = logging.getLogger("nmf_prototype")


def select_optimal_k(
    feature_matrix: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    n_runs: int = 5,
    elbow_drop_rate_fraction: float = 0.20,
    min_stability_threshold: float = 0.7,
    max_iter: int = 500,
) -> Tuple[int, Dict, float]:
    """
    Select the optimal number of NMF components (K) using elbow detection.

    Algorithm:
      1. For each k in [k_min, k_max], run NMF n_runs times with
         different random seeds.
      2. Record the mean reconstruction error and its std across runs.
      3. Compute the error drop rate between consecutive k values.
      4. Select k where the drop rate first falls below
         elbow_drop_rate_fraction x max_drop_rate (elbow point).
      5. Verify stability at the selected k by computing the average
         cosine similarity between H matrices from different runs.

    Args:
        feature_matrix: 2D array (customers x features), non-negative.
        k_min: Minimum number of components to try. Default 2.
        k_max: Maximum number of components to try. Default 8.
            Capped at min(n_customers, n_features) - 1.
        n_runs: Number of NMF runs per k value for stability assessment.
            Default 5. More runs = more reliable stability estimate but slower.
        elbow_drop_rate_fraction: The elbow is detected where the error
            drop rate falls below this fraction of the maximum drop rate.
            Default 0.20 (20%). Derived from standard elbow-method practice
            where the "knee" is approximately where marginal improvement
            drops to 20% of the peak improvement.
        min_stability_threshold: Minimum acceptable stability score at
            the selected k. Below this, patterns are considered unreliable.
            Default 0.7 based on NMF literature recommendation for
            interpretable factorizations (Brunet et al. 2004).
        max_iter: Maximum NMF iterations per run. Default 500.

    Returns:
        Tuple of:
          - optimal_k (int): The selected number of patterns.
          - elbow_plot_data (dict): k->{mean_error, std_error, drop_rate,
            stability} for plotting.
          - stability_score (float): Cosine similarity at optimal_k.

    PASS result: stability_score >= min_stability_threshold.
    """
    logger.info("=" * 60)
    logger.info("K SELECTION: Testing k = %d to %d (%d runs each)", k_min, k_max, n_runs)
    logger.info("=" * 60)

    n_customers, n_features = feature_matrix.shape
    # K cannot exceed the smaller matrix dimension
    effective_k_max = min(k_max, n_customers - 1, n_features - 1)
    if effective_k_max < k_min:
        logger.warning(
            "  Matrix too small (%dx%d) for k_min=%d. Forcing k=%d.",
            n_customers, n_features, k_min, k_min,
        )
        return k_min, {}, 0.0

    results = {}

    for k in range(k_min, effective_k_max + 1):
        errors = []
        h_matrices = []

        for run_idx in range(n_runs):
            model = NMF(
                n_components=k,
                init="nndsvda",
                random_state=run_idx * 42,
                max_iter=max_iter,
            )
            W = model.fit_transform(feature_matrix)
            H = model.components_
            errors.append(model.reconstruction_err_)
            h_matrices.append(H)

        # Stability: average pairwise cosine similarity between H matrices
        stability = _compute_h_matrix_stability(h_matrices)

        results[k] = {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "stability": stability,
            "drop_rate": 0.0,  # Filled below
        }

    # Compute drop rates between consecutive k values
    k_values = sorted(results.keys())
    for i in range(1, len(k_values)):
        prev_err = results[k_values[i - 1]]["mean_error"]
        curr_err = results[k_values[i]]["mean_error"]
        if prev_err > 0:
            results[k_values[i]]["drop_rate"] = (prev_err - curr_err) / prev_err
        else:
            results[k_values[i]]["drop_rate"] = 0.0

    # Find maximum drop rate
    drop_rates = [results[k]["drop_rate"] for k in k_values[1:]]
    max_drop = max(drop_rates) if drop_rates else 0.0

    # Print the selection table
    logger.info("")
    logger.info("  %-4s  %-12s  %-10s  %-10s  %-10s", "K", "Mean Error", "Std Error", "Drop Rate", "Stability")
    logger.info("  %s", "-" * 52)
    for k in k_values:
        r = results[k]
        logger.info(
            "  %-4d  %-12.4f  %-10.4f  %-10.4f  %-10.4f",
            k, r["mean_error"], r["std_error"], r["drop_rate"], r["stability"],
        )

    # Elbow detection: find k where drop rate falls below threshold
    optimal_k = k_values[-1]  # Default to max if no elbow found
    elbow_threshold = elbow_drop_rate_fraction * max_drop

    for i in range(1, len(k_values)):
        k = k_values[i]
        if results[k]["drop_rate"] < elbow_threshold:
            # The elbow is at the PREVIOUS k (last one before diminishing returns)
            optimal_k = k_values[i - 1]
            break

    # If the first k already has low drop, use k_min
    if optimal_k < k_min:
        optimal_k = k_min

    stability_score = results[optimal_k]["stability"]

    logger.info("")
    logger.info("  -> Selected optimal K = %d", optimal_k)
    logger.info("    Stability score: %.4f", stability_score)
    if stability_score < min_stability_threshold:
        logger.warning(
            "  [!] Stability (%.4f) is below threshold (%.2f). "
            "Patterns may not be reliable.",
            stability_score, min_stability_threshold,
        )
    else:
        logger.info("  [OK] Stability is above threshold (%.2f)", min_stability_threshold)

    return optimal_k, results, stability_score


def run_nmf(
    feature_matrix: np.ndarray,
    k: int,
    n_runs: int = 10,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Run NMF n_runs times and keep the best run (lowest reconstruction error).

    Args:
        feature_matrix: 2D array (customers x features), non-negative.
        k: Number of components (patterns) to discover.
        n_runs: Number of independent runs with different random seeds.
            Default 10. The run with the lowest reconstruction error is kept.
        max_iter: Maximum NMF iterations per run. Default 500 to ensure
            convergence. scikit-learn default is 200 which often does not
            suffice for complex feature matrices.

    Returns:
        Tuple of:
          - H matrix (k x features): Pattern definitions.
          - W matrix (customers x k): Customer-pattern weights.
          - reconstruction_error (float): Error of the best run.
          - converged (bool): Whether the best run converged within max_iter.
    """
    logger.info("Running NMF with K=%d (%d runs, max_iter=%d) ...", k, n_runs, max_iter)

    best_error = float("inf")
    best_H = None
    best_W = None
    best_converged = False

    for run_idx in range(n_runs):
        model = NMF(
            n_components=k,
            init="nndsvda",
            random_state=run_idx * 42,
            max_iter=max_iter,
        )
        W = model.fit_transform(feature_matrix)
        H = model.components_
        error = model.reconstruction_err_
        converged = model.n_iter_ < max_iter

        if error < best_error:
            best_error = error
            best_H = H
            best_W = W
            best_converged = converged

    logger.info("  Best reconstruction error: %.6f", best_error)
    if not best_converged:
        logger.warning(
            "  [!] NMF did NOT converge within %d iterations. "
            "Consider increasing max_iter or reviewing data quality.",
            max_iter,
        )
    else:
        logger.info("  [OK] NMF converged successfully")

    return best_H, best_W, best_error, best_converged


def auto_label_patterns(
    H_matrix: np.ndarray,
    feature_names: List[str],
    feature_categories: Optional[Dict[str, str]] = None,
    top_n: int = 3,
) -> Tuple[List[str], Dict[int, Dict[str, float]]]:
    """
    Generate plain-English labels for each discovered pattern.

    For each row in H (each pattern):
      1. Find the top_n features with the highest loading values.
      2. Determine the dominant feature category using either:
         - The feature_categories dict (from feature_registry), or
         - Keyword matching via FEATURE_CATEGORY_KEYWORDS in config.py.
      3. Generate a label like "Engagement-Dominant Pattern" or
         "Revenue & Support Pattern".

    Args:
        H_matrix: Pattern matrix (k x features).
        feature_names: Column names matching H columns.
        feature_categories: Optional dict {feature_name: category_string}.
            If provided, takes priority over keyword matching.
        top_n: Number of top features to consider per pattern. Default 3.

    Returns:
        Tuple of:
          - labels: List of label strings, one per pattern.
          - feature_importance: Dict {pattern_index: {feature: loading_value}}
            for the top features in each pattern.
    """
    logger.info("Auto-labeling %d patterns (top %d features each) ...", len(H_matrix), top_n)

    raw_labels = []
    feature_importance = {}

    for pattern_idx, pattern_vector in enumerate(H_matrix):
        # Get top-N feature indices by loading value
        top_indices = np.argsort(pattern_vector)[::-1][:top_n]
        top_features = {
            feature_names[i]: float(pattern_vector[i]) for i in top_indices
        }
        feature_importance[pattern_idx] = top_features

        # Determine categories for top features
        categories = []
        for feat_name in top_features:
            cat = _classify_feature(feat_name, feature_categories)
            categories.append(cat)

        # Generate label from the dominant category
        label = _generate_pattern_label(pattern_idx, categories, top_features)
        raw_labels.append(label)

    # Deduplicate labels to ensure uniqueness
    labels = []
    counts = {}
    for label in raw_labels:
        counts[label] = counts.get(label, 0) + 1

    assigned = {}
    for pattern_idx, label in enumerate(raw_labels):
        if counts[label] > 1:
            assigned[label] = assigned.get(label, 0) + 1
            unique_label = f"{label} {assigned[label]}"
        else:
            unique_label = label
        labels.append(unique_label)

        logger.info(
            "  Pattern %d: '%s' - top features: %s",
            pattern_idx,
            unique_label,
            {k: round(v, 4) for k, v in feature_importance[pattern_idx].items()},
        )

    return labels, feature_importance


def align_patterns(
    H_new: np.ndarray,
    H_previous: np.ndarray,
    similarity_threshold: float = 0.6,
) -> Tuple[Dict[int, object], Dict, float]:
    """
    Align new patterns to previous patterns using the Hungarian algorithm.

    Uses cosine similarity between pattern vectors (rows of H) to build
    a cost matrix, then finds the optimal one-to-one assignment.

    Args:
        H_new: New pattern matrix (k_new x features).
        H_previous: Previous pattern matrix (k_old x features).
        similarity_threshold: Minimum cosine similarity for a match to be
            considered valid. Below this, the new pattern is marked as
            "NEW_PATTERN" (a genuinely new behavioral pattern has emerged).
            Default 0.6 chosen as a moderate threshold: patterns with >60%
            cosine similarity share enough structure to be considered the
            "same" pattern. Literature suggests 0.5-0.7 range.

    Returns:
        Tuple of:
          - alignment_map: Dict {new_idx: old_idx or "NEW_PATTERN"}.
          - similarity_scores: Dict {new_idx: best_similarity_score}.
          - alignment_quality: Mean similarity of matched pairs (excluding
            NEW_PATTERN entries).
    """
    logger.info("Aligning %d new patterns against %d previous patterns ...",
                len(H_new), len(H_previous))

    # Ensure both have the same number of features
    n_features_new = H_new.shape[1]
    n_features_old = H_previous.shape[1]
    if n_features_new != n_features_old:
        # Pad the smaller one with zeros
        max_features = max(n_features_new, n_features_old)
        if n_features_new < max_features:
            H_new = np.pad(H_new, ((0, 0), (0, max_features - n_features_new)))
        if n_features_old < max_features:
            H_previous = np.pad(H_previous, ((0, 0), (0, max_features - n_features_old)))

    # Compute similarity matrix (higher = more similar)
    sim_matrix = cosine_similarity(H_new, H_previous)

    # Convert to cost matrix for Hungarian algorithm (it minimises)
    cost_matrix = 1.0 - sim_matrix

    # Handle rectangular matrices (different k values)
    n_new, n_old = cost_matrix.shape
    if n_new != n_old:
        # Pad to make square
        max_dim = max(n_new, n_old)
        padded = np.ones((max_dim, max_dim))  # High cost for dummy assignments
        padded[:n_new, :n_old] = cost_matrix
        row_indices, col_indices = linear_sum_assignment(padded)
    else:
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

    alignment_map = {}
    similarity_scores = {}
    matched_similarities = []

    for new_idx, old_idx in zip(row_indices, col_indices):
        if new_idx >= n_new or old_idx >= n_old:
            continue  # Skip dummy assignments

        sim = sim_matrix[new_idx, old_idx]
        similarity_scores[int(new_idx)] = float(sim)

        if sim >= similarity_threshold:
            alignment_map[int(new_idx)] = int(old_idx)
            matched_similarities.append(sim)
            logger.info(
                "  Pattern %d -> Previous Pattern %d (similarity: %.4f)",
                new_idx, old_idx, sim,
            )
        else:
            alignment_map[int(new_idx)] = "NEW_PATTERN"
            logger.info(
                "  Pattern %d -> NEW_PATTERN (best similarity: %.4f < threshold %.2f)",
                new_idx, sim, similarity_threshold,
            )

    alignment_quality = float(np.mean(matched_similarities)) if matched_similarities else 0.0

    logger.info("  Alignment quality (mean matched similarity): %.4f", alignment_quality)
    return alignment_map, similarity_scores, alignment_quality


# ===================================================================
# Private helpers
# ===================================================================

def _compute_h_matrix_stability(h_matrices: List[np.ndarray]) -> float:
    """
    Compute stability as mean pairwise cosine similarity between H matrices.

    For each pair of H matrices, aligns them using the Hungarian algorithm
    and computes the mean cosine similarity of matched patterns.

    Returns:
        float: Mean stability score across all pairs (0 to 1).
    """
    if len(h_matrices) < 2:
        return 1.0

    pair_similarities = []
    for i in range(len(h_matrices)):
        for j in range(i + 1, len(h_matrices)):
            sim = _aligned_cosine_similarity(h_matrices[i], h_matrices[j])
            pair_similarities.append(sim)

    return float(np.mean(pair_similarities))


def _aligned_cosine_similarity(H1: np.ndarray, H2: np.ndarray) -> float:
    """
    Compute aligned cosine similarity between two H matrices.

    Uses the Hungarian algorithm to find the optimal pattern assignment,
    then returns the mean cosine similarity of matched pairs.
    """
    sim_matrix = cosine_similarity(H1, H2)
    cost_matrix = 1.0 - sim_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    similarities = [sim_matrix[r, c] for r, c in zip(row_indices, col_indices)]
    return float(np.mean(similarities)) if similarities else 0.0


def _classify_feature(
    feature_name: str,
    feature_categories: Optional[Dict[str, str]] = None,
) -> str:
    """
    Classify a feature into a behavioral category.

    First checks the feature_categories dict (from feature_registry),
    then falls back to keyword matching via FEATURE_CATEGORY_KEYWORDS.
    """
    # Try registry first
    if feature_categories and feature_name in feature_categories:
        return feature_categories[feature_name]

    # Fall back to keyword matching
    lower = feature_name.lower()
    for category, keywords in FEATURE_CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower:
                return category

    return "unknown"


def _generate_pattern_label(
    pattern_idx: int,
    categories: List[str],
    top_features: Dict[str, float],
) -> str:
    """
    Generate a human-readable label from dominant categories.

    Examples:
      - "Engagement-Dominant Pattern"
      - "Revenue & Support Pattern"
      - "High-Risk Pattern"
    """
    # Count category occurrences
    cat_counts = {}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])

    if len(sorted_cats) == 0:
        return f"Pattern-{pattern_idx}"

    primary = sorted_cats[0][0].title()

    if len(sorted_cats) == 1 or sorted_cats[0][1] > sorted_cats[1][1]:
        return f"{primary}-Dominant Pattern"
    else:
        secondary = sorted_cats[1][0].title()
        return f"{primary} & {secondary} Pattern"
