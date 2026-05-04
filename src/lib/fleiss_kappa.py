"""
Fleiss' kappa for inter-rater reliability with bootstrap confidence intervals.
For 5 raters, 2 categories (CODE / KNOWLEDGE).

Reference: Fleiss, J.L. (1971). Measuring nominal scale agreement among many raters.
           Landis, J.R. & Koch, G.G. (1977). The measurement of observer agreement for categorical data.
"""

import numpy as np


def compute_fleiss_kappa(rating_matrix):
    """
    Compute Fleiss' kappa.

    Args:
        rating_matrix: numpy array of shape (N, k) where
            N = number of items (prompts)
            k = number of categories (2: CODE, KNOWLEDGE)
            rating_matrix[i][j] = number of raters who assigned category j to item i

    Returns:
        dict with kappa, P_bar (observed agreement), P_e (expected agreement)
    """
    N, k = rating_matrix.shape
    n = rating_matrix.sum(axis=1)[0]  # number of raters per item (should be constant)

    # P_i for each item
    P_i = (1.0 / (n * (n - 1))) * (np.sum(rating_matrix ** 2, axis=1) - n)

    # Mean observed agreement
    P_bar = np.mean(P_i)

    # Proportion of all assignments to each category
    p_j = np.sum(rating_matrix, axis=0) / (N * n)

    # Expected agreement by chance
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        kappa = 1.0
    else:
        kappa = (P_bar - P_e) / (1.0 - P_e)

    return {
        "kappa": float(kappa),
        "P_bar": float(P_bar),
        "P_e": float(P_e),
        "N": int(N),
        "n_raters": int(n),
        "k_categories": int(k),
    }


def bootstrap_ci(rating_matrix, n_bootstrap=10000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence interval for Fleiss' kappa.

    Args:
        rating_matrix: (N, k) array
        n_bootstrap: number of bootstrap iterations
        ci: confidence level (default 0.95)
        seed: random seed for reproducibility

    Returns:
        dict with kappa, ci_lower, ci_upper
    """
    rng = np.random.RandomState(seed)
    N = rating_matrix.shape[0]

    base_result = compute_fleiss_kappa(rating_matrix)
    kappa_samples = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        indices = rng.choice(N, size=N, replace=True)
        sample = rating_matrix[indices]
        try:
            result = compute_fleiss_kappa(sample)
            kappa_samples[b] = result["kappa"]
        except Exception:
            kappa_samples[b] = np.nan

    valid = kappa_samples[~np.isnan(kappa_samples)]
    alpha = 1 - ci
    ci_lower = float(np.percentile(valid, 100 * alpha / 2))
    ci_upper = float(np.percentile(valid, 100 * (1 - alpha / 2)))

    return {
        "kappa": base_result["kappa"],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
        "ci_level": ci,
        "n_valid_samples": len(valid),
    }


def build_rating_matrix(results, judge_short_names):
    """
    Build the rating matrix from classification results.

    Args:
        results: list of dicts, each with 'classifications' dict mapping judge name -> label
        judge_short_names: list of judge short names to include

    Returns:
        rating_matrix: (N, 2) array where col 0 = CODE count, col 1 = KNOWLEDGE count
        excluded: list of indices excluded due to ERROR/UNCLEAR
    """
    matrix_rows = []
    excluded = []

    for i, r in enumerate(results):
        code_count = 0
        knowledge_count = 0
        valid = True

        for judge in judge_short_names:
            label = r.get("classifications", {}).get(judge, "")
            if label == "CODE":
                code_count += 1
            elif label == "KNOWLEDGE":
                knowledge_count += 1
            else:
                valid = False
                break

        if valid and (code_count + knowledge_count) == len(judge_short_names):
            matrix_rows.append([code_count, knowledge_count])
        else:
            excluded.append(i)

    return np.array(matrix_rows), excluded


def interpret_kappa(kappa):
    """Interpret kappa using Landis & Koch (1977) scale."""
    if kappa < 0:
        return "Poor"
    elif kappa <= 0.20:
        return "Slight"
    elif kappa <= 0.40:
        return "Fair"
    elif kappa <= 0.60:
        return "Moderate"
    elif kappa <= 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"
