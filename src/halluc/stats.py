"""Small statistics helpers: AUROC, image-level bootstrap, selective prediction."""
import numpy as np


def auroc(y, score):
    """Area under the ROC curve for binary labels y (1 = positive) and scores
    (higher = more likely positive). Ties count half. Returns nan if one class is missing."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):           # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auroc_ci(y, score, groups, n_boot=1000, seed=0, level=0.95):
    """Percentile CI for the AUROC, resampling whole groups (e.g. images) with replacement."""
    y, score, groups = np.asarray(y), np.asarray(score), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        a = auroc(y[idx], score[idx])
        if not np.isnan(a):
            vals.append(a)
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def cv_logistic_auroc(X, y, groups, n_splits=5, seed=0):
    """Out-of-fold AUROC of a logistic regression, with folds split by group (image)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, groups = np.asarray(X, dtype=float), np.asarray(y).astype(int), np.asarray(groups)
    # Shuffle group order deterministically so folds are not sorted by file name.
    uniq = np.unique(groups)
    perm = {g: i for i, g in enumerate(np.random.default_rng(seed).permutation(uniq))}
    gid = np.array([perm[g] for g in groups])
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, gid):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return auroc(y, pred)


def risk_coverage(correct, confidence):
    """Selective prediction: keep the most confident examples first.

    Returns (coverage, accuracy) arrays: accuracy on the kept examples when the
    top-k most confident are answered and the rest abstained, for k = 1..n.
    """
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="mergesort")
    c = correct[order]
    k = np.arange(1, len(c) + 1)
    return k / len(c), np.cumsum(c) / k


def area_under_accuracy_coverage(correct, confidence):
    cov, acc = risk_coverage(correct, confidence)
    return float(np.mean(acc))


def accuracy_at_threshold(correct, confidence, threshold):
    """Coverage and accuracy when answering only examples with confidence >= threshold."""
    correct = np.asarray(correct, dtype=float)
    keep = np.asarray(confidence, dtype=float) >= threshold
    cov = float(keep.mean())
    acc = float(correct[keep].mean()) if keep.any() else float("nan")
    return cov, acc


def threshold_for_coverage(confidence, coverage):
    """Confidence threshold that keeps (about) the given fraction of examples."""
    return float(np.quantile(np.asarray(confidence, dtype=float), 1 - coverage))
