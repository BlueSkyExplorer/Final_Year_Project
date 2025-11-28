import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")


def per_class_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)


def _safe_confusion(y_true, y_pred):
    """Return tn, fp, fn, tp; fall back to NaN if a class is missing."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return np.nan, np.nan, np.nan, np.nan
    tn, fp, fn, tp = cm.ravel()
    return float(tn), float(fp), float(fn), float(tp)


def binary_metrics(y_true, y_score, threshold=0.5):
    """Compute binary metrics safely using scores (probabilities preferred).

    Args:
        y_true: array-like of 0/1 labels where 1 represents the positive class.
        y_score: array-like of continuous scores or probabilities for the positive class.
        threshold: decision threshold for converting scores to predictions.
    """

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = _safe_confusion(y_true, y_pred)

    def _safe_ratio(num, den):
        return float(num) / float(den) if den not in (0.0, 0) else np.nan

    specificity = _safe_ratio(tn, tn + fp)
    sensitivity = _safe_ratio(tp, tp + fn)
    ppv = _safe_ratio(tp, tp + fp)
    npv = _safe_ratio(tn, tn + fn)

    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = np.nan

    try:
        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
    except ValueError:
        acc = np.nan
        kappa = np.nan

    return {
        "auroc": float(auroc) if auroc == auroc else np.nan,
        "accuracy": float(acc) if acc == acc else np.nan,
        "sensitivity": float(sensitivity) if sensitivity == sensitivity else np.nan,
        "specificity": float(specificity) if specificity == specificity else np.nan,
        "ppv": float(ppv) if ppv == ppv else np.nan,
        "npv": float(npv) if npv == npv else np.nan,
        "kappa": float(kappa) if kappa == kappa else np.nan,
    }


def evaluate_all(y_true, y_pred, y_proba=None):
    """Compute multiclass and derived binary metrics.

    Args:
        y_true: array-like of MES labels (0-3).
        y_pred: array-like of predicted MES labels (0-3).
        y_proba: optional scores/probabilities. If shape (N, 4), treated as
            class probabilities. If shape (N,), treated as a severity score
            for both binaries.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class_f1": per_class_f1(y_true, y_pred).tolist(),
        "accuracy": accuracy_score(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }

    # Binary tasks
    bin_01_23_true = (y_true >= 2).astype(int)  # active disease (2/3) vs remission (0/1)
    bin_0_123_true = (y_true > 0).astype(int)   # any disease (1/2/3) vs inactive (0)

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim == 2 and y_proba.shape[1] == 4:
            p0, p1, p2, p3 = y_proba[:, 0], y_proba[:, 1], y_proba[:, 2], y_proba[:, 3]
            score_01_23 = p2 + p3
            score_0_123 = p1 + p2 + p3
        else:
            score_01_23 = y_proba
            score_0_123 = y_proba
    else:
        score_01_23 = (y_pred >= 2).astype(float)
        score_0_123 = (y_pred > 0).astype(float)

    metrics["bin_01_23"] = binary_metrics(bin_01_23_true, score_01_23)
    metrics["bin_0_123"] = binary_metrics(bin_0_123_true, score_0_123)

    return metrics
