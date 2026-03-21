"""
Metrics Module - Performance metric functions
=============================================
Federal University of Bahia – MSc in Industrial Engineering
Author: Ghabriel Anton Gomes de Sá
"""

import numpy as np
from typing import Tuple


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean((y_true - y_pred) ** 2))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(calculate_mse(y_true, y_pred)))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R²."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - (ss_res / ss_tot))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute all core metrics at once.

    Returns:
        Tuple: (mse, rmse, r2)
    """
    mse = calculate_mse(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = calculate_r2(y_true, y_pred)
    return mse, rmse, r2


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """Compute Mean Absolute Percentage Error (MAPE, in %)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))) * 100


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def calculate_bce(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Binary Cross-Entropy loss."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def calculate_accuracy(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    """Compute accuracy = (TP + TN) / total."""
    y_true = np.asarray(y_true).flatten()
    y_pred_class = np.asarray(y_pred_class).flatten()
    return float(np.mean(y_true == y_pred_class))


def calculate_precision(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    """Compute precision = TP / (TP + FP)."""
    y_true = np.asarray(y_true).flatten()
    y_pred_class = np.asarray(y_pred_class).flatten()
    tp = np.sum((y_pred_class == 1) & (y_true == 1))
    fp = np.sum((y_pred_class == 1) & (y_true == 0))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def calculate_recall(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    """Compute recall = TP / (TP + FN)."""
    y_true = np.asarray(y_true).flatten()
    y_pred_class = np.asarray(y_pred_class).flatten()
    tp = np.sum((y_pred_class == 1) & (y_true == 1))
    fn = np.sum((y_pred_class == 0) & (y_true == 1))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def calculate_f1(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    """Compute F1 = 2 * precision * recall / (precision + recall)."""
    p = calculate_precision(y_true, y_pred_class)
    r = calculate_recall(y_true, y_pred_class)
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def calculate_mcc(y_true: np.ndarray, y_pred_class: np.ndarray) -> float:
    """Compute Matthews Correlation Coefficient."""
    y_true = np.asarray(y_true).flatten()
    y_pred_class = np.asarray(y_pred_class).flatten()
    tp = np.sum((y_pred_class == 1) & (y_true == 1))
    tn = np.sum((y_pred_class == 0) & (y_true == 0))
    fp = np.sum((y_pred_class == 1) & (y_true == 0))
    fn = np.sum((y_pred_class == 0) & (y_true == 1))
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def calculate_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                      threshold: float = 0.5) -> dict:
    """
    Compute all classification metrics from probabilities.

    Args:
        y_true: True binary labels {0, 1}
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dict with bce, accuracy, precision, recall, f1, mcc
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    y_pred_class = (y_pred_proba >= threshold).astype(int)

    return {
        'bce': calculate_bce(y_true, y_pred_proba),
        'accuracy': calculate_accuracy(y_true, y_pred_class),
        'precision': calculate_precision(y_true, y_pred_class),
        'recall': calculate_recall(y_true, y_pred_class),
        'f1': calculate_f1(y_true, y_pred_class),
        'mcc': calculate_mcc(y_true, y_pred_class),
    }