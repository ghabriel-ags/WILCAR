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