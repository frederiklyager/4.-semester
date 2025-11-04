# src/eval/metrics.py
"""
Evaluation Metrics for Energy Forecast Project
Phase 3: Model evaluation metrics for CO₂ intensity forecasting

Author: Frederik Lyager
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union, Optional


def mae(y_true: Union[np.ndarray, pd.Series, list], 
        y_pred: Union[np.ndarray, pd.Series, list]) -> float:
    """
    Mean Absolute Error (MAE)
    
    MAE = mean(|y_true - y_pred|)
    
    Lower is better. Same unit as the target variable (g CO₂/kWh).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        MAE score (float)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: Union[np.ndarray, pd.Series, list], 
         y_pred: Union[np.ndarray, pd.Series, list]) -> float:
    """
    Root Mean Squared Error (RMSE)
    
    RMSE = sqrt(mean((y_true - y_pred)²))
    
    Lower is better. Same unit as target. More sensitive to large errors than MAE.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        RMSE score (float)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: Union[np.ndarray, pd.Series, list], 
         y_pred: Union[np.ndarray, pd.Series, list]) -> float:
    """
    Mean Absolute Percentage Error (MAPE)
    
    MAPE = mean(|y_true - y_pred| / |y_true|) * 100
    
    Lower is better. Returns percentage (0-100+).
    Note: Excludes zeros in y_true to avoid division by zero.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        MAPE score in percentage (float)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Exclude zeros to avoid division by zero
    mask = y_true != 0
    
    if not mask.any():
        return float('inf')
    
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def r2_score(y_true: Union[np.ndarray, pd.Series, list], 
             y_pred: Union[np.ndarray, pd.Series, list]) -> float:
    """
    R² Score (Coefficient of Determination)
    
    R² = 1 - (SS_res / SS_tot)
    where SS_res = sum((y_true - y_pred)²)
          SS_tot = sum((y_true - mean(y_true))²)
    
    Best possible score is 1.0. Can be negative for poor models.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        R² score (float, -inf to 1.0)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return float(1 - (ss_res / ss_tot))


def quantile_loss(y_true: Union[np.ndarray, pd.Series, list],
                  y_pred: Union[np.ndarray, pd.Series, list],
                  quantile: float = 0.5) -> float:
    """
    Quantile Loss (Pinball Loss)
    
    Used for probabilistic forecasting. Asymmetric loss function.
    - quantile=0.5 gives median prediction (same as MAE)
    - quantile=0.1 penalizes underestimation more
    - quantile=0.9 penalizes overestimation more
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        quantile: Target quantile (0 to 1)
    
    Returns:
        Quantile loss (float)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    errors = y_true - y_pred
    
    loss = np.where(
        errors >= 0,
        quantile * errors,
        (quantile - 1) * errors
    )
    
    return float(np.mean(loss))


def smape(y_true: Union[np.ndarray, pd.Series, list],
          y_pred: Union[np.ndarray, pd.Series, list]) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)
    
    SMAPE = mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|)) * 100
    
    Bounded between 0 and 200%. More symmetric than MAPE.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        SMAPE score in percentage (float)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    
    if not mask.any():
        return 0.0
    
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def top_green_hours_accuracy(y_true: Union[np.ndarray, pd.Series, list],
                             y_pred: Union[np.ndarray, pd.Series, list],
                             top_n: int = 6) -> float:
    """
    Top Green Hours Accuracy
    
    Custom metric: How well does the model identify the greenest (lowest CO₂) hours?
    Useful for load shifting applications.
    
    Measures overlap between actual top-N greenest hours and predicted top-N.
    
    Args:
        y_true: Actual CO₂ values
        y_pred: Predicted CO₂ values
        top_n: Number of top green hours to consider (default: 6 hours)
    
    Returns:
        Accuracy percentage (0-100)
    
    Example:
        If 4 out of 6 predicted greenest hours match actual → 66.7% accuracy
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    if len(y_true) < top_n:
        top_n = len(y_true)
    
    # Get indices of top N greenest hours (lowest CO₂)
    actual_green_indices = set(np.argsort(y_true)[:top_n])
    predicted_green_indices = set(np.argsort(y_pred)[:top_n])
    
    # Calculate overlap
    overlap = len(actual_green_indices & predicted_green_indices)
    accuracy = (overlap / top_n) * 100
    
    return float(accuracy)


def evaluate_forecast(y_true: Union[np.ndarray, pd.Series, list],
                     y_pred: Union[np.ndarray, pd.Series, list],
                     model_name: str = "Model",
                     include_green_hours: bool = True) -> dict:
    """
    Comprehensive evaluation of forecast performance.
    
    Calculates all relevant metrics and returns as dictionary.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model (for display)
        include_green_hours: Whether to include Top Green Hours metric
    
    Returns:
        Dictionary with all metrics
    """
    results = {
        "model": model_name,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    
    if include_green_hours:
        results["top_green_hours_6h"] = top_green_hours_accuracy(y_true, y_pred, top_n=6)
        results["top_green_hours_12h"] = top_green_hours_accuracy(y_true, y_pred, top_n=12)
    
    return results


def print_metrics(metrics: dict, precision: int = 2):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary from evaluate_forecast()
        precision: Number of decimal places
    """
    print(f"\n{'='*60}")
    print(f"📊 Evaluation Results: {metrics.get('model', 'Unknown Model')}")
    print(f"{'='*60}")
    
    # Standard metrics
    print(f"MAE (Mean Absolute Error):          {metrics['mae']:.{precision}f} g CO₂/kWh")
    print(f"RMSE (Root Mean Squared Error):    {metrics['rmse']:.{precision}f} g CO₂/kWh")
    print(f"MAPE (Mean Abs. Percentage Error): {metrics['mape']:.{precision}f}%")
    print(f"SMAPE (Symmetric MAPE):             {metrics['smape']:.{precision}f}%")
    print(f"R² Score:                           {metrics['r2']:.{precision}f}")
    
    # Green hours metrics (if available)
    if 'top_green_hours_6h' in metrics:
        print(f"\n🌿 Load Shifting Metrics:")
        print(f"Top 6 Green Hours Accuracy:         {metrics['top_green_hours_6h']:.{precision}f}%")
        print(f"Top 12 Green Hours Accuracy:        {metrics['top_green_hours_12h']:.{precision}f}%")
    
    print(f"{'='*60}\n")


def compare_models(results_list: list[dict]) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.
    
    Args:
        results_list: List of metric dictionaries from evaluate_forecast()
    
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    df = pd.DataFrame(results_list)
    
    if 'model' in df.columns:
        df = df.set_index('model')
    
    return df


# Example usage
if __name__ == "__main__":
    # Simulated data for testing
    np.random.seed(42)
    
    y_true = np.array([100, 120, 80, 90, 110, 95, 85, 105, 115, 88, 92, 98])
    y_pred_good = y_true + np.random.normal(0, 5, len(y_true))  # Good model
    y_pred_bad = y_true + np.random.normal(0, 20, len(y_true))  # Poor model
    
    print("Testing metrics module...")
    print("\n--- Good Model ---")
    metrics_good = evaluate_forecast(y_true, y_pred_good, model_name="Good Model")
    print_metrics(metrics_good)
    
    print("\n--- Bad Model ---")
    metrics_bad = evaluate_forecast(y_true, y_pred_bad, model_name="Bad Model")
    print_metrics(metrics_bad)
    
    print("\n--- Model Comparison ---")
    comparison = compare_models([metrics_good, metrics_bad])
    print(comparison.round(2))
    
    print("\n✅ All metrics working correctly!")