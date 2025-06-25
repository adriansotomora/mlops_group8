"""
Regression model evaluation utilities.

Regression model evaluation utilities.
- Computes standard regression metrics.
- Logs metrics.
- Designed for modular use with statsmodels or sklearn-like models.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, Optional, List
import statsmodels.api as sm  # For type hinting and potential direct use

# Initialize logger for this module.
# Assumes logging is configured by the calling script (e.g., model_draft.py)
logger = logging.getLogger(__name__)


def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error, handling zeros in y_true."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        logger.warning("MAPE calculation failed: all true values are zero.")
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def round_metrics_dict(
        metrics_dict: Dict[str, Any], ndigits: int = 4) -> Dict[str, Any]:
    """Round float values in a metrics dictionary."""
    rounded = {}
    for k, v in metrics_dict.items():
        if isinstance(
                v,
                dict):  # For nested dicts, like confusion matrix (not used here but good pattern)
            rounded[k] = {
                ik: (
                    round(iv, ndigits) if isinstance(
                        iv, (float, np.floating, int)
                    ) else iv
                ) for ik, iv in v.items()
            }
        elif isinstance(v, (float, np.floating, int)):
            rounded[k] = round(float(v), ndigits)
        else:
            rounded[k] = v
    return rounded


def calculate_regression_metrics_from_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    num_features: int
) -> Dict[str, float]:
    """
    Calculate a standard set of regression metrics from true and predicted values.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        num_features: Number of features used in the model to calculate Adjusted R-squared.

    Returns:
        Dictionary of metric names and their values.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length of y_true ({len(y_true)}) and y_pred ({
                len(y_pred)}) must be the same."
        )
    if y_true.empty:
        logger.warning("y_true is empty. Returning NaN for all metrics.")
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "mape": np.nan}

    metrics = {}
    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)
    metrics["mape"] = _calculate_mape(y_true.values, y_pred.values)

    # Adjusted R-squared
    n = len(y_true)
    p = num_features
    if n - p - 1 > 0:
        metrics["adj_r2"] = 1 - (1 - metrics["r2"]) * (n - 1) / (n - p - 1)
    else:
        logger.warning(
            f"Cannot calculate Adjusted R-squared: n ({n}) - p ({p}) - 1 "
            "is not positive."
        )
        metrics["adj_r2"] = np.nan

    return metrics


def evaluate_statsmodels_model(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    selected_features: List[str],
    split_label: Optional[str] = None  # e.g., "test", "validation"
) -> Dict[str, float]:
    """
    Evaluate a trained statsmodels OLS regression model.

    Args:
        model: Trained statsmodels OLS model.
        X_eval: DataFrame of features for evaluation.
        y_eval: Series of true target values for evaluation.
        selected_features: List of feature names the model was trained on.
        split_label: Optional string to label the evaluation set (e.g., "test").

    Returns:
        Dictionary of calculated regression metrics (unrounded).
    """
    if not selected_features:
        logger.warning(
            f"No features provided for evaluation of '{split_label}' split. "
            "Returning NaN metrics."
        )
        # Return NaN for a standard set of metrics
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "mape": np.nan}

    if X_eval.empty or y_eval.empty:
        logger.warning(f"X_eval or y_eval is empty for '{
                       split_label}' split. Returning NaN metrics.")
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "mape": np.nan}

    X_eval_selected = X_eval[selected_features].astype(float)
    # Align y_eval with X_eval_selected's index, in case X_eval had rows dropped (e.g. due to NaNs)
    # that were not present in y_eval originally, or vice-versa.
    common_index = X_eval_selected.index.intersection(y_eval.index)
    if len(common_index) != len(X_eval_selected) or len(
            common_index) != len(y_eval):
        logger.warning(
            f"Indices of X_eval_selected and y_eval for '{split_label}' "
            f"split do not perfectly match. Using common intersection of "
            f"size {len(common_index)} for evaluation."
        )
    X_eval_aligned = X_eval_selected.loc[common_index]
    y_eval_aligned = y_eval.loc[common_index]

    if X_eval_aligned.empty or y_eval_aligned.empty:
        logger.warning(
            f"X_eval or y_eval became empty after alignment for "
            f"'{split_label}' split. Returning NaN metrics."
        )
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "mape": np.nan}

    X_eval_const = sm.add_constant(X_eval_aligned)

    raw_metrics = {}
    try:
        y_pred = model.predict(X_eval_const)
        # Ensure y_pred is a Series with matching index
        y_pred_series = pd.Series(y_pred, index=y_eval_aligned.index)

        raw_metrics = calculate_regression_metrics_from_predictions(
            y_true=y_eval_aligned,
            y_pred=y_pred_series,
            num_features=len(selected_features)
        )
    except Exception as e:
        logger.error(f"Error during prediction or metric calculation for '{
                     split_label}' split: {e}")
        logger.debug(
            f"Model exog_names: {
                model.model.exog_names if hasattr(
                    model, 'model') else 'N/A'}")
        logger.debug(f"X_eval_const columns: {X_eval_const.columns.tolist()}")
        logger.debug(f"X_eval_const shape: {X_eval_const.shape}")
        # Return NaN for a standard set of metrics if prediction fails
        raw_metrics = {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "adj_r2": np.nan,
            "mape": np.nan}

    # Log rounded metrics
    rounded_metrics = round_metrics_dict(raw_metrics)
    log_msg_split = f" for '{split_label}' split" if split_label else ""
    logger.info(f"Regression Evaluation Metrics{
                log_msg_split}: {rounded_metrics}")

    return raw_metrics  # Return the original (unrounded) metrics
