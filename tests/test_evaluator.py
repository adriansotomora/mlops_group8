import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm # For creating a dummy model
from unittest.mock import MagicMock, patch
import logging

# Ensure 'src' directory is in PYTHONPATH for imports
import sys
import os
TEST_DIR_EVAL = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_EVAL = os.path.abspath(os.path.join(TEST_DIR_EVAL, '..'))
if PROJECT_ROOT_EVAL not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_EVAL)

# Import the module to be tested
from src.evaluation import evaluator 

# Logger for test setup information (useful for debugging tests)
test_setup_logger_eval = logging.getLogger("test_evaluator_setup")
if not test_setup_logger_eval.handlers:
    handler_eval = logging.StreamHandler(sys.stdout)
    formatter_eval = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_eval.setFormatter(formatter_eval)
    test_setup_logger_eval.addHandler(handler_eval)
    test_setup_logger_eval.setLevel(logging.INFO)


# --- Tests for _calculate_mape ---
def test_calculate_mape_basic():
    """Test basic MAPE calculation."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 180, 330])
    # Expected: (|10/100| + |-20/200| + |30/300|) / 3 * 100 = 10.0
    assert np.isclose(evaluator._calculate_mape(y_true, y_pred), 10.0)

def test_calculate_mape_with_zeros_in_true():
    """Test MAPE calculation when y_true contains zeros."""
    y_true = np.array([0, 200, 300])
    y_pred = np.array([10, 180, 330]) # The pair (0,10) should be ignored
    # Expected: (|-20/200| + |30/300|) / 2 * 100 = 10.0
    assert np.isclose(evaluator._calculate_mape(y_true, y_pred), 10.0)

def test_calculate_mape_all_zeros_in_true(caplog):
    """Test MAPE when all y_true values are zero (should return NaN and log warning)."""
    y_true = np.array([0, 0, 0])
    y_pred = np.array([10, 20, 30])
    with caplog.at_level(logging.WARNING):
        result = evaluator._calculate_mape(y_true, y_pred)
    assert np.isnan(result)
    assert "MAPE calculation failed: all true values are zero." in caplog.text

def test_calculate_mape_empty_input():
    """Test MAPE with empty input arrays."""
    y_true = np.array([])
    y_pred = np.array([])
    assert np.isnan(evaluator._calculate_mape(y_true, y_pred)) # np.mean of empty array is NaN

# --- Tests for round_metrics_dict ---
def test_round_metrics_dict_basic():
    """Test rounding of float values in a metrics dictionary."""
    metrics = {"mse": 12.34567, "r2": 0.98765, "count": 100}
    rounded = evaluator.round_metrics_dict(metrics, ndigits=2)
    assert rounded["mse"] == 12.35
    assert rounded["r2"] == 0.99
    assert rounded["count"] == 100 # Non-float types should remain unchanged

def test_round_metrics_dict_nested():
    """Test rounding for nested dictionaries within metrics."""
    metrics = {"main": {"mse": 12.34567}, "other": "value"}
    rounded = evaluator.round_metrics_dict(metrics, ndigits=3)
    assert rounded["main"]["mse"] == 12.346
    assert rounded["other"] == "value"

# --- Tests for calculate_regression_metrics_from_predictions ---
@pytest.fixture
def sample_predictions():
    """Provides sample true values, predicted values, and number of features."""
    y_true = pd.Series([10, 20, 30, 40, 50])
    y_pred = pd.Series([12, 18, 33, 38, 52]) 
    num_features = 2
    return y_true, y_pred, num_features

def test_calculate_regression_metrics_basic(sample_predictions):
    """Test calculation of standard regression metrics."""
    y_true, y_pred, num_features = sample_predictions
    metrics = evaluator.calculate_regression_metrics_from_predictions(y_true, y_pred, num_features)
    
    expected_keys = ["mse", "rmse", "mae", "r2", "adj_r2", "mape"]
    for key in expected_keys:
        assert key in metrics

    assert metrics["mse"] > 0
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert -float('inf') < metrics["r2"] <= 1.0 # R-squared can be negative for poor fits
    assert -float('inf') < metrics["adj_r2"] <= 1.0
    assert metrics["mape"] >= 0

def test_calculate_regression_metrics_perfect_fit():
    """Test metrics for a perfect prediction scenario, including Adjusted R2 edge cases."""
    y_true = pd.Series([10, 20, 30])
    y_pred = pd.Series([10, 20, 30]) 
    
    # Case 1: n - p - 1 = 0 (3 samples, 2 features). Adjusted R2 should be NaN.
    num_features_edge = 2 
    metrics_edge = evaluator.calculate_regression_metrics_from_predictions(y_true, y_pred, num_features_edge)
    assert np.isclose(metrics_edge["mse"], 0.0)
    assert np.isclose(metrics_edge["rmse"], 0.0)
    assert np.isclose(metrics_edge["mae"], 0.0)
    assert np.isclose(metrics_edge["r2"], 1.0)
    assert np.isnan(metrics_edge["adj_r2"]) 
    assert np.isclose(metrics_edge["mape"], 0.0)

    # Case 2: n - p - 1 > 0 (3 samples, 1 feature). Adjusted R2 should be 1.0 for perfect R2.
    num_features_ok = 1 
    metrics_ok = evaluator.calculate_regression_metrics_from_predictions(y_true, y_pred, num_features_ok)
    assert np.isclose(metrics_ok["r2"], 1.0)
    assert np.isclose(metrics_ok["adj_r2"], 1.0)


def test_calculate_regression_metrics_adj_r2_edge_case(caplog):
    """Test Adjusted R2 calculation when n-p-1 is not positive, expecting a warning."""
    y_true = pd.Series([10, 20, 30])
    y_pred = pd.Series([12, 18, 33])
    num_features = 2 # n=3, p=2, so n-p-1 = 0
    with caplog.at_level(logging.WARNING):
        metrics = evaluator.calculate_regression_metrics_from_predictions(y_true, y_pred, num_features)
    assert np.isnan(metrics["adj_r2"])
    assert "Cannot calculate Adjusted R-squared" in caplog.text
    
    num_features_ok = 1 # n=3, p=1, so n-p-1 = 1 (valid case)
    metrics_ok = evaluator.calculate_regression_metrics_from_predictions(y_true, y_pred, num_features_ok)
    assert not np.isnan(metrics_ok["adj_r2"]) # Should be a valid number


def test_calculate_regression_metrics_value_error_length_mismatch():
    """Test ValueError is raised if y_true and y_pred have different lengths."""
    y_true = pd.Series([1, 2, 3])
    y_pred = pd.Series([1, 2])
    with pytest.raises(ValueError, match="Length of y_true .* and y_pred .* must be the same."):
        evaluator.calculate_regression_metrics_from_predictions(y_true, y_pred, 1)

def test_calculate_regression_metrics_empty_input(caplog):
    """Test behavior with empty y_true and y_pred (should return NaNs and log warning)."""
    y_true = pd.Series([], dtype=float) # Specify dtype for empty Series
    y_pred = pd.Series([], dtype=float)
    with caplog.at_level(logging.WARNING):
        metrics = evaluator.calculate_regression_metrics_from_predictions(y_true, y_pred, 1)
    assert all(np.isnan(v) for v in metrics.values())
    assert "y_true is empty. Returning NaN for all metrics." in caplog.text


# --- Tests for evaluate_statsmodels_model ---
@pytest.fixture
def dummy_statsmodels_model_and_data():
    """Creates a simple fitted OLS model and data for evaluate_statsmodels_model tests."""
    X_data = pd.DataFrame({
        'feature1': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0]), # Ensure float
        'feature2': np.array([2, 3, 2, 5, 4, 6, 5, 7, 6, 8.0])  # Ensure float
    })
    y_data = pd.Series(2 * X_data['feature1'] + 3 * X_data['feature2'] + np.random.normal(0, 0.1, 10))
    
    X_const = sm.add_constant(X_data.astype(float)) # Ensure data is float for OLS
    model = sm.OLS(y_data, X_const).fit()
    
    X_eval = X_data.sample(5, random_state=42).copy()
    y_eval = y_data.loc[X_eval.index].copy()
    
    selected_features = ['feature1', 'feature2']
    return model, X_eval, y_eval, selected_features

def test_evaluate_statsmodels_model_runs_and_returns_metrics(dummy_statsmodels_model_and_data):
    """Test main evaluation function for statsmodels, checking metrics and logging."""
    model, X_eval, y_eval, selected_features = dummy_statsmodels_model_and_data
    
    # Patch the logger within the 'evaluator' module to check its calls
    with patch.object(evaluator, 'logger', MagicMock()) as mock_eval_logger:
        metrics = evaluator.evaluate_statsmodels_model(model, X_eval, y_eval, selected_features, split_label="test_split")

    assert isinstance(metrics, dict)
    assert "mse" in metrics and "r2" in metrics
    assert not np.isnan(metrics["r2"]) 

    # Verify that the evaluator's logger made an info call containing the metrics summary
    assert any("Regression Evaluation Metrics for 'test_split' split" in call.args[0] for call in mock_eval_logger.info.call_args_list)


def test_evaluate_statsmodels_model_no_selected_features(dummy_statsmodels_model_and_data, caplog):
    """Test evaluation when no features are selected (should return NaNs and warn)."""
    model, X_eval, y_eval, _ = dummy_statsmodels_model_and_data
    with caplog.at_level(logging.WARNING):
        metrics = evaluator.evaluate_statsmodels_model(model, X_eval, y_eval, [], split_label="empty_features")
    assert all(np.isnan(v) for v in metrics.values())
    assert "No features provided for evaluation" in caplog.text

def test_evaluate_statsmodels_model_empty_eval_data(dummy_statsmodels_model_and_data, caplog):
    """Test evaluation with empty X_eval or y_eval (should return NaNs and warn)."""
    model, _, _, selected_features = dummy_statsmodels_model_and_data
    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype=float)
    with caplog.at_level(logging.WARNING):
        metrics = evaluator.evaluate_statsmodels_model(model, X_empty, y_empty, selected_features, split_label="empty_data")
    assert all(np.isnan(v) for v in metrics.values())
    assert "X_eval or y_eval is empty" in caplog.text


def test_evaluate_statsmodels_model_index_mismatch(dummy_statsmodels_model_and_data, caplog):
    """Test evaluation with misaligned indices between X_eval and y_eval."""
    model, X_eval_orig, y_eval_orig, selected_features = dummy_statsmodels_model_and_data
    
    X_eval = X_eval_orig.copy()
    y_eval_misaligned = y_eval_orig.copy()
    y_eval_misaligned.index = range(100, 100 + len(y_eval_misaligned)) # Create misaligned index

    with caplog.at_level(logging.WARNING):
         metrics = evaluator.evaluate_statsmodels_model(model, X_eval, y_eval_misaligned, selected_features, split_label="misaligned_idx")
    
    if X_eval.index.intersection(y_eval_misaligned.index).empty:
        # If no common index, all metrics should be NaN due to empty aligned data
        assert all(np.isnan(v) for v in metrics.values())
        assert "X_eval or y_eval became empty after alignment" in caplog.text
    else: 
        # If some common index exists, metrics should be calculable
        assert not all(np.isnan(v) for v in metrics.values())
        assert "Indices of X_eval_selected and y_eval for 'misaligned_idx' split do not perfectly match" in caplog.text


def test_evaluate_statsmodels_model_prediction_error(dummy_statsmodels_model_and_data, caplog):
    """Test graceful handling if model.predict raises an error (should return NaNs and log error)."""
    model_mock, X_eval, y_eval, selected_features = dummy_statsmodels_model_and_data
    
    model_mock.predict = MagicMock(side_effect=ValueError("Simulated prediction failure!"))

    with caplog.at_level(logging.ERROR): # Capture ERROR level logs
        metrics = evaluator.evaluate_statsmodels_model(model_mock, X_eval, y_eval, selected_features, "predict_error_test")
    
    assert all(np.isnan(v) for v in metrics.values())
    assert "Error during prediction or metric calculation" in caplog.text
    assert "Simulated prediction failure!" in caplog.text

