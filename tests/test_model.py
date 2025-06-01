import os
import pandas as pd
import numpy as np
import yaml
import pickle
import json
import pytest
from unittest.mock import MagicMock, patch
import logging
import statsmodels.api as sm # For type hinting and creating a dummy model
from pathlib import Path

# Ensure 'src' directory is in PYTHONPATH for imports
import sys
TEST_DIR_MODEL = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_MODEL = os.path.abspath(os.path.join(TEST_DIR_MODEL, '..'))
if PROJECT_ROOT_MODEL not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_MODEL)

# Import the module to be tested
from src.model import model # Assumes your main modeling script is src/model/model.py
# Import the evaluator; ensure this path matches your project structure and model.py's import
# model.py uses 'from src.evaluation.evaluator import evaluate_statsmodels_model'
from src.evaluation import evaluator # Assuming evaluator.py contains evaluate_statsmodels_model

# Logger for test setup information
test_setup_logger_model = logging.getLogger("test_model_setup")
if not test_setup_logger_model.handlers:
    handler_model = logging.StreamHandler(sys.stdout)
    formatter_model = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_model.setFormatter(formatter_model)
    test_setup_logger_model.addHandler(handler_model)
    test_setup_logger_model.setLevel(logging.INFO)


@pytest.fixture(scope="module")
def dummy_data_for_model_tests():
    """Provides dummy X (features) and y (target) data for unit tests."""
    X = pd.DataFrame({
        "A": np.random.rand(100),
        "B": np.random.rand(100) * 2,
        "C": np.random.rand(100) * 0.5,
        "D_irrelevant": np.random.rand(100) 
    })
    y = 3 * X["A"] - 2 * X["B"] + 0.5 * X["C"] + np.random.normal(0, 0.1, 100)
    return X, y

@pytest.fixture(scope="function")
def minimal_model_test_config(tmp_path):
    """Creates a minimal, valid config dictionary for model testing using temporary paths."""
    mock_features_csv = tmp_path / "processed" / "features.csv"
    mock_target_source_csv = tmp_path / "processed" / "target_source_data.csv"
    mock_model_output_dir = tmp_path / "models"
    mock_model_pkl = mock_model_output_dir / "test_linear_regression.pkl"
    mock_selected_features_json = mock_model_output_dir / "test_linear_regression_selected_features.json"
    mock_metrics_json = mock_model_output_dir / "test_metrics.json"
    mock_log_file = tmp_path / "logs" / "test_model.log"

    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    return {
        "data_source": {"processed_path": str(mock_target_source_csv)},
        "artifacts": {
            "processed_dir": str(tmp_path / "processed"),
            "engineered_features_filename": "features.csv",
            "metrics_path": str(mock_metrics_json),
        },
        "model": {
            "active": "linear_regression",
            "linear_regression": {
                "save_path": str(mock_model_pkl),
                "selected_features_path": str(mock_selected_features_json),
                "stepwise": {"enabled": True, "threshold_in": 0.1, "threshold_out": 0.15, "verbose": False}
            }
        },
        "target": "mock_target_column",
        "data_split": {"test_size": 0.25, "random_state": 42},
        "logging": {"log_file": str(mock_log_file), "level": "DEBUG"}
    }

def test_validate_modeling_config_raises_keyerror(minimal_model_test_config):
    """Test validate_modeling_config raises KeyError for missing top-level keys."""
    config = minimal_model_test_config.copy()
    # Test by removing an essential top-level key
    del config["data_source"] 
    with pytest.raises(KeyError) as excinfo:
        model.validate_modeling_config(config) # Uses module-level logger in model.py
    assert "Missing required top-level config key: 'data_source'" in str(excinfo.value)

    config_missing_model = minimal_model_test_config.copy()
    del config_missing_model["model"]
    with pytest.raises(KeyError) as excinfo:
        model.validate_modeling_config(config_missing_model)
    assert "Missing required top-level config key: 'model'" in str(excinfo.value)


def test_stepwise_selection_selects_relevant_features(dummy_data_for_model_tests):
    """Test stepwise_selection identifies features with strong signals."""
    X, y = dummy_data_for_model_tests
    
    with patch.object(model, 'logger', MagicMock()): # Patch logger for isolated unit test
        selected_features = model.stepwise_selection(X, y, threshold_in=0.05, threshold_out=0.1, verbose=False)
    
    assert isinstance(selected_features, list)
    assert len(selected_features) > 0
    # Check for features known to be in the generated dummy data
    assert "A" in selected_features
    assert "B" in selected_features
    assert "C" in selected_features
    test_setup_logger_model.info(f"Stepwise selected in test: {selected_features}")


def test_train_linear_regression_returns_model_and_features(dummy_data_for_model_tests, minimal_model_test_config):
    """Test train_linear_regression returns a fitted model and selected feature list."""
    X, y = dummy_data_for_model_tests
    lr_config = minimal_model_test_config["model"]["linear_regression"] 

    with patch.object(model, 'logger', MagicMock()): # Patch logger
        trained_model, selected_features = model.train_linear_regression(X, y, lr_config)
    
    assert isinstance(trained_model, sm.regression.linear_model.RegressionResultsWrapper)
    assert isinstance(selected_features, list)
    assert len(selected_features) > 0
    assert all(feature in X.columns for feature in selected_features)


def test_main_modeling_e2e(tmp_path, minimal_model_test_config, dummy_data_for_model_tests):
    """End-to-end test for the main_modeling function, checking artifact creation."""
    
    test_config = minimal_model_test_config 
    X_dummy, y_dummy = dummy_data_for_model_tests
    
    # Create mock input CSV files in the temporary directory
    features_csv_path = Path(test_config["artifacts"]["processed_dir"]) / test_config["artifacts"]["engineered_features_filename"]
    X_dummy.to_csv(features_csv_path, index=False)

    target_source_csv_path = Path(test_config["data_source"]["processed_path"])
    df_for_target_source = X_dummy.copy() 
    df_for_target_source[test_config["target"]] = y_dummy 
    df_for_target_source.to_csv(target_source_csv_path, index=False)

    # Create the temporary config.yaml file
    config_yaml_path = tmp_path / "test_model_config.yaml"
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(test_config, f)

    # Run the main modeling function; it will initialize its own logger
    model.main_modeling(config_path=str(config_yaml_path))

    # Assert that output artifacts are created at paths defined in test_config
    model_output_path = test_config["model"]["linear_regression"]["save_path"]
    selected_features_output_path = test_config["model"]["linear_regression"]["selected_features_path"]
    metrics_output_path = test_config["artifacts"]["metrics_path"]

    assert os.path.exists(model_output_path), "Model .pkl file was not created."
    assert os.path.exists(selected_features_output_path), "Selected features .json file was not created."
    assert os.path.exists(metrics_output_path), "Metrics .json file was not created."

    # Basic content checks for JSON artifacts
    with open(selected_features_output_path, 'r') as f:
        sel_features_data = json.load(f)
    assert "selected_features" in sel_features_data and isinstance(sel_features_data["selected_features"], list)
    assert sel_features_data["count"] == len(sel_features_data["selected_features"]) and len(sel_features_data["selected_features"]) > 0

    with open(metrics_output_path, 'r') as f:
        metrics_data = json.load(f)
    assert all(k in metrics_data for k in ["mse", "r2", "adj_r2", "rmse", "mape"]) # Check for all expected keys
    assert isinstance(metrics_data["r2"], float)

    # Check if the pickled model can be loaded
    try:
        with open(model_output_path, 'rb') as f:
            loaded_model = pickle.load(f)
        assert isinstance(loaded_model, sm.regression.linear_model.RegressionResultsWrapper)
    except Exception as e:
        pytest.fail(f"Failed to load the pickled model: {e}")

    test_setup_logger_model.info("test_main_modeling_e2e passed successfully.")

