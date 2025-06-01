import os
import pandas as pd
import numpy as np
import yaml
import joblib
import pytest
from unittest.mock import MagicMock
import logging
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Ensure 'src' directory is in PYTHONPATH for imports
import sys
TEST_DIR_PREPROC = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PREPROC = os.path.abspath(os.path.join(TEST_DIR_PREPROC, '..'))
if PROJECT_ROOT_PREPROC not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_PREPROC)

# Import the module to be tested
from src.preprocess import preprocessing 

def minimal_config_for_tests(tmpdir_path: Path): 
    """Creates a minimal, valid config for testing preprocessing.py using temporary paths."""
    raw_data_file = tmpdir_path / "raw_test_data.csv"
    processed_data_file = tmpdir_path / "processed_test_data.csv"
    scaler_file = tmpdir_path / "scaler_test.pkl"
    log_file = tmpdir_path / "test_preprocessing.log"
    holdout_data_file = tmpdir_path / "holdout_test_data.csv"

    return {
        "data_source": {
            "raw_path": str(raw_data_file),
            "processed_path": str(processed_data_file),
            "type": "csv", 
            "delimiter": ",", "header": 0, "encoding": "utf-8",
            "inference_holdout_path": str(holdout_data_file), 
            "inference_holdout_size": 0.1 
        },
        "preprocessing": {
            "drop_columns": ["dropme"],
            "outlier_removal": {"enabled": True, "features": ["A"], "iqr_multiplier": 1.5},
            "scale": {"columns": ["A", "B"], "method": "minmax"}
        },
        "artifacts": {
            "processed_dir": str(tmpdir_path), 
            "preprocessing_pipeline": str(scaler_file)
        },
        "logging": {"log_file": str(log_file), "level": "DEBUG"},
        "data_split": {"random_state": 42} # For reproducible holdout split
    }

def test_validate_preprocessing_config_raises_for_missing_top_level_keys():
    """Test KeyError for missing essential top-level config keys."""
    config_missing = {"preprocessing": {}, "artifacts": {}} 
    with pytest.raises(KeyError) as excinfo:
        preprocessing.validate_preprocessing_config(config_missing, logger_param=MagicMock())
    assert "Missing required top-level config key" in str(excinfo.value)

def test_validate_preprocessing_config_warns_missing_sub_keys(caplog, tmp_path: Path):
    """Test warnings for missing optional sub-keys in 'preprocessing' config."""
    config = minimal_config_for_tests(tmp_path)
    del config["preprocessing"]["drop_columns"] 
    mock_logger = MagicMock() 
    with caplog.at_level(logging.WARNING): 
        preprocessing.validate_preprocessing_config(config, logger_param=mock_logger)
    assert mock_logger.warning.called
    assert any("Config: 'preprocessing.drop_columns' not found" in call.args[0] for call in mock_logger.warning.call_args_list)
    
def test_drop_columns_removes_specified():
    """Test correct removal of specified columns."""
    df = pd.DataFrame({"A": [1, 2], "dropme": [3, 4], "keepme": [5, 6]})
    logger_mock = MagicMock()
    df_dropped = preprocessing.drop_columns(df.copy(), ["dropme"], logger_param=logger_mock)
    assert "dropme" not in df_dropped.columns
    assert "A" in df_dropped.columns and "keepme" in df_dropped.columns
    logger_mock.info.assert_any_call("Dropping columns: ['dropme']")

def test_remove_outliers_iqr_removes_rows():
    """Test IQR outlier removal correctly removes rows."""
    df = pd.DataFrame({"A": [1, 2, 100, 3, 4, np.nan], "B": [5, 6, 7, 8, 9, 10]})
    logger_mock = MagicMock()
    df_cleaned_A = preprocessing.remove_outliers_iqr(df.copy(), ["A"], 1.5, logger_param=logger_mock)
    assert df_cleaned_A["A"].max() < 100 
    assert df_cleaned_A.shape[0] < df.dropna(subset=['A']).shape[0] 
    df_cleaned_B = preprocessing.remove_outliers_iqr(df.copy(), ["B"], 1.5, logger_param=logger_mock)
    assert df_cleaned_B.shape[0] == df.shape[0]

def test_scale_columns_minmax():
    """Test MinMax scaling of numeric columns."""
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0], "C": ["x", "y", "z"]})
    logger_mock = MagicMock()
    df_scaled, scaler = preprocessing.scale_columns(df.copy(), ["A", "B"], "minmax", logger_param=logger_mock)
    assert scaler is not None
    assert "C" in df_scaled.columns 
    pd.testing.assert_series_equal(df_scaled["C"], df["C"])
    for col_name in ["A", "B"]:
        np.testing.assert_almost_equal(df_scaled[col_name].min(), 0.0)
        np.testing.assert_almost_equal(df_scaled[col_name].max(), 1.0)
    logger_mock.info.assert_any_call("Scaling columns ['A', 'B'] with 'minmax' method.")

def test_main_preprocessing_e2e(tmp_path: Path):
    """End-to-end test for main_preprocessing, including data loading via data_loader."""
    num_rows = 10 # Use enough rows for predictable holdout split
    raw_df_dict = {
        "A": list(range(1, num_rows)) + [100], 
        "B": list(range(10, 10 * num_rows, 10)) + [10 * num_rows + 10],
        "dropme": [0] * num_rows,
        'danceability': np.random.rand(num_rows), 'energy': np.random.rand(num_rows), 
        'loudness': np.random.rand(num_rows), 'speechiness': np.random.rand(num_rows),
        'acousticness': np.random.rand(num_rows),'liveness': np.random.rand(num_rows),
        'valence': np.random.rand(num_rows),'tempo': np.random.rand(num_rows),
        'duration_ms': np.random.rand(num_rows), 'key': np.random.rand(num_rows)
    }
    raw_df = pd.DataFrame(raw_df_dict)
    
    test_config = minimal_config_for_tests(tmp_path) 
    raw_df.to_csv(test_config["data_source"]["raw_path"], index=False)

    config_yaml_path = tmp_path / "test_config.yaml"
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(test_config, f)

    # Run main_preprocessing; it uses data_loader.get_raw_data internally
    preprocessing.main_preprocessing(config_path=str(config_yaml_path))

    # Assertions for outputs
    processed_output_path = Path(test_config["data_source"]["processed_path"])
    assert processed_output_path.exists(), "Processed data CSV not found."
    df_proc = pd.read_csv(processed_output_path)

    holdout_output_path = Path(test_config["data_source"]["inference_holdout_path"])
    assert holdout_output_path.exists(), "Inference holdout CSV not found."
    df_holdout = pd.read_csv(holdout_output_path)

    # Check shapes based on holdout split
    expected_holdout_rows = 1 # For 10 rows and 0.1 test_size
    expected_pipeline_rows_before_outliers = len(raw_df) - expected_holdout_rows
    assert len(df_holdout) == expected_holdout_rows, \
        f"Holdout rows: expected {expected_holdout_rows}, got {len(df_holdout)}"
    assert len(df_proc) <= expected_pipeline_rows_before_outliers

    assert "dropme" not in df_proc.columns
    if "A" in df_proc.columns and not df_proc["A"].empty: 
        assert np.isclose(df_proc["A"].min(), 0.0, atol=1e-7)
        assert np.isclose(df_proc["A"].max(), 1.0, atol=1e-7)
    
    scaler_output_path = Path(test_config["artifacts"]["preprocessing_pipeline"])
    assert scaler_output_path.exists(), "Scaler artifact not found."
    scaler = joblib.load(scaler_output_path)
    assert isinstance(scaler, (MinMaxScaler, StandardScaler))
