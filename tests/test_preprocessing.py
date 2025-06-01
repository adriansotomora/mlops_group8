import os
import pandas as pd
import numpy as np
import yaml
import joblib
import pytest
from unittest.mock import MagicMock
import logging

# Import the module to be tested
from src.preprocess import preprocessing 

def minimal_config_for_tests(tmpdir_path):
    """
    Creates a minimal but valid configuration dictionary for testing.
    Uses paths within the pytest-provided temporary directory (tmpdir_path)
    to ensure test isolation and automatic cleanup.
    """
    raw_data_file = tmpdir_path / "raw_test_data.csv"
    processed_data_file = tmpdir_path / "processed_test_data.csv"
    scaler_file = tmpdir_path / "scaler_test.pkl"
    log_file = tmpdir_path / "test_preprocessing.log"

    return {
        "data_source": {
            "raw_path": str(raw_data_file),
            "processed_path": str(processed_data_file),
            "type": "csv",
            "delimiter": ",",
            "header": 0,
            "encoding": "utf-8"
        },
        "preprocessing": {
            "drop_columns": ["dropme"],
            "outlier_removal": {
                "enabled": True,
                "features": ["A"], 
                "iqr_multiplier": 1.5
            },
            "scale": {
                "columns": ["A", "B"], 
                "method": "minmax"
            }
        },
        "artifacts": {
            "processed_dir": str(tmpdir_path), 
            "preprocessing_pipeline": str(scaler_file)
        },
        "logging": {
            "log_file": str(log_file),
            "level": "DEBUG" 
        }
    }

def test_validate_preprocessing_config_raises_for_missing_top_level_keys():
    """Test that KeyError is raised if essential top-level keys are missing from the config."""
    config_missing_top_keys = {"preprocessing": {}, "artifacts": {}} 
    
    with pytest.raises(KeyError) as excinfo:
        # Call the validation function with a config missing 'data_source' and 'logging'
        preprocessing.validate_preprocessing_config(config_missing_top_keys, logger_param=MagicMock())
    # Assert that the error message mentions one of the expected missing keys
    assert "Missing required top-level config key: 'data_source'" in str(excinfo.value) or \
           "Missing required top-level config key: 'logging'" in str(excinfo.value)


def test_validate_preprocessing_config_warns_missing_sub_keys(caplog, tmp_path):
    """Test that warnings are logged for missing optional sub-keys within the 'preprocessing' section."""
    config = minimal_config_for_tests(tmp_path)
    # Intentionally remove a sub-key to trigger the warning mechanism
    del config["preprocessing"]["drop_columns"] 
    
    mock_logger = MagicMock() # Use a mock logger to check its calls

    # Capture logs at WARNING level or higher
    with caplog.at_level(logging.WARNING): 
        preprocessing.validate_preprocessing_config(config, logger_param=mock_logger)

    # Verify that the mock_logger's warning method was called
    assert mock_logger.warning.called
    # Verify that the warning message content is as expected
    assert any("Missing 'drop_columns' in 'preprocessing' config" in call.args[0] for call in mock_logger.warning.call_args_list)
    

def test_drop_columns_removes_specified():
    """Test that specified columns are correctly dropped from the DataFrame."""
    df = pd.DataFrame({"A": [1, 2], "dropme": [3, 4], "keepme": [5, 6]})
    logger_mock = MagicMock()
    df_dropped = preprocessing.drop_columns(df.copy(), ["dropme"], logger_param=logger_mock)
    
    assert "dropme" not in df_dropped.columns
    assert "A" in df_dropped.columns
    assert "keepme" in df_dropped.columns
    # Check if the logger was called with the expected info message
    logger_mock.info.assert_any_call("Dropping columns: ['dropme']")


def test_remove_outliers_iqr_removes_rows():
    """Test that outlier removal using IQR correctly removes rows with outliers."""
    df = pd.DataFrame({"A": [1, 2, 100, 3, 4, np.nan], "B": [5, 6, 7, 8, 9, 10]}) # Column 'A' has an outlier (100)
    logger_mock = MagicMock()
    
    # Test outlier removal on column 'A'
    df_cleaned_A = preprocessing.remove_outliers_iqr(df.copy(), ["A"], 1.5, logger_param=logger_mock)
    assert df_cleaned_A["A"].max() < 100 # The outlier should be removed
    assert df_cleaned_A.shape[0] < df.dropna(subset=['A']).shape[0] # Number of rows should decrease
    
    # Test on column 'B' which has no outliers by this method's definition
    df_cleaned_B = preprocessing.remove_outliers_iqr(df.copy(), ["B"], 1.5, logger_param=logger_mock)
    assert df_cleaned_B.shape[0] == df.shape[0] # No rows should be removed for column B


def test_scale_columns_minmax():
    """Test MinMax scaling of specified numeric columns."""
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0], "C": ["x", "y", "z"]}) # 'C' is non-numeric
    logger_mock = MagicMock()
    df_scaled, scaler = preprocessing.scale_columns(df.copy(), ["A", "B"], "minmax", logger_param=logger_mock)
    
    assert scaler is not None # Scaler object should be returned
    assert "C" in df_scaled.columns # Non-scaled column should remain unchanged
    pd.testing.assert_series_equal(df_scaled["C"], df["C"])

    # Check if scaled columns are in the range [0, 1]
    np.testing.assert_almost_equal(df_scaled["A"].min(), 0.0)
    np.testing.assert_almost_equal(df_scaled["A"].max(), 1.0)
    np.testing.assert_almost_equal(df_scaled["B"].min(), 0.0)
    np.testing.assert_almost_equal(df_scaled["B"].max(), 1.0)
    # Corrected log message assertion
    logger_mock.info.assert_any_call("Scaling columns ['A', 'B'] with 'minmax'.")


def test_main_preprocessing_e2e(tmp_path):
    """End-to-end test for the main_preprocessing function."""
    # Define a small raw DataFrame for the test
    raw_df_dict = {
        "A": [1, 2, 3, 4, 100], # '100' is an outlier for column 'A'
        "B": [10, 20, 30, 40, 50],
        "dropme": [0, 0, 0, 0, 0], # This column should be dropped
        # Add other columns expected by the scale config to avoid issues
        'danceability': np.random.rand(5), 'energy': np.random.rand(5), 'loudness': np.random.rand(5),
        'speechiness': np.random.rand(5),'acousticness': np.random.rand(5),'liveness': np.random.rand(5),
        'valence': np.random.rand(5),'tempo': np.random.rand(5),'duration_ms': np.random.rand(5),
        'key': np.random.rand(5)
    }
    raw_df = pd.DataFrame(raw_df_dict)
    
    # Use the helper to get a config dictionary pointing to temporary paths
    test_config = minimal_config_for_tests(tmp_path)
    raw_csv_path_in_config = test_config["data_source"]["raw_path"]
    # Save the raw DataFrame to the temporary path specified in the test_config
    raw_df.to_csv(raw_csv_path_in_config, index=False)

    # Create a temporary config.yaml file for the main_preprocessing function to use
    config_yaml_path = tmp_path / "test_config.yaml"
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(test_config, f)

    # Execute the main preprocessing pipeline
    preprocessing.main_preprocessing(config_path=str(config_yaml_path))

    # --- Assertions for outputs ---
    # Check if the processed data file was created
    processed_output_path = test_config["data_source"]["processed_path"]
    assert os.path.exists(processed_output_path)
    df_proc = pd.read_csv(processed_output_path)

    # Check if specified columns were dropped
    assert "dropme" not in df_proc.columns
    
    # Check scaling and outlier removal effects on column 'A'
    if "A" in df_proc.columns and not df_proc["A"].empty:
        assert np.isclose(df_proc["A"].min(), 0.0, atol=1e-7) # Min should be 0 after scaling
        assert np.isclose(df_proc["A"].max(), 1.0, atol=1e-7) # Max should be 1 after scaling
    else:
        pytest.fail("Column 'A' is missing or empty in processed data.")

    # Check scaling effects on column 'B' (no outlier removal applied to B by this config)
    if "B" in df_proc.columns and not df_proc["B"].empty: 
        assert np.isclose(df_proc["B"].min(), 0.0, atol=1e-7)
        assert np.isclose(df_proc["B"].max(), 1.0, atol=1e-7)
    else:
        pytest.fail("Column 'B' is missing or empty in processed data.")

    # Check if the scaler artifact was saved
    scaler_output_path = test_config["artifacts"]["preprocessing_pipeline"]
    assert os.path.exists(scaler_output_path)
    scaler = joblib.load(scaler_output_path)
    
    # Verify the scaler can transform data of the correct shape
    num_scaled_cols = len(test_config["preprocessing"]["scale"]["columns"])
    dummy_scaler_input_data = np.random.rand(2, num_scaled_cols) 
    dummy_scaler_input_df = pd.DataFrame(dummy_scaler_input_data, columns=test_config["preprocessing"]["scale"]["columns"])
    try:
        transformed_arr = scaler.transform(dummy_scaler_input_df)
        assert transformed_arr.shape == (2, num_scaled_cols)
    except Exception as e:
        pytest.fail(f"Scaler failed to transform data: {e}")

