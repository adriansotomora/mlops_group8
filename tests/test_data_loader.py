"""Test suite for the data_loader module."""

import os
import pytest
import pandas as pd
from unittest.mock import patch 
import yaml # For testing load_config directly if needed

# Ensure the 'src' directory is in PYTHONPATH for imports
import sys
TEST_DIR_DATALOADER = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DATALOADER = os.path.abspath(os.path.join(TEST_DIR_DATALOADER, '..'))
if PROJECT_ROOT_DATALOADER not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DATALOADER)

from src.data_load import data_loader 

# Define paths for test files, relative to this test file's location
MOCK_DATA_DIR = os.path.join(TEST_DIR_DATALOADER, "mock_data")
MOCK_CSV_VALID = os.path.join(MOCK_DATA_DIR, "mock_data_spotify.csv") 
MOCK_CSV_NONEXISTENT = os.path.join(MOCK_DATA_DIR, "nonexistent_file.csv")

# Minimal configuration dictionary for testing raw CSV loading
MOCK_RAW_CSV_CONFIG = {
    "data_source": {
        "raw_path": MOCK_CSV_VALID, 
        "type": "csv",
        "delimiter": ",",
        "header": 0,
        "encoding": "utf-8"
    }
}

def test_get_raw_data_csv_success(monkeypatch):
    """Test successful raw data loading from a CSV file."""
    monkeypatch.setattr(data_loader, "load_config", lambda config_path: MOCK_RAW_CSV_CONFIG)
    
    df = data_loader.get_raw_data(config_path="dummy_config.yaml") 
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    # Example: Verify shape if MOCK_CSV_VALID is consistent
    # mock_df_check = pd.read_csv(MOCK_CSV_VALID)
    # assert df.shape == mock_df_check.shape

def test_get_raw_data_missing_file(monkeypatch):
    """Test FileNotFoundError if the raw_path data file is missing."""
    config_with_missing_file = {
        "data_source": {
            "raw_path": MOCK_CSV_NONEXISTENT, 
            "type": "csv"
        }
    }
    monkeypatch.setattr(data_loader, "load_config", lambda config_path: config_with_missing_file)
    
    with pytest.raises(FileNotFoundError):
        data_loader.get_raw_data(config_path="dummy_config.yaml")

def test_get_raw_data_unsupported_type(monkeypatch):
    """Test ValueError for unsupported file types in config."""
    config_unsupported_type = {
        "data_source": {
            "raw_path": MOCK_CSV_VALID,
            "type": "parquet" # data_loader.py only supports 'csv'
        }
    }
    monkeypatch.setattr(data_loader, "load_config", lambda config_path: config_unsupported_type)
    
    with pytest.raises(ValueError, match="Unsupported file type: 'parquet'"):
        data_loader.get_raw_data(config_path="dummy_config.yaml")

def test_get_raw_data_no_raw_path_in_config(monkeypatch):
    """Test ValueError if 'data_source.raw_path' is missing in config."""
    config_no_raw_path = {
        "data_source": { 
            "type": "csv"
        }
    }
    monkeypatch.setattr(data_loader, "load_config", lambda config_path: config_no_raw_path)
    
    with pytest.raises(ValueError, match="Config must specify 'data_source.raw_path' for raw data."):
        data_loader.get_raw_data(config_path="dummy_config.yaml")

def test_get_raw_data_missing_data_source_section(monkeypatch):
    """Test ValueError if 'data_source' section is missing in config."""
    config_no_data_source = {} 
    monkeypatch.setattr(data_loader, "load_config", lambda config_path: config_no_data_source)
    
    with pytest.raises(ValueError, match="Config must specify 'data_source.raw_path' for raw data."):
        data_loader.get_raw_data(config_path="dummy_config.yaml")

def test_load_config_file_not_found():
    """Test FileNotFoundError by load_config if config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        data_loader.load_config(config_path="non_existent_config.yaml")

# Test for the internal _read_csv_data helper function
def test_internal_read_csv_data_success(tmp_path):
    """Test the _read_csv_data helper for successful read."""
    temp_csv_file = tmp_path / "temp_data.csv"
    sample_data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    sample_data.to_csv(temp_csv_file, index=False)
    
    df = data_loader._read_csv_data(path=str(temp_csv_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    pd.testing.assert_frame_equal(df, sample_data)

def test_internal_read_csv_data_file_not_found():
    """Test FileNotFoundError for the _read_csv_data helper."""
    with pytest.raises(FileNotFoundError):
        data_loader._read_csv_data(path="surely_this_file_does_not_exist.csv")

def test_internal_read_csv_data_no_path():
    """Test ValueError for _read_csv_data if no path is given."""
    with pytest.raises(ValueError, match="No valid data path specified for CSV loading."):
        data_loader._read_csv_data(path=None) # type: ignore
    with pytest.raises(ValueError, match="No valid data path specified for CSV loading."):
        data_loader._read_csv_data(path="")

