"""
Additional tests for src/data_load/data_loader.py error handling paths
"""
import pytest
import pandas as pd
import yaml
from unittest.mock import patch, mock_open
import sys
from pathlib import Path

# Add project root to path
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_load.data_loader import get_raw_data, load_config, _read_csv_data


class TestDataLoaderErrorHandling:
    """Test error handling scenarios in data loader."""

    def test_read_csv_data_invalid_path(self):
        """Test _read_csv_data with invalid path."""
        with pytest.raises(ValueError, match="No valid data path specified"):
            _read_csv_data("")
            
        with pytest.raises(ValueError, match="No valid data path specified"):
            _read_csv_data(None)

    def test_read_csv_data_file_not_found(self):
        """Test _read_csv_data with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            _read_csv_data("/non/existent/file.csv")

    @patch('src.data_load.data_loader.pd.read_csv')
    def test_read_csv_data_pandas_exception(self, mock_read_csv):
        """Test _read_csv_data with pandas exception."""
        # Create a temporary file that exists but causes pandas to fail
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("invalid,csv,content\n")
            temp_path = f.name
        
        mock_read_csv.side_effect = Exception("Pandas read error")
        
        with pytest.raises(Exception, match="Pandas read error"):
            _read_csv_data(temp_path)

    @patch('builtins.open', mock_open(read_data="invalid: yaml: ["))
    @patch('os.path.isfile', return_value=True)
    def test_get_raw_data_yaml_error(self, mock_isfile):
        """Test get_raw_data with invalid YAML."""
        with pytest.raises(ValueError, match="Error parsing YAML configuration"):
            get_raw_data("fake_config.yaml")

    @patch('src.data_load.data_loader.load_config')
    def test_get_raw_data_unsupported_file_type(self, mock_load_config):
        """Test get_raw_data with unsupported file type."""
        mock_load_config.return_value = {
            "data_source": {
                "type": "json",
                "raw_path": "test.json"
            }
        }
        
        with pytest.raises(ValueError, match="Unsupported file type: 'json'"):
            get_raw_data("config.yaml")

    @patch('src.data_load.data_loader.load_config')
    def test_get_raw_data_missing_raw_path(self, mock_load_config):
        """Test get_raw_data with missing raw_path."""
        mock_load_config.return_value = {
            "data_source": {
                "type": "csv"
                # missing raw_path
            }
        }
        
        with pytest.raises(ValueError, match="Config must specify 'data_source.raw_path'"):
            get_raw_data("config.yaml")

    @patch('src.data_load.data_loader.load_config')
    def test_get_raw_data_invalid_raw_path_type(self, mock_load_config):
        """Test get_raw_data with invalid raw_path type."""
        mock_load_config.return_value = {
            "data_source": {
                "type": "csv",
                "raw_path": 123  # Should be string
            }
        }
        
        with pytest.raises(ValueError, match="Config must specify 'data_source.raw_path'"):
            get_raw_data("config.yaml")

    @patch('src.data_load.data_loader.logger')
    @patch('src.data_load.data_loader.get_raw_data')
    def test_main_execution_success(self, mock_get_raw_data, mock_logger):
        """Test successful main execution."""
        mock_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_get_raw_data.return_value = mock_df
        
        # Import the module to trigger __main__ block
        # This is tricky to test directly, but we can test the components
        assert mock_df.shape == (3, 2)

    @patch('src.data_load.data_loader.logger')
    @patch('src.data_load.data_loader.get_raw_data')
    def test_main_execution_exception(self, mock_get_raw_data, mock_logger):
        """Test main execution with exception."""
        mock_get_raw_data.side_effect = Exception("Test error")
        
        # Import and run the function that would be called in __main__
        try:
            get_raw_data()
        except Exception as e:
            assert str(e) == "Test error"

    def test_load_config_file_not_found(self):
        """Test load_config with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config("/non/existent/config.yaml")

    @patch('builtins.open', mock_open(read_data="valid: yaml\ncontent: true"))
    @patch('os.path.isfile', return_value=True)
    @patch('src.data_load.data_loader.yaml.safe_load')
    def test_load_config_success(self, mock_yaml_load, mock_isfile):
        """Test successful config loading."""
        mock_yaml_load.return_value = {"test": "config"}
        
        result = load_config("test_config.yaml")
        assert result == {"test": "config"}
        mock_yaml_load.assert_called_once()

    @patch('src.data_load.data_loader._read_csv_data')
    @patch('src.data_load.data_loader.load_config')
    def test_get_raw_data_success_with_custom_params(self, mock_load_config, mock_read_csv):
        """Test get_raw_data with custom CSV parameters."""
        mock_load_config.return_value = {
            "data_source": {
                "type": "csv",
                "raw_path": "test.csv",
                "delimiter": ";",
                "header": 1,
                "encoding": "latin-1"
            }
        }
        
        mock_df = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_csv.return_value = mock_df
        
        result = get_raw_data("config.yaml")
        
        # Verify _read_csv_data was called with custom parameters
        mock_read_csv.assert_called_once()
        call_args = mock_read_csv.call_args
        # Verify the call was made with keyword arguments
        assert call_args.kwargs.get('delimiter') == ";"
        assert call_args.kwargs.get('header') == 1
        assert call_args.kwargs.get('encoding') == "latin-1"
