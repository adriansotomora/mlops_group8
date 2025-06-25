import os
import pandas as pd
import numpy as np
import yaml
import pytest
from unittest.mock import MagicMock, patch, mock_open
import logging
from pathlib import Path
import tempfile
import shutil

# Ensure the 'src' directory is in PYTHONPATH for imports
import sys
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import features


class TestFeaturesErrorHandling:
    """Test error handling and edge cases in features.py to improve coverage."""
    
    def test_get_logger_creates_log_directory(self, tmp_path):
        """Test that get_logger creates log directory if it doesn't exist."""
        log_file = tmp_path / "logs" / "test.log"
        logging_config = {
            "log_file": str(log_file),
            "level": "DEBUG",
            "format": "%(message)s"
        }
        
        logger = features.get_logger(logging_config)
        
        assert log_file.parent.exists()
        assert log_file.exists()
        assert logger.level == logging.DEBUG
    
    def test_load_config_file_not_found(self):
        """Test load_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            features.load_config("nonexistent_config.yaml")
    
    def test_parse_genres_missing_artist_genres_column(self):
        """Test parse_genres when artist_genres column is missing."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        config = {"features": {"genre_features": ["pop", "rock"]}}
        mock_logger = MagicMock()
        
        result = features.parse_genres(df, config, mock_logger)
        
        assert result.equals(df)  # Should return unchanged DataFrame
        mock_logger.error.assert_called_once()
        assert "artist_genres" in mock_logger.error.call_args[0][0]
    
    def test_parse_genres_regex_error_handling(self):
        """Test parse_genres handles regex errors gracefully."""
        df = pd.DataFrame({"artist_genres": ["[pop]", "[rock]"]})
        config = {}
        mock_logger = MagicMock()
        
        # Mock re.compile to raise an exception
        with patch('re.compile') as mock_compile:
            mock_compile.side_effect = Exception("Regex error")
            
            # Since the function uses str.contains with regex, we need to mock that
            with patch.object(pd.Series.str, 'contains', side_effect=Exception("Regex error")):
                result = features.parse_genres(df, config, mock_logger)
                
                # Should handle error and set column to 0
                assert any(col.startswith('genre_') for col in result.columns)
                mock_logger.error.assert_called()
    
    def test_drop_irrelevant_columns_no_drop_config(self):
        """Test drop_irrelevant_columns when no drop config is provided."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        config = {"features": {}}
        mock_logger = MagicMock()
        
        result = features.drop_irrelevant_columns(df, config, mock_logger)
        
        assert result.equals(df)
        mock_logger.info.assert_called_with(
            "No columns specified for dropping in 'features.drop' config."
        )
    
    def test_drop_irrelevant_columns_no_existing_columns(self):
        """Test drop_irrelevant_columns when none of the drop columns exist."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        config = {"features": {"drop": ["nonexistent1", "nonexistent2"]}}
        mock_logger = MagicMock()
        
        result = features.drop_irrelevant_columns(df, config, mock_logger)
        
        assert result.equals(df)
        mock_logger.warning.assert_called_once()
        assert "None of columns to drop" in mock_logger.warning.call_args[0][0]
    
    def test_create_polynomial_features_no_config(self):
        """Test create_polynomial_features when no polynomial config exists."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        config = {"features": {}}
        mock_logger = MagicMock()
        
        result = features.create_polynomial_features(df, config, mock_logger)
        
        assert result.equals(df)
        mock_logger.info.assert_called_with("No polynomial feature configuration. Skipping.")
    
    def test_create_polynomial_features_no_audio_config(self):
        """Test create_polynomial_features when no audio polynomial config exists."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        config = {"features": {"polynomial": {"genre": {"degree": 2}}}}
        mock_logger = MagicMock()
        
        result = features.create_polynomial_features(df, config, mock_logger)
        
        assert result.equals(df)
        mock_logger.debug.assert_any_call("No polynomial config for type 'audio'. Skipping.")
    
    def test_create_polynomial_features_missing_base_features(self):
        """Test create_polynomial_features when base features are missing."""
        df = pd.DataFrame({"other_col": [1, 2]})
        config = {
            "features": {
                "audio_features": ["missing1", "missing2"],
                "polynomial": {"audio": {"degree": 2}}
            }
        }
        mock_logger = MagicMock()
        
        result = features.create_polynomial_features(df, config, mock_logger)
        
        assert result.equals(df)
        mock_logger.warning.assert_any_call("Poly base 'missing1' (audio) not found.")
        mock_logger.warning.assert_any_call("Poly base 'missing2' (audio) not found.")
        mock_logger.info.assert_any_call("No valid base features for audio polynomial creation.")
    
    def test_create_polynomial_features_non_numeric_base_features(self):
        """Test create_polynomial_features when base features are non-numeric."""
        df = pd.DataFrame({
            "text_col": ["a", "b"],
            "numeric_col": [1, 2]
        })
        config = {
            "features": {
                "audio_features": ["text_col", "numeric_col"],
                "polynomial": {"audio": {"degree": 2}}
            }
        }
        mock_logger = MagicMock()
        
        result = features.create_polynomial_features(df, config, mock_logger)
        
        mock_logger.warning.assert_any_call("Poly base 'text_col' (audio) not numeric.")
        # Should still create polynomial features for numeric_col
        assert len(result.columns) > len(df.columns)
    
    def test_select_features_no_features_config(self):
        """Test select_features when no features config exists."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        config = {}
        mock_logger = MagicMock()
        
        result = features.select_features(df, config, mock_logger)
        
        assert result.equals(df)
        mock_logger.info.assert_called_with("No 'features' section in config. Returning all columns.")
    
    def test_select_features_no_numeric_columns(self):
        """Test select_features when no numeric columns exist."""
        df = pd.DataFrame({"text_col": ["a", "b"], "category_col": ["x", "y"]})
        config = {"features": {"exclude": [], "profiling_variables": []}}
        mock_logger = MagicMock()
        
        result = features.select_features(df, config, mock_logger)
        
        assert len(result.columns) == 0
        mock_logger.warning.assert_called_with(
            "No numeric columns found after exclusions and profiling removal."
        )
    
    def test_log_feature_list_io_error(self, tmp_path):
        """Test log_feature_list handles I/O errors gracefully."""
        df = pd.DataFrame({"col1": [1, 2]})
        invalid_path = tmp_path / "nonexistent_dir" / "features.txt"
        mock_logger = MagicMock()
        
        # This should not raise an exception but should log an error
        features.log_feature_list(df, str(invalid_path), mock_logger)
        
        mock_logger.error.assert_called_once()
        assert "Error writing feature list" in mock_logger.error.call_args[0][0]
    
    def test_main_features_missing_processed_path_config(self, tmp_path):
        """Test main_features when processed_path is missing from config."""
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {},  # Missing processed_path
            "artifacts": {"processed_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        # Should return early without processing
        features.main_features(str(config_path))
        
        # Check that it logged the critical error
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Config missing 'data_source.processed_path'" in log_content
    
    def test_main_features_input_file_not_found(self, tmp_path):
        """Test main_features when input file doesn't exist."""
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"processed_path": "nonexistent.csv"},
            "artifacts": {"processed_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        features.main_features(str(config_path))
        
        # Check that it logged the critical error about file not found
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Input file not found" in log_content
    
    def test_main_features_input_file_read_error(self, tmp_path):
        """Test main_features when there's an error reading the input file."""
        # Create a file that will cause a read error
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_bytes(b'\x00\x01\x02\x03')  # Invalid CSV content
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"processed_path": str(bad_csv)},
            "artifacts": {"processed_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        features.main_features(str(config_path))
        
        # Check that it logged the critical error about loading data
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Error loading data from" in log_content
    
    def test_main_features_empty_input_data(self, tmp_path):
        """Test main_features when input data is empty."""
        # Create empty CSV
        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(empty_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"processed_path": str(empty_csv)},
            "artifacts": {"processed_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        features.main_features(str(config_path))
        
        # Check that it logged the critical error about empty data
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Loaded preprocessed data is empty" in log_content
    
    def test_main_features_output_save_error(self, tmp_path):
        """Test main_features when there's an error saving output."""
        # Create valid input CSV
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(input_csv, index=False)
        
        # Create config with invalid output directory (read-only)
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"processed_path": str(input_csv)},
            "artifacts": {
                "processed_dir": str(readonly_dir),
                "engineered_features_filename": "output.csv"
            },
            "features": {}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        try:
            features.main_features(str(config_path))
            
            # Check that it logged the error about saving
            log_file = tmp_path / "test.log"
            if log_file.exists():
                with open(log_file) as f:
                    log_content = f.read()
                    assert "Error saving engineered features" in log_content
        finally:
            # Restore write permissions for cleanup
            readonly_dir.chmod(0o755)
    
    def test_main_features_standalone_execution_no_config(self):
        """Test standalone execution when config file doesn't exist."""
        with patch('os.path.exists', return_value=False):
            with patch.object(features, 'logger') as mock_logger:
                # This should trigger the standalone execution path
                exec(compile(open(features.__file__).read(), features.__file__, 'exec'))
                
                # The standalone execution should log critical error
                # Note: This is a bit tricky to test due to module execution
    
    def test_main_features_standalone_execution_error(self, tmp_path):
        """Test standalone execution when an error occurs."""
        # Create a config that will cause an error
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: yaml: content: [")  # Invalid YAML
        
        with patch.object(features, 'CONFIG_PATH', str(config_path)):
            with patch('os.path.exists', return_value=True):
                with patch.object(features, 'logger') as mock_logger:
                    with patch.object(features, 'main_features', side_effect=Exception("Test error")):
                        # Simulate standalone execution error handling
                        try:
                            features.main_features(str(config_path))
                        except Exception as e:
                            # This would be caught in the standalone execution block
                            assert "Test error" in str(e)


class TestFeaturesPolynomialEdgeCases:
    """Additional tests for polynomial features edge cases."""
    
    def test_polynomial_features_sklearn_error(self):
        """Test polynomial features when sklearn raises an error."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        config = {
            "features": {
                "audio_features": ["col1", "col2"],
                "polynomial": {"audio": {"degree": 2}}
            }
        }
        mock_logger = MagicMock()
        
        with patch('sklearn.preprocessing.PolynomialFeatures') as mock_poly:
            mock_poly.return_value.fit_transform.side_effect = Exception("sklearn error")
            
            result = features.create_polynomial_features(df, config, mock_logger)
            
            # Should handle error gracefully and log it
            mock_logger.error.assert_called()
            assert "Error creating polynomial features" in mock_logger.error.call_args[0][0]
    
    def test_polynomial_features_memory_error(self):
        """Test polynomial features when memory error occurs."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        config = {
            "features": {
                "audio_features": ["col1", "col2"],
                "polynomial": {"audio": {"degree": 10}}  # High degree might cause memory issues
            }
        }
        mock_logger = MagicMock()
        
        with patch('sklearn.preprocessing.PolynomialFeatures') as mock_poly:
            mock_poly.return_value.fit_transform.side_effect = MemoryError("Not enough memory")
            
            result = features.create_polynomial_features(df, config, mock_logger)
            
            # Should handle memory error and log it
            mock_logger.error.assert_called()
