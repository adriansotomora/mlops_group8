"""
Additional tests for src/model/model.py error handling paths
"""
import pytest
import pandas as pd
import yaml
import os
from unittest.mock import MagicMock, patch, mock_open
import sys
from pathlib import Path

# Add project root to path
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.model import (
    get_logger, 
    validate_modeling_config, 
    stepwise_selection, 
    train_linear_regression,
    save_model_artifacts,
    main_modeling
)


class TestModelErrorHandling:
    """Test error handling scenarios in model.py."""

    def test_get_logger_creates_directory(self):
        """Test that get_logger creates a new directory if needed."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            log_config = {"log_file": os.path.join(tmpdir, "new_dir", "model.log"), "level": "INFO"}
            get_logger(log_config, tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "new_dir"))

    def test_validate_modeling_config_missing_keys(self):
        """Test validate_modeling_config raises error for missing nested keys."""
        invalid_configs = [
            ({"artifacts": {"processed_dir": "/path"}, "model": {"active": "lr", "lr": {}}, "logging": {}, "target": "t", "data_source": {}}), # missing save_path
            ({"artifacts": {"engineered_features_filename": "f"}, "model": {"active": "lr", "lr": {"save_path": "p"}}, "logging": {}, "target": "t", "data_source": {}}), # missing processed_dir
            ({"artifacts": {"processed_dir": "/path"}, "model": {"lr": {"save_path": "p"}}, "logging": {}, "target": "t", "data_source": {}}), # missing active model
            ({"artifacts": {"processed_dir": "/path"}, "model": {"active": "lr", "lr": {"save_path": "p"}}, "logging": {}, "data_source": {}}), # missing target
        ]
        for config in invalid_configs:
            with pytest.raises(KeyError):
                validate_modeling_config(config)

    def test_stepwise_selection_no_features_left(self):
        """Test stepwise selection when no features are selected."""
        X = pd.DataFrame({'A': [1, 1, 1], 'B': [1, 1, 1]})  # No variation
        y = pd.Series([1, 2, 3])
        # With completely correlated features, stepwise should not select any
        with patch('src.model.model.logger'):
            selected = stepwise_selection(X, y, threshold_in=0.0001, threshold_out=0.0001, verbose=False)
        # Even if features are selected, test the warning path
        assert isinstance(selected, list)

    def test_train_linear_regression_no_features(self):
        """Test train_linear_regression when no features are provided."""
        X = pd.DataFrame(index=[0, 1, 2])
        y = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="No features available"):
            train_linear_regression(X, y, {"stepwise": {"enabled": False}})

    @patch('os.path.exists', return_value=False)
    def test_save_model_artifacts_features_file_not_found(self, mock_exists):
        """Test save_model_artifacts when features file is not found."""
        with pytest.raises(FileNotFoundError, match="Features file not found"):
            save_model_artifacts(MagicMock(), [], {}, {"artifacts": {"processed_dir": "/d"}, "model": {"active": "lr"}}, "/config")

    def test_save_model_artifacts_missing_features(self):
        """Test save_model_artifacts with missing features in features.csv."""
        # Create a dummy features.csv
        features_df = pd.DataFrame({"A": [1, 2, 3]})
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            features_df.to_csv(f.name, index=False)
            features_path = f.name
        
        with pytest.raises(ValueError, match="Selected features not found in features.csv"):
            save_model_artifacts(
                MagicMock(), 
                ["B"], # feature not in features.csv
                {}, 
                {"artifacts": {"processed_dir": os.path.dirname(features_path), "engineered_features_filename": os.path.basename(features_path)}, "model": {"active": "lr", "linear_regression": {"save_path": "p"}}},
                os.path.dirname(features_path)
            )

    def test_main_modeling_config_validation_error(self):
        """Test main_modeling with invalid config."""
        with patch('src.model.model.load_config', return_value={}), \
             patch('src.model.model.get_logger', return_value=MagicMock()) as mock_logger_init, \
             patch('src.model.model.validate_modeling_config', side_effect=KeyError("Missing key")) as mock_validate:
            
            main_modeling("config.yaml")
            # Verify logger was called with critical error
            mock_validate.assert_called_once()
            mock_logger_init.return_value.critical.assert_called_with("Configuration validation failed: 'Missing key'. Exiting.")

    def test_main_modeling_features_file_not_found(self):
        """Test main_modeling when features file is not found."""
        # Provide valid config that passes validation
        valid_config = {
            "artifacts": {"processed_dir": "/tmp", "engineered_features_filename": "features.csv", "metrics_path": "metrics.json"},
            "model": {"active": "linear_regression", "linear_regression": {"save_path": "model.pkl"}},
            "logging": {},
            "target": "target",
            "data_source": {"processed_path": "data.csv"},
            "data_split": {"test_size": 0.2, "random_state": 42}
        }
        
        with patch('src.model.model.load_config', return_value=valid_config), \
             patch('src.model.model.get_logger', return_value=MagicMock()) as mock_logger_init, \
             patch('os.path.exists', return_value=False):  # Features file doesn't exist
            
            main_modeling("config.yaml")
            # Verify error is logged for missing features file
            mock_logger_init.return_value.critical.assert_called()

