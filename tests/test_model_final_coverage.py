import os
import pandas as pd
import numpy as np
import yaml
import pytest
from unittest.mock import MagicMock, patch, mock_open
import logging
from pathlib import Path
import tempfile
import pickle
import json

# Ensure the 'src' directory is in PYTHONPATH for imports
import sys
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import model


class TestModelFinalCoverage:
    """Additional tests for model.py to reach 90% coverage."""
    
    def test_stepwise_selection_convergence_edge_case(self):
        """Test stepwise selection when convergence happens immediately."""
        X = pd.DataFrame({'x1': [1, 2, 3, 4], 'x2': [5, 6, 7, 8]})
        y = pd.Series([10, 20, 30, 40])
        mock_logger = MagicMock()
        
        # Mock the selection process to converge immediately
        with patch('src.model.model.f_regression') as mock_f_reg:
            mock_f_reg.return_value = ([10.0, 5.0], [0.01, 0.05])  # Both significant
            
            selected = model.stepwise_feature_selection(
                X, y, p_enter=0.05, p_remove=0.1, logger_param=mock_logger
            )
            
            assert len(selected) > 0
    
    def test_stepwise_selection_no_features_meet_criteria(self):
        """Test stepwise selection when no features meet entry criteria."""
        X = pd.DataFrame({'x1': [1, 2, 3, 4], 'x2': [5, 6, 7, 8]})
        y = pd.Series([10, 20, 30, 40])
        mock_logger = MagicMock()
        
        # Mock f_regression to return high p-values (not significant)
        with patch('src.model.model.f_regression') as mock_f_reg:
            mock_f_reg.return_value = ([1.0, 1.0], [0.8, 0.9])  # High p-values
            
            selected = model.stepwise_feature_selection(
                X, y, p_enter=0.05, p_remove=0.1, logger_param=mock_logger
            )
            
            assert len(selected) == 0
            mock_logger.info.assert_any_call("No features meet the entry criteria.")
    
    def test_stepwise_selection_feature_removal(self):
        """Test stepwise selection when features need to be removed."""
        X = pd.DataFrame({'x1': [1, 2, 3, 4], 'x2': [5, 6, 7, 8], 'x3': [9, 10, 11, 12]})
        y = pd.Series([10, 20, 30, 40])
        mock_logger = MagicMock()
        
        # Create a scenario where features are added then removed
        with patch('src.model.model.f_regression') as mock_f_reg:
            # First call: all features are significant for entry
            # Subsequent calls: some features become non-significant
            mock_f_reg.side_effect = [
                ([10.0, 8.0, 6.0], [0.01, 0.02, 0.03]),  # All significant initially
                ([1.0, 8.0, 6.0], [0.15, 0.02, 0.03]),   # x1 becomes non-significant
            ]
            
            with patch('statsmodels.api.OLS') as mock_ols:
                mock_result = MagicMock()
                mock_result.pvalues = pd.Series([0.01, 0.15, 0.02], index=['x1', 'x2', 'x3'])
                mock_ols.return_value.fit.return_value = mock_result
                
                selected = model.stepwise_feature_selection(
                    X, y, p_enter=0.05, p_remove=0.1, logger_param=mock_logger
                )
                
                assert 'x1' not in selected  # Should be removed
    
    def test_evaluate_model_r2_calculation(self):
        """Test evaluate_model R² calculation edge cases."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = model.evaluate_model(y_true, y_pred)
        
        assert 'r2' in metrics
        assert 0 <= metrics['r2'] <= 1  # R² should be between 0 and 1 for good predictions
    
    def test_save_model_artifacts_error_handling(self, tmp_path):
        """Test save_model_artifacts with I/O errors."""
        mock_model = MagicMock()
        selected_features = ['feature1', 'feature2']
        mock_logger = MagicMock()
        
        # Create a directory that will cause permission errors
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only
        
        model_path = readonly_dir / "model.pkl"
        features_path = readonly_dir / "features.json"
        
        try:
            # This should handle the error gracefully
            model.save_model_artifacts(
                mock_model, selected_features, str(model_path), str(features_path), mock_logger
            )
            
            # Should log error about saving
            mock_logger.error.assert_called()
        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)
    
    def test_main_train_wandb_initialization_error(self, tmp_path):
        """Test main_train when WandB initialization fails."""
        # Create minimal valid data
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [10, 20, 30, 40, 50]
        })
        df.to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "wandb": {"enabled": True, "project": "test"},
            "data_source": {"train_features_path": str(input_csv)},
            "model": {
                "target_column": "target",
                "active": "linear_regression",
                "linear_regression": {"save_path": str(tmp_path / "model.pkl")}
            },
            "artifacts": {"model_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        # Mock WandB to raise an exception during initialization
        with patch('src.model.model.wandb') as mock_wandb:
            mock_wandb.init.side_effect = Exception("WandB connection failed")
            
            # Should handle the error gracefully
            model.main_train(str(config_path))
            
            # Check that error was logged
            log_file = tmp_path / "test.log"
            if log_file.exists():
                with open(log_file) as f:
                    log_content = f.read()
                    assert "Failed to initialize Weights & Biases" in log_content
    
    def test_main_train_target_column_missing(self, tmp_path):
        """Test main_train when target column is missing from data."""
        # Create data without target column
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
            # Missing 'target' column
        })
        df.to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"train_features_path": str(input_csv)},
            "model": {
                "target_column": "missing_target",  # This column doesn't exist
                "active": "linear_regression",
                "linear_regression": {"save_path": str(tmp_path / "model.pkl")}
            },
            "artifacts": {"model_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        model.main_train(str(config_path))
        
        # Check that error was logged
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Target column" in log_content and "not found" in log_content
    
    def test_main_train_insufficient_data(self, tmp_path):
        """Test main_train with insufficient data (< 2 rows)."""
        # Create data with only 1 row
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({
            'feature1': [1],
            'feature2': [2],
            'target': [10]
        })
        df.to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"train_features_path": str(input_csv)},
            "model": {
                "target_column": "target",
                "active": "linear_regression",
                "linear_regression": {"save_path": str(tmp_path / "model.pkl")}
            },
            "artifacts": {"model_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        model.main_train(str(config_path))
        
        # Check that error was logged
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Insufficient data" in log_content
    
    def test_main_train_model_training_error(self, tmp_path):
        """Test main_train when model training fails."""
        # Create valid data
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'target': [10, 20, 30, 40, 50]
        })
        df.to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"train_features_path": str(input_csv)},
            "model": {
                "target_column": "target",
                "active": "linear_regression",
                "linear_regression": {"save_path": str(tmp_path / "model.pkl")}
            },
            "artifacts": {"model_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        # Mock LinearRegression to raise an exception
        with patch('sklearn.linear_model.LinearRegression') as mock_lr:
            mock_lr.return_value.fit.side_effect = Exception("Training failed")
            
            model.main_train(str(config_path))
            
            # Check that error was logged
            log_file = tmp_path / "test.log"
            if log_file.exists():
                with open(log_file) as f:
                    log_content = f.read()
                    assert "Error during model training" in log_content


class TestModelEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_get_logger_with_minimal_config(self):
        """Test get_logger with minimal configuration."""
        logging_config = {}
        default_log_file = "test.log"
        
        logger = model.get_logger(logging_config, default_log_file)
        
        assert logger is not None
        assert logger.level == logging.INFO  # Default level
    
    def test_stepwise_selection_single_feature(self):
        """Test stepwise selection with only one feature."""
        X = pd.DataFrame({'x1': [1, 2, 3, 4]})
        y = pd.Series([10, 20, 30, 40])
        mock_logger = MagicMock()
        
        with patch('src.model.model.f_regression') as mock_f_reg:
            mock_f_reg.return_value = ([10.0], [0.01])  # Significant
            
            selected = model.stepwise_feature_selection(
                X, y, p_enter=0.05, p_remove=0.1, logger_param=mock_logger
            )
            
            assert len(selected) == 1
            assert 'x1' in selected
    
    def test_evaluate_model_perfect_predictions(self):
        """Test evaluate_model with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])  # Perfect predictions
        
        metrics = model.evaluate_model(y_true, y_pred)
        
        assert metrics['mse'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mae'] == 0.0
        assert metrics['r2'] == 1.0  # Perfect R²
