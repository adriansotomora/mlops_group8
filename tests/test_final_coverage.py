import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path

# Ensure the 'src' directory is in PYTHONPATH for imports
import sys
import os
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_load import data_loader
from src.model import model


class TestDataLoaderFinalCoverage:
    """Additional tests for data_loader.py to reach 90% coverage."""

    def test_data_loader_io_error(self, tmp_path):
        """Test I/O error handling in the data loader."""
        invalid_path = tmp_path / "nonexistent.csv"
        mock_logger = MagicMock()

        # This should not raise an exception but should log an error
        with patch.object(data_loader, 'logger', mock_logger):
            data_loader.load_csv_data(str(invalid_path))

        mock_logger.error.assert_called()
        assert "Error loading data" in mock_logger.error.call_args[0][0]

    def test_data_loader_empty_data(self, tmp_path):
        """Test loading an empty CSV file."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        mock_logger = MagicMock()

        with patch.object(data_loader, 'logger', mock_logger):
            df = data_loader.load_csv_data(str(empty_csv))

        assert df is not None
        mock_logger.warning.assert_called()
        assert "Loaded data is empty" in mock_logger.warning.call_args[0][0]


class TestModelFinalCoverageAddition:
    """Additional tests for model.py beyond 90% coverage."""

    def test_model_training_with_invalid_data(self, tmp_path):
        """Test model training with invalid (non-numeric) data."""
        invalid_df = pd.DataFrame({
            'feature1': ['a', 'b', 'c'],
            'feature2': ['d', 'e', 'f'],
            'target': [10, 20, 30]
        })
        mock_logger = MagicMock()

        # Create config
        config = {
            "logger": mock_logger,
            "config": {
                "data_source": {"train_features_path": str(tmp_path / "dummy.csv")},
                "model": {"target_column": "target"}
            }
        }

        # This function must handle invalid input gracefully
        model.train_model(invalid_df, config)

        mock_logger.error.assert_called()
        assert "Model training failed" in mock_logger.error.call_args[0][0]
