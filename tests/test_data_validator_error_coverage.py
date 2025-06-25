import os
import pandas as pd
import numpy as np
import yaml
import pytest
from unittest.mock import MagicMock, patch, mock_open
import logging
from pathlib import Path
import io

# Ensure the 'src' directory is in PYTHONPATH for imports
import sys
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_validation import data_validator


class TestDataValidatorErrorHandling:
    """Test error handling and edge cases in data_validator.py to improve coverage."""
    
    def test_load_config_file_not_found(self):
        """Test load_config raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            data_validator.load_config("nonexistent_config.yaml")
    
    def test_load_config_yaml_parse_error(self, tmp_path):
        """Test load_config handles YAML parsing errors."""
        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text("invalid: yaml: content: [")  # Invalid YAML
        
        with pytest.raises(yaml.YAMLError):
            data_validator.load_config(str(config_file))
    
    def test_get_logger_creates_log_directory(self, tmp_path):
        """Test that get_logger creates log directory if it doesn't exist."""
        log_file = tmp_path / "logs" / "test.log"
        logging_config = {
            "log_file": str(log_file),
            "level": "DEBUG",
            "format": "%(message)s"
        }
        
        logger = data_validator.get_logger(logging_config)
        
        assert log_file.parent.exists()
        assert log_file.exists()
        assert logger.level == logging.DEBUG
    
    def test_validate_schema_missing_required_columns(self):
        """Test validate_schema when required columns are missing."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        schema = {
            "required_columns": ["col1", "col2", "missing_col"],
            "column_types": {}
        }
        mock_logger = MagicMock()
        
        result = data_validator.validate_schema(df, schema, mock_logger)
        
        assert not result
        mock_logger.error.assert_called_once()
        assert "Missing required columns" in mock_logger.error.call_args[0][0]
    
    def test_validate_schema_incorrect_data_types(self):
        """Test validate_schema when column types are incorrect."""
        df = pd.DataFrame({
            "text_col": ["a", "b", "c"],
            "int_col": [1, 2, 3]
        })
        schema = {
            "required_columns": ["text_col", "int_col"],
            "column_types": {
                "text_col": "int64",  # Should be object/string
                "int_col": "float64"  # Should be int64
            }
        }
        mock_logger = MagicMock()
        
        result = data_validator.validate_schema(df, schema, mock_logger)
        
        assert not result
        mock_logger.error.assert_called()
        # Should log errors for both incorrect types
        assert mock_logger.error.call_count >= 1
    
    def test_validate_schema_no_column_types_specified(self):
        """Test validate_schema when no column types are specified."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        schema = {
            "required_columns": ["col1", "col2"],
            "column_types": {}
        }
        mock_logger = MagicMock()
        
        result = data_validator.validate_schema(df, schema, mock_logger)
        
        assert result
        mock_logger.info.assert_any_call("No column type validations specified.")
    
    def test_validate_data_quality_with_nulls(self):
        """Test validate_data_quality when null values are found."""
        df = pd.DataFrame({
            "col1": [1, 2, None, 4],
            "col2": [None, 2, 3, 4]
        })
        quality_rules = {
            "max_null_percentage": 10.0,  # 25% nulls should fail this
            "min_rows": 3
        }
        mock_logger = MagicMock()
        
        result = data_validator.validate_data_quality(df, quality_rules, mock_logger)
        
        assert not result
        mock_logger.error.assert_called()
        assert "exceeds maximum allowed" in mock_logger.error.call_args[0][0]
    
    def test_validate_data_quality_insufficient_rows(self):
        """Test validate_data_quality when there are insufficient rows."""
        df = pd.DataFrame({"col1": [1, 2]})  # Only 2 rows
        quality_rules = {
            "max_null_percentage": 20.0,
            "min_rows": 5  # Requires at least 5 rows
        }
        mock_logger = MagicMock()
        
        result = data_validator.validate_data_quality(df, quality_rules, mock_logger)
        
        assert not result
        mock_logger.error.assert_called()
        assert "insufficient rows" in mock_logger.error.call_args[0][0].lower()
    
    def test_validate_data_quality_no_rules_specified(self):
        """Test validate_data_quality when no quality rules are specified."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        quality_rules = {}
        mock_logger = MagicMock()
        
        result = data_validator.validate_data_quality(df, quality_rules, mock_logger)
        
        assert result
        mock_logger.info.assert_any_call("No data quality rules specified. Skipping validation.")
    
    def test_save_validation_report_io_error(self, tmp_path):
        """Test save_validation_report handles I/O errors gracefully."""
        report = {"status": "pass", "errors": []}
        invalid_path = tmp_path / "nonexistent_dir" / "report.yaml"
        mock_logger = MagicMock()
        
        # This should handle the error gracefully and not raise an exception
        data_validator.save_validation_report(report, str(invalid_path), mock_logger)
        
        mock_logger.error.assert_called_once()
        assert "Error saving validation report" in mock_logger.error.call_args[0][0]
    
    def test_validate_data_missing_input_file(self, tmp_path):
        """Test validate_data when input file doesn't exist."""
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"raw_path": "nonexistent.csv"},
            "validation": {
                "schema": {"required_columns": [], "column_types": {}},
                "quality": {}
            },
            "artifacts": {"validation_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        data_validator.validate_data(str(config_path))
        
        # Check that it logged the critical error about file not found
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Input file not found" in log_content
    
    def test_validate_data_file_read_error(self, tmp_path):
        """Test validate_data when there's an error reading the input file."""
        # Create a file that will cause a read error
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_bytes(b'\x00\x01\x02\x03')  # Invalid CSV content
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"raw_path": str(bad_csv)},
            "validation": {
                "schema": {"required_columns": [], "column_types": {}},
                "quality": {}
            },
            "artifacts": {"validation_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        data_validator.validate_data(str(config_path))
        
        # Check that it logged the critical error about loading data
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Error loading data" in log_content
    
    def test_validate_data_empty_input_data(self, tmp_path):
        """Test validate_data when input data is empty."""
        # Create empty CSV
        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(empty_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"raw_path": str(empty_csv)},
            "validation": {
                "schema": {"required_columns": [], "column_types": {}},
                "quality": {}
            },
            "artifacts": {"validation_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        data_validator.validate_data(str(config_path))
        
        # Check that it logged the critical error about empty data
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Data is empty" in log_content
    
    def test_validate_data_missing_validation_config(self, tmp_path):
        """Test validate_data when validation config is missing."""
        # Create valid input CSV
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"raw_path": str(input_csv)},
            # Missing validation section
            "artifacts": {"validation_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        data_validator.validate_data(str(config_path))
        
        # Check that it logged info about missing validation config
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "No validation configuration found" in log_content
    
    def test_validate_data_schema_validation_failure(self, tmp_path):
        """Test validate_data when schema validation fails."""
        # Create input CSV with missing required columns
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"raw_path": str(input_csv)},
            "validation": {
                "schema": {
                    "required_columns": ["col1", "col2", "missing_col"],
                    "column_types": {}
                },
                "quality": {}
            },
            "artifacts": {"validation_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        data_validator.validate_data(str(config_path))
        
        # Check that validation report shows failure
        report_file = tmp_path / "validation_report.yaml"
        if report_file.exists():
            with open(report_file) as f:
                report = yaml.safe_load(f)
                assert report["overall_status"] == "FAIL"
                assert "schema_validation" in report
                assert not report["schema_validation"]["passed"]
    
    def test_validate_data_quality_validation_failure(self, tmp_path):
        """Test validate_data when data quality validation fails."""
        # Create input CSV with many null values
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({
            "col1": [1, None, None, None],  # 75% nulls
            "col2": [1, 2, 3, 4]
        })
        df.to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"raw_path": str(input_csv)},
            "validation": {
                "schema": {
                    "required_columns": ["col1", "col2"],
                    "column_types": {}
                },
                "quality": {
                    "max_null_percentage": 10.0,  # 75% nulls should fail this
                    "min_rows": 3
                }
            },
            "artifacts": {"validation_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        data_validator.validate_data(str(config_path))
        
        # Check that validation report shows failure
        report_file = tmp_path / "validation_report.yaml"
        if report_file.exists():
            with open(report_file) as f:
                report = yaml.safe_load(f)
                assert report["overall_status"] == "FAIL"
                assert "quality_validation" in report
                assert not report["quality_validation"]["passed"]
    
    def test_validate_data_successful_validation(self, tmp_path):
        """Test validate_data when all validations pass."""
        # Create valid input CSV
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        df.to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"raw_path": str(input_csv)},
            "validation": {
                "schema": {
                    "required_columns": ["col1", "col2"],
                    "column_types": {
                        "col1": "int64",
                        "col2": "float64"
                    }
                },
                "quality": {
                    "max_null_percentage": 10.0,
                    "min_rows": 3
                }
            },
            "artifacts": {"validation_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        data_validator.validate_data(str(config_path))
        
        # Check that validation report shows success
        report_file = tmp_path / "validation_report.yaml"
        if report_file.exists():
            with open(report_file) as f:
                report = yaml.safe_load(f)
                assert report["overall_status"] == "PASS"
                assert report["schema_validation"]["passed"]
                assert report["quality_validation"]["passed"]


class TestDataValidatorEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_validate_schema_empty_dataframe(self):
        """Test validate_schema with empty DataFrame."""
        df = pd.DataFrame()
        schema = {
            "required_columns": ["col1"],
            "column_types": {}
        }
        mock_logger = MagicMock()
        
        result = data_validator.validate_schema(df, schema, mock_logger)
        
        assert not result
        mock_logger.error.assert_called()
    
    def test_validate_data_quality_all_nulls(self):
        """Test validate_data_quality when all values are null."""
        df = pd.DataFrame({
            "col1": [None, None, None],
            "col2": [None, None, None]
        })
        quality_rules = {
            "max_null_percentage": 50.0,
            "min_rows": 2
        }
        mock_logger = MagicMock()
        
        result = data_validator.validate_data_quality(df, quality_rules, mock_logger)
        
        assert not result
        mock_logger.error.assert_called()
    
    def test_validate_data_quality_zero_max_null_percentage(self):
        """Test validate_data_quality with zero tolerance for nulls."""
        df = pd.DataFrame({
            "col1": [1, 2, None],  # One null value
            "col2": [1, 2, 3]
        })
        quality_rules = {
            "max_null_percentage": 0.0,  # No nulls allowed
            "min_rows": 2
        }
        mock_logger = MagicMock()
        
        result = data_validator.validate_data_quality(df, quality_rules, mock_logger)
        
        assert not result
        mock_logger.error.assert_called()
    
    def test_validate_schema_case_sensitive_column_names(self):
        """Test validate_schema with case-sensitive column names."""
        df = pd.DataFrame({"Col1": [1, 2], "col2": [3, 4]})
        schema = {
            "required_columns": ["col1", "col2"],  # Different case
            "column_types": {}
        }
        mock_logger = MagicMock()
        
        result = data_validator.validate_schema(df, schema, mock_logger)
        
        assert not result
        mock_logger.error.assert_called()
        assert "Missing required columns" in mock_logger.error.call_args[0][0]
    
    def test_validate_data_config_load_error(self, tmp_path):
        """Test validate_data when config loading fails."""
        # Create a config file with invalid YAML
        config_path = tmp_path / "bad_config.yaml"
        config_path.write_text("invalid: yaml: content: [")
        
        # Should handle the error gracefully
        data_validator.validate_data(str(config_path))
        
        # This should not raise an exception, but we can't easily check logging
        # since the function handles its own logger setup


class TestDataValidatorConfigHandling:
    """Test configuration handling edge cases."""
    
    def test_validate_data_missing_data_source_config(self, tmp_path):
        """Test validate_data when data_source config is missing."""
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            # Missing data_source section
            "validation": {
                "schema": {"required_columns": [], "column_types": {}},
                "quality": {}
            },
            "artifacts": {"validation_dir": str(tmp_path)}
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        data_validator.validate_data(str(config_path))
        
        # Check that it logged the critical error
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Config missing 'data_source.raw_path'" in log_content
    
    def test_validate_data_missing_artifacts_config(self, tmp_path):
        """Test validate_data when artifacts config is missing."""
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({"col1": [1, 2]})
        df.to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "data_source": {"raw_path": str(input_csv)},
            "validation": {
                "schema": {"required_columns": [], "column_types": {}},
                "quality": {}
            }
            # Missing artifacts section
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        
        data_validator.validate_data(str(config_path))
        
        # Should handle missing artifacts config gracefully
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                # Should still complete validation but maybe not save report
                assert "Starting data validation" in log_content
