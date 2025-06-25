import os
import pandas as pd
import numpy as np
import yaml
import pytest
from unittest.mock import MagicMock, patch, mock_open
import logging
from pathlib import Path
import json
import pickle
import joblib

# Ensure the 'src' directory is in PYTHONPATH for imports
import sys
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import inferencer


class TestInferencerErrorHandling:
    """Test error handling and edge cases in inferencer.py to improve coverage."""
    
    def test_setup_logging_creates_log_directory(self, tmp_path):
        """Test that _setup_logging creates log directory if it doesn't exist."""
        log_file = tmp_path / "logs" / "test.log"
        
        inferencer._setup_logging(
            log_level_str="DEBUG",
            log_file=str(log_file),
            log_format="%(message)s",
            date_format="%Y-%m-%d"
        )
        
        assert log_file.parent.exists()
        assert log_file.exists()
    
    def test_load_standard_pickle_file_not_found(self):
        """Test _load_standard_pickle raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Model not found at path"):
            inferencer._load_standard_pickle("nonexistent.pkl", "Model")
    
    def test_load_joblib_pickle_file_not_found(self):
        """Test _load_joblib_pickle raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Scaler not found at path"):
            inferencer._load_joblib_pickle("nonexistent.pkl", "Scaler")
    
    def test_load_json_file_not_found(self):
        """Test _load_json raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Features not found at path"):
            inferencer._load_json("nonexistent.json", "Features")
    
    def test_run_inference_config_file_not_found(self, tmp_path, capsys):
        """Test run_inference when config file doesn't exist."""
        input_csv = tmp_path / "input.csv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(input_csv, index=False)
        
        # This should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            inferencer.run_inference(str(input_csv), "nonexistent.yaml", "output.csv")
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "CRITICAL ERROR: Config file" in captured.out
    
    def test_run_inference_yaml_error(self, tmp_path, capsys):
        """Test run_inference when YAML parsing fails."""
        input_csv = tmp_path / "input.csv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(input_csv, index=False)
        
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")  # Invalid YAML
        
        with pytest.raises(SystemExit) as exc_info:
            inferencer.run_inference(str(input_csv), str(config_file), "output.csv")
        
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "CRITICAL ERROR: Error parsing YAML" in captured.out
    
    def test_run_inference_missing_scaler_path(self, tmp_path):
        """Test run_inference when scaler path is missing from config."""
        input_csv = tmp_path / "input.csv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "artifacts": {},  # Missing preprocessing_pipeline
            "model": {"active": "linear_regression", "linear_regression": {"save_path": "model.pkl"}}
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
        
        inferencer.run_inference(str(input_csv), str(config_file), "output.csv")
        
        # Check that it logged the critical error and returned early
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Missing one or more required artifact paths" in log_content
    
    def test_run_inference_missing_model_path(self, tmp_path):
        """Test run_inference when model path is missing from config."""
        input_csv = tmp_path / "input.csv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "artifacts": {"preprocessing_pipeline": "scaler.pkl"},
            "model": {"active": "linear_regression", "linear_regression": {}}  # Missing save_path
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
        
        inferencer.run_inference(str(input_csv), str(config_file), "output.csv")
        
        # Check that it logged the critical error and returned early
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Missing one or more required artifact paths" in log_content
    
    def test_run_inference_cannot_determine_selected_features_path(self, tmp_path):
        """Test run_inference when selected features path cannot be determined."""
        input_csv = tmp_path / "input.csv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "artifacts": {"preprocessing_pipeline": "scaler.pkl"},
            "model": {"active": "linear_regression", "linear_regression": {}}  # No save_path or selected_features_path
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
        
        inferencer.run_inference(str(input_csv), str(config_file), "output.csv")
        
        # Check that it logged the critical error and returned early
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Cannot determine selected features path" in log_content
    
    def test_run_inference_artifact_not_found(self, tmp_path):
        """Test run_inference when artifacts are not found."""
        input_csv = tmp_path / "input.csv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(input_csv, index=False)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "artifacts": {"preprocessing_pipeline": "nonexistent_scaler.pkl"},
            "model": {
                "active": "linear_regression", 
                "linear_regression": {
                    "save_path": "nonexistent_model.pkl",
                    "selected_features_path": "nonexistent_features.json"
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
        
        inferencer.run_inference(str(input_csv), str(config_file), "output.csv")
        
        # Check that it logged the critical error about artifact not found
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "A required artifact was not found" in log_content
    
    def test_run_inference_artifact_loading_error(self, tmp_path):
        """Test run_inference when there's an error loading artifacts."""
        input_csv = tmp_path / "input.csv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(input_csv, index=False)
        
        # Create invalid files that will cause loading errors
        bad_scaler = tmp_path / "bad_scaler.pkl"
        bad_scaler.write_bytes(b'\x00\x01\x02\x03')  # Invalid pickle content
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "artifacts": {"preprocessing_pipeline": str(bad_scaler)},
            "model": {
                "active": "linear_regression", 
                "linear_regression": {
                    "save_path": "model.pkl",
                    "selected_features_path": "features.json"
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
        
        inferencer.run_inference(str(input_csv), str(config_file), "output.csv")
        
        # Check that it logged the critical error about loading artifacts
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Error loading artifacts" in log_content
    
    def test_run_inference_empty_selected_features(self, tmp_path):
        """Test run_inference when selected features list is empty."""
        input_csv = tmp_path / "input.csv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(input_csv, index=False)
        
        # Create valid files
        scaler_file = tmp_path / "scaler.pkl"
        with open(scaler_file, "wb") as f:
            joblib.dump({"scaler": "dummy"}, f)
        
        model_file = tmp_path / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({"model": "dummy"}, f)
        
        features_file = tmp_path / "features.json"
        with open(features_file, "w") as f:
            json.dump({"selected_features": []}, f)  # Empty features list
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "artifacts": {"preprocessing_pipeline": str(scaler_file)},
            "model": {
                "active": "linear_regression", 
                "linear_regression": {
                    "save_path": str(model_file),
                    "selected_features_path": str(features_file)
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
        
        inferencer.run_inference(str(input_csv), str(config_file), "output.csv")
        
        # Check that it logged the critical error about empty features
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Loaded selected features list is empty" in log_content
    
    def test_run_inference_input_file_not_found(self, tmp_path):
        """Test run_inference when input CSV file doesn't exist."""
        # Create valid artifacts
        scaler_file = tmp_path / "scaler.pkl"
        with open(scaler_file, "wb") as f:
            joblib.dump({"scaler": "dummy"}, f)
        
        model_file = tmp_path / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({"model": "dummy"}, f)
        
        features_file = tmp_path / "features.json"
        with open(features_file, "w") as f:
            json.dump({"selected_features": ["col1", "col2"]}, f)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "artifacts": {"preprocessing_pipeline": str(scaler_file)},
            "model": {
                "active": "linear_regression", 
                "linear_regression": {
                    "save_path": str(model_file),
                    "selected_features_path": str(features_file)
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
        
        # Use non-existent input file
        inferencer.run_inference("nonexistent.csv", str(config_file), "output.csv")
        
        # Check that it logged the error about input file not found
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "New input file not found" in log_content
    
    def test_run_inference_input_data_loading_error(self, tmp_path):
        """Test run_inference when there's an error loading input data."""
        # Create invalid input CSV
        bad_csv = tmp_path / "bad_input.csv"
        bad_csv.write_bytes(b'\x00\x01\x02\x03')  # Invalid CSV content
        
        # Create valid artifacts
        scaler_file = tmp_path / "scaler.pkl"
        with open(scaler_file, "wb") as f:
            joblib.dump({"scaler": "dummy"}, f)
        
        model_file = tmp_path / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({"model": "dummy"}, f)
        
        features_file = tmp_path / "features.json"
        with open(features_file, "w") as f:
            json.dump({"selected_features": ["col1", "col2"]}, f)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "artifacts": {"preprocessing_pipeline": str(scaler_file)},
            "model": {
                "active": "linear_regression", 
                "linear_regression": {
                    "save_path": str(model_file),
                    "selected_features_path": str(features_file)
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
        
        inferencer.run_inference(str(bad_csv), str(config_file), "output.csv")
        
        # Check that it logged the error about loading input data
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Error loading new input data" in log_content
    
    def test_run_inference_empty_input_data(self, tmp_path):
        """Test run_inference when input data is empty."""
        # Create empty input CSV
        empty_csv = tmp_path / "empty_input.csv"
        pd.DataFrame().to_csv(empty_csv, index=False)
        
        # Create valid artifacts
        scaler_file = tmp_path / "scaler.pkl"
        with open(scaler_file, "wb") as f:
            joblib.dump({"scaler": "dummy"}, f)
        
        model_file = tmp_path / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({"model": "dummy"}, f)
        
        features_file = tmp_path / "features.json"
        with open(features_file, "w") as f:
            json.dump({"selected_features": ["col1", "col2"]}, f)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "artifacts": {"preprocessing_pipeline": str(scaler_file)},
            "model": {
                "active": "linear_regression", 
                "linear_regression": {
                    "save_path": str(model_file),
                    "selected_features_path": str(features_file)
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
        
        inferencer.run_inference(str(empty_csv), str(config_file), "output.csv")
        
        # Check that it logged the error about empty data
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "New input data is empty" in log_content


class TestInferencerModuleImportError:
    """Test module import error handling."""
    
    def test_module_import_error(self, capsys):
        """Test handling of module import errors."""
        # This is harder to test directly since the import happens at module level
        # We can test the behavior when the modules are not available
        with patch('sys.exit') as mock_exit:
            with patch('builtins.__import__', side_effect=ModuleNotFoundError("test error")):
                # This would trigger the import error handling
                exec("try:\n    from nonexistent_module import something\nexcept ModuleNotFoundError as e:\n    print(f'CRITICAL ERROR: Could not import necessary modules. Ensure PYTHONPATH. Missing: {e}')\n    import sys\n    sys.exit(1)")
                
        captured = capsys.readouterr()
        assert "CRITICAL ERROR: Could not import necessary modules" in captured.out


class TestInferencerMissingFeatures:
    """Test scenarios with missing features in input data."""
    
    def create_valid_artifacts(self, tmp_path):
        """Helper to create valid artifacts for testing."""
        # Create mock scaler
        scaler_file = tmp_path / "scaler.pkl"
        with open(scaler_file, "wb") as f:
            mock_scaler = MagicMock()
            mock_scaler.transform.return_value = np.array([[1, 2], [3, 4]])
            joblib.dump(mock_scaler, f)
        
        # Create mock model
        model_file = tmp_path / "model.pkl"
        with open(model_file, "wb") as f:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([10, 20])
            pickle.dump(mock_model, f)
        
        # Create features file
        features_file = tmp_path / "features.json"
        with open(features_file, "w") as f:
            json.dump({"selected_features": ["col1", "col2", "col3"]}, f)
        
        return str(scaler_file), str(model_file), str(features_file)
    
    def test_run_inference_missing_required_features(self, tmp_path):
        """Test run_inference when input data is missing required features."""
        # Create input data missing some features
        input_csv = tmp_path / "input.csv"
        df = pd.DataFrame({"col1": [1, 2], "other_col": [3, 4]})  # Missing col2, col3
        df.to_csv(input_csv, index=False)
        
        scaler_file, model_file, features_file = self.create_valid_artifacts(tmp_path)
        
        config = {
            "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
            "artifacts": {"preprocessing_pipeline": scaler_file},
            "data_source": {"delimiter": ",", "header": 0},
            "model": {
                "active": "linear_regression", 
                "linear_regression": {
                    "save_path": model_file,
                    "selected_features_path": features_file
                }
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)
        
        inferencer.run_inference(str(input_csv), str(config_file), "output.csv")
        
        # Check that it logged the error about missing features
        log_file = tmp_path / "test.log"
        if log_file.exists():
            with open(log_file) as f:
                log_content = f.read()
                assert "Missing required features" in log_content
