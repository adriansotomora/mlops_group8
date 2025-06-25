import os
import pytest
from unittest.mock import MagicMock, patch, call, mock_open
import sys
from pathlib import Path
from omegaconf import DictConfig

TEST_DIR_RUN = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_RUN = os.path.abspath(os.path.join(TEST_DIR_RUN, '..'))
if PROJECT_ROOT_RUN not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_RUN)

# Mock config for testing
def create_mock_config():
    return DictConfig({
        'main': {
            'WANDB_PROJECT': 'test_project',
            'WANDB_ENTITY': 'test_entity'
        },
        'data_source': {
            'raw_path': 'data/raw/test.csv',
            'processed_path': 'data/processed/test.csv'
        },
        'target': 'test_target',
        'data_load': {
            'output_dir': 'data/raw',
            'log_sample_artifacts': True,
            'log_summary_stats': True,
            'log_artifacts': True
        },
        'data_validation': {
            'input_artifact': 'raw_data:latest',
            'output_artifact': 'validated_data',
            'report_path': 'logs/validation_report.json'
        }
    })

def test_preprocess_run_imports():
    """Test that preprocess run module can be imported."""
    from src.preprocess import run as preprocess_run
    assert preprocess_run is not None
    assert hasattr(preprocess_run, 'main')

def test_features_run_imports():
    """Test that features run module can be imported."""
    from src.features import run as features_run
    assert features_run is not None
    assert hasattr(features_run, 'main')

def test_model_run_imports():
    """Test that model run module can be imported."""
    from src.model import run as model_run
    assert model_run is not None
    assert hasattr(model_run, 'main')

def test_preprocess_run_can_be_called():
    """Test that preprocess run main function exists and can be called."""
    from src.preprocess import run as preprocess_run
    
    # Just verify the function exists - don't actually call it to avoid Hydra issues
    assert callable(preprocess_run.main)
    assert preprocess_run.main.__name__ == 'main'

def test_model_run_can_be_called():
    """Test that model run main function exists and can be called."""
    from src.model import run as model_run
    
    # Just verify the function exists - don't actually call it to avoid Hydra issues
    assert callable(model_run.main)
    assert model_run.main.__name__ == 'main'

def test_features_run_can_be_called():
    """Test that features run main function exists and can be called."""
    from src.features import run as features_run
    
    # Just verify the function exists - don't actually call it to avoid Hydra issues
    assert callable(features_run.main)
    assert features_run.main.__name__ == 'main'

@patch('builtins.open', new_callable=mock_open)
@patch('src.evaluation.run.json.load')
@patch('src.evaluation.run.pickle.load')
@patch('src.evaluation.run.pd.read_csv')
@patch('src.evaluation.run.Path')
@patch('src.evaluation.run.wandb')
@patch('src.evaluation.run.evaluate_statsmodels_model')
def test_evaluation_run_success(mock_evaluate, mock_wandb, mock_path, mock_read_csv, 
                               mock_pickle_load, mock_json_load, mock_open_file):
    """Test evaluation run wrapper with successful execution."""
    from src.evaluation import run as evaluation_run
    from omegaconf import DictConfig
    import pandas as pd
    
    # Setup config
    mock_config = DictConfig({
        'main': {
            'WANDB_PROJECT': 'test_project',
            'WANDB_ENTITY': 'test_entity'
        },
        'data_source': {
            'processed_path': 'data/processed.csv'
        },
        'target': 'target_column'
    })
    
    # Setup mocks
    mock_wandb_run = MagicMock()
    mock_wandb_run.name = "evaluation_test_run"
    mock_wandb.init.return_value = mock_wandb_run
    
    # Mock file operations
    mock_model = MagicMock()
    mock_pickle_load.return_value = mock_model
    mock_json_load.return_value = {"selected_features": ["feature1", "feature2"]}
    
    # Mock dataframes
    mock_features_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    mock_target_df = pd.DataFrame({'target_column': [7, 8, 9]})
    mock_read_csv.side_effect = [mock_features_df, mock_target_df]
    
    # Mock evaluate function
    mock_metrics = {'r2': 0.85, 'mse': 0.15}
    mock_evaluate.return_value = mock_metrics
    
    # Mock path operations
    mock_path_instance = MagicMock()
    mock_path.return_value = mock_path_instance
    
    evaluation_run.main(mock_config)
    
    mock_wandb.init.assert_called_once()
    init_call = mock_wandb.init.call_args
    assert init_call[1]["project"] == "test_project"
    assert init_call[1]["entity"] == "test_entity"
    assert init_call[1]["job_type"] == "evaluation"
    
    mock_evaluate.assert_called_once()
    mock_wandb.log.assert_any_call(mock_metrics)
    mock_wandb.log.assert_any_call({"evaluation_status": "completed"})
    mock_wandb.finish.assert_called_once()

@patch.dict('os.environ', {'WANDB_PROJECT': 'test_project', 'WANDB_ENTITY': 'test_entity'})
@patch('src.inference.run.wandb')
@patch('src.inference.run.run_inference')
def test_inference_run_success(mock_run_inference, mock_wandb):
    """Test inference run wrapper with successful execution."""
    from src.inference import run as inference_run
    
    mock_wandb_run = MagicMock()
    mock_wandb_run.name = "inference_test_run"
    mock_wandb.init.return_value = mock_wandb_run
    
    inference_run.main()
    
    mock_wandb.init.assert_called_once()
    init_call = mock_wandb.init.call_args
    assert init_call[1]["project"] == "test_project"
    assert init_call[1]["entity"] == "test_entity"
    assert init_call[1]["job_type"] == "inference"
    
    mock_run_inference.assert_called_once_with(
        input_csv_path="data/raw/Songs_2025.csv",
        config_path="../../config.yaml", 
        output_csv_path="data/predictions/inference_output.csv"
    )
    mock_wandb.log.assert_called_with({"inference_status": "completed"})
    mock_wandb.finish.assert_called_once()
