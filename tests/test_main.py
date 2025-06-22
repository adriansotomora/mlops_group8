import os
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import tempfile
import sys

TEST_DIR_MAIN = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_MAIN = os.path.abspath(os.path.join(TEST_DIR_MAIN, '..'))
if PROJECT_ROOT_MAIN not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_MAIN)

from src import main
from omegaconf import DictConfig

@pytest.fixture
def mock_config():
    """Create a mock Hydra config for testing."""
    return DictConfig({
        "main": {
            "WANDB_PROJECT": "test_project",
            "WANDB_ENTITY": "test_entity", 
            "steps": "preprocess,features,model",
            "hydra_options": "--test-option"
        }
    })

@pytest.fixture
def mock_config_all_steps():
    """Create a mock config with steps='all'."""
    return DictConfig({
        "main": {
            "WANDB_PROJECT": "test_project",
            "WANDB_ENTITY": "test_entity",
            "steps": "all"
        }
    })

@patch('src.main.wandb')
@patch('src.main.mlflow')
@patch('src.main.hydra.utils.get_original_cwd')
def test_main_orchestrator_basic_flow(mock_get_cwd, mock_mlflow, mock_wandb, mock_config):
    """Test basic orchestration flow with mocked dependencies."""
    mock_get_cwd.return_value = "/fake/project/root"
    mock_wandb_run = MagicMock()
    mock_wandb_run.name = "test_run_123"
    mock_wandb.init.return_value = mock_wandb_run
    
    main.main(mock_config)
    
    assert os.environ.get("WANDB_PROJECT") == "test_project"
    assert os.environ.get("WANDB_ENTITY") == "test_entity"
    
    mock_wandb.init.assert_called_once()
    init_call = mock_wandb.init.call_args
    assert init_call[1]["project"] == "test_project"
    assert init_call[1]["entity"] == "test_entity"
    assert init_call[1]["job_type"] == "orchestrator"
    
    expected_steps = ["preprocess", "features", "model"]
    assert mock_mlflow.run.call_count == len(expected_steps)
    
    mock_wandb.finish.assert_called_once()

@patch('src.main.wandb')
@patch('src.main.mlflow')
@patch('src.main.hydra.utils.get_original_cwd')
def test_main_orchestrator_all_steps(mock_get_cwd, mock_mlflow, mock_wandb, mock_config_all_steps):
    """Test orchestration with steps='all'."""
    mock_get_cwd.return_value = "/fake/project/root"
    mock_wandb_run = MagicMock()
    mock_wandb_run.name = "test_run_all"
    mock_wandb.init.return_value = mock_wandb_run
    
    main.main(mock_config_all_steps)
    
    expected_steps = main.PIPELINE_STEPS
    assert mock_mlflow.run.call_count == len(expected_steps)

@patch('src.main.wandb')
@patch('src.main.mlflow')
@patch('src.main.hydra.utils.get_original_cwd')
def test_main_orchestrator_with_hydra_options(mock_get_cwd, mock_mlflow, mock_wandb, mock_config):
    """Test orchestration with hydra options for model step."""
    mock_get_cwd.return_value = "/fake/project/root"
    mock_wandb_run = MagicMock()
    mock_wandb_run.name = "test_run_hydra"
    mock_wandb.init.return_value = mock_wandb_run
    
    main.main(mock_config)
    
    model_calls = [call for call in mock_mlflow.run.call_args_list 
                   if 'model' in str(call)]
    assert len(model_calls) == 1
    
    model_call = model_calls[0]
    assert "hydra_options" in model_call[1]["parameters"]
    assert model_call[1]["parameters"]["hydra_options"] == "--test-option"

def test_step_parsing_comma_separated():
    """Test parsing of comma-separated steps."""
    steps_raw = "preprocess, features , model"
    active_steps = [s.strip() for s in steps_raw.split(",") if s.strip()]
    
    assert active_steps == ["preprocess", "features", "model"]

def test_step_parsing_all():
    """Test parsing of 'all' steps."""
    steps_raw = "all"
    active_steps = main.PIPELINE_STEPS if steps_raw == "all" else [s.strip() for s in steps_raw.split(",") if s.strip()]
    
    assert active_steps == main.PIPELINE_STEPS
    assert "preprocess" in active_steps
    assert "features" in active_steps
    assert "model" in active_steps
    assert "evaluation" in active_steps
    assert "inference" in active_steps

def test_pipeline_steps_constant():
    """Test that PIPELINE_STEPS constant is properly defined."""
    assert isinstance(main.PIPELINE_STEPS, list)
    assert len(main.PIPELINE_STEPS) > 0
    assert "preprocess" in main.PIPELINE_STEPS
    assert "features" in main.PIPELINE_STEPS
    assert "model" in main.PIPELINE_STEPS

def test_steps_with_overrides_constant():
    """Test that STEPS_WITH_OVERRIDES constant is properly defined."""
    assert isinstance(main.STEPS_WITH_OVERRIDES, set)
    assert "model" in main.STEPS_WITH_OVERRIDES
