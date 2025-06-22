import os
import pytest
from unittest.mock import MagicMock, patch, call
import sys

TEST_DIR_RUN = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_RUN = os.path.abspath(os.path.join(TEST_DIR_RUN, '..'))
if PROJECT_ROOT_RUN not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_RUN)

@patch.dict('os.environ', {'WANDB_PROJECT': 'test_project', 'WANDB_ENTITY': 'test_entity'})
@patch('src.preprocess.run.wandb')
@patch('src.preprocess.run.main_preprocessing')
def test_preprocess_run_success(mock_main_preprocessing, mock_wandb):
    """Test preprocess run wrapper with successful execution."""
    from src.preprocess import run as preprocess_run
    
    mock_wandb_run = MagicMock()
    mock_wandb_run.name = "preprocess_test_run"
    mock_wandb.init.return_value = mock_wandb_run
    
    preprocess_run.main()
    
    mock_wandb.init.assert_called_once()
    init_call = mock_wandb.init.call_args
    assert init_call[1]["project"] == "test_project"
    assert init_call[1]["entity"] == "test_entity"
    assert init_call[1]["job_type"] == "preprocess"
    
    mock_main_preprocessing.assert_called_once_with(config_path="../../config.yaml")
    
    mock_wandb.log.assert_called_with({"preprocess_status": "completed"})
    mock_wandb.finish.assert_called_once()

@patch.dict('os.environ', {'WANDB_PROJECT': 'test_project', 'WANDB_ENTITY': 'test_entity'})
@patch('src.features.run.wandb')
@patch('src.features.run.main_features')
def test_features_run_success(mock_main_features, mock_wandb):
    """Test features run wrapper with successful execution."""
    from src.features import run as features_run
    
    mock_wandb_run = MagicMock()
    mock_wandb_run.name = "features_test_run"
    mock_wandb.init.return_value = mock_wandb_run
    
    features_run.main()
    
    mock_wandb.init.assert_called_once()
    mock_main_features.assert_called_once_with(config_path="../../config.yaml")
    mock_wandb.log.assert_called_with({"features_status": "completed"})
    mock_wandb.finish.assert_called_once()

@patch.dict('os.environ', {'WANDB_PROJECT': 'test_project', 'WANDB_ENTITY': 'test_entity'})
@patch('src.model.run.wandb')
@patch('src.model.run.main_modeling')
def test_model_run_with_metrics(mock_main_modeling, mock_wandb):
    """Test model run wrapper with metrics logging."""
    from src.model import run as model_run
    
    mock_wandb_run = MagicMock()
    mock_wandb_run.name = "model_test_run"
    mock_wandb.init.return_value = mock_wandb_run
    
    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = '{"mse": 0.1, "r2": 0.9}'
        with patch('json.load', return_value={"mse": 0.1, "r2": 0.9}):
            model_run.main()
    
    mock_wandb.init.assert_called_once()
    mock_main_modeling.assert_called_once_with(config_path="../../config.yaml")
    
    expected_calls = [
        call({"mse": 0.1, "r2": 0.9}),
        call({"model_status": "completed"})
    ]
    mock_wandb.log.assert_has_calls(expected_calls)
    mock_wandb.finish.assert_called_once()

@patch.dict('os.environ', {'WANDB_PROJECT': 'test_project', 'WANDB_ENTITY': 'test_entity'})
@patch('src.preprocess.run.wandb')
@patch('src.preprocess.run.main_preprocessing')
def test_preprocess_run_failure(mock_main_preprocessing, mock_wandb):
    """Test preprocess run wrapper with failure handling."""
    from src.preprocess import run as preprocess_run
    
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    
    mock_main_preprocessing.side_effect = Exception("Processing failed")
    
    with pytest.raises(Exception, match="Processing failed"):
        preprocess_run.main()
    
    mock_wandb.log.assert_called_with({
        "preprocess_status": "failed", 
        "error": "Processing failed"
    })
    mock_wandb.finish.assert_called_once()

@patch.dict('os.environ', {'WANDB_PROJECT': 'test_project', 'WANDB_ENTITY': 'test_entity'})
@patch('src.model.run.wandb')
@patch('src.model.run.main_modeling')
def test_model_run_failure(mock_main_modeling, mock_wandb):
    """Test model run wrapper with failure handling."""
    from src.model import run as model_run
    
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    
    mock_main_modeling.side_effect = Exception("Model training failed")
    
    with pytest.raises(Exception, match="Model training failed"):
        model_run.main()
    
    mock_wandb.log.assert_called_with({
        "model_status": "failed", 
        "error": "Model training failed"
    })
    mock_wandb.finish.assert_called_once()

@patch.dict('os.environ', {'WANDB_PROJECT': 'test_project', 'WANDB_ENTITY': 'test_entity'})
@patch('src.features.run.wandb')
@patch('src.features.run.main_features')
def test_features_run_failure(mock_main_features, mock_wandb):
    """Test features run wrapper with failure handling."""
    from src.features import run as features_run
    
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    
    mock_main_features.side_effect = Exception("Feature engineering failed")
    
    with pytest.raises(Exception, match="Feature engineering failed"):
        features_run.main()
    
    mock_wandb.log.assert_called_with({
        "features_status": "failed", 
        "error": "Feature engineering failed"
    })
    mock_wandb.finish.assert_called_once()

@patch.dict('os.environ', {'WANDB_PROJECT': 'test_project', 'WANDB_ENTITY': 'test_entity'})
@patch('src.evaluation.run.wandb')
@patch('src.evaluation.run.evaluate_statsmodels_model')
def test_evaluation_run_success(mock_evaluate, mock_wandb):
    """Test evaluation run wrapper with successful execution."""
    from src.evaluation import run as evaluation_run
    
    mock_wandb_run = MagicMock()
    mock_wandb_run.name = "evaluation_test_run"
    mock_wandb.init.return_value = mock_wandb_run
    
    evaluation_run.main()
    
    mock_wandb.init.assert_called_once()
    init_call = mock_wandb.init.call_args
    assert init_call[1]["project"] == "test_project"
    assert init_call[1]["entity"] == "test_entity"
    assert init_call[1]["job_type"] == "evaluation"
    
    mock_wandb.log.assert_called_with({"evaluation_status": "completed"})
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
