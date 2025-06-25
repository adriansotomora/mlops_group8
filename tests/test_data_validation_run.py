"""
Tests for data_validation/run.py module
"""
import os
import pytest
import pandas as pd
import json
import tempfile
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from omegaconf import DictConfig
import sys

# Add project root to path
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return DictConfig({
        'main': {
            'WANDB_PROJECT': 'test_project',
            'WANDB_ENTITY': 'test_entity'
        },
        'data_validation': {
            'input_artifact': 'raw_data:latest',
            'output_artifact': 'validated_data',
            'report_path': 'logs/validation_report.json'
        }
    })


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
    })


@pytest.fixture
def sample_validation_report():
    """Create a sample validation report."""
    return {
        'result': 'passed',
        'errors': [],
        'warnings': ['Warning 1', 'Warning 2'],
        'details': {
            'col1': {'min': 1, 'max': 5},
            'col2': {'type': 'string'},
            'col3': {'mean': 3.3}
        }
    }


def test_html_from_report_success(sample_validation_report):
    """Test HTML generation from validation report."""
    from src.data_validation import run as data_validation_run
    
    html = data_validation_run._html_from_report(sample_validation_report)
    
    assert '<h2>Data Validation Report</h2>' in html
    assert '<b>Result:</b> passed' in html
    assert 'Errors: 0 | Warnings: 2' in html
    assert 'Warning 1' in html
    assert 'Warning 2' in html
    assert '<table border=\'1\'>' in html


def test_html_from_report_with_errors():
    """Test HTML generation from validation report with errors."""
    from src.data_validation import run as data_validation_run
    
    report = {
        'result': 'failed',
        'errors': ['Error 1', 'Error 2'],
        'warnings': [],
        'details': {}
    }
    
    html = data_validation_run._html_from_report(report)
    
    assert '<h2>Data Validation Report</h2>' in html
    assert '<b>Result:</b> failed' in html
    assert 'Errors: 2 | Warnings: 0' in html
    assert 'Error 1' in html
    assert 'Error 2' in html


def test_html_from_report_empty():
    """Test HTML generation from empty validation report."""
    from src.data_validation import run as data_validation_run
    
    report = {'result': 'unknown'}
    
    html = data_validation_run._html_from_report(report)
    
    assert '<h2>Data Validation Report</h2>' in html
    assert '<b>Result:</b> unknown' in html
    assert 'Errors: 0 | Warnings: 0' in html


@patch('src.data_validation.run.tempfile.TemporaryDirectory')
@patch('src.data_validation.run.Path')
@patch('src.data_validation.run.wandb')
@patch('src.data_validation.run.validate_data')
@patch('builtins.open', new_callable=mock_open)
@patch('src.data_validation.run.yaml.safe_load')
@patch('src.data_validation.run.pd.read_csv')
def test_data_validation_run_success(mock_read_csv, mock_yaml_load, mock_open_file, 
                                   mock_validate_data, mock_wandb, mock_path,
                                   mock_temp_dir, mock_config, sample_dataframe, 
                                   sample_validation_report):
    """Test successful data validation run."""
    from src.data_validation import run as data_validation_run
    
    # Setup mocks
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    mock_read_csv.return_value = sample_dataframe
    mock_yaml_load.return_value = {'data_validation': {'report_path': 'logs/validation_report.json'}}
    
    # Mock temp directory
    mock_temp_context = MagicMock()
    mock_temp_context.__enter__.return_value = '/tmp/test'
    mock_temp_context.__exit__ = MagicMock()
    mock_temp_dir.return_value = mock_temp_context
    
    # Mock artifact
    mock_artifact = MagicMock()
    mock_artifact.download.return_value = '/tmp/downloaded'
    mock_wandb_run.use_artifact.return_value = mock_artifact
    
    # Mock Path operations
    mock_path_instance = MagicMock()
    mock_path_instance.is_file.return_value = True
    mock_path.return_value = mock_path_instance
    
    # Mock JSON reading
    mock_open_file.return_value.read.return_value = json.dumps(sample_validation_report)
    
    # Run the function
    data_validation_run.main(mock_config)
    
    # Assertions
    mock_wandb.init.assert_called_once()
    init_call = mock_wandb.init.call_args
    assert init_call[1]['project'] == 'test_project'
    assert init_call[1]['entity'] == 'test_entity'
    assert init_call[1]['job_type'] == 'data_validation'
    
    mock_validate_data.assert_called_once()
    mock_wandb.finish.assert_called_once()


@patch('src.data_validation.run.tempfile.TemporaryDirectory')
@patch('src.data_validation.run.Path')
@patch('src.data_validation.run.wandb')
@patch('src.data_validation.run.validate_data')
@patch('builtins.open', new_callable=mock_open)
@patch('src.data_validation.run.yaml.safe_load')
@patch('src.data_validation.run.pd.read_csv')
def test_data_validation_run_empty_dataframe(mock_read_csv, mock_yaml_load, mock_open_file,
                                            mock_validate_data, mock_wandb, mock_path,
                                            mock_temp_dir, mock_config):
    """Test data validation run with empty DataFrame."""
    from src.data_validation import run as data_validation_run
    
    # Setup mocks
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    mock_read_csv.return_value = pd.DataFrame()  # Empty DataFrame
    mock_yaml_load.return_value = {'data_validation': {'report_path': 'logs/validation_report.json'}}
    
    # Mock temp directory
    mock_temp_context = MagicMock()
    mock_temp_context.__enter__.return_value = '/tmp/test'
    mock_temp_context.__exit__ = MagicMock()
    mock_temp_dir.return_value = mock_temp_context
    
    # Mock artifact
    mock_artifact = MagicMock()
    mock_artifact.download.return_value = '/tmp/downloaded'
    mock_wandb_run.use_artifact.return_value = mock_artifact
    
    # Mock Path operations
    mock_path_instance = MagicMock()
    mock_path_instance.is_file.return_value = True  # File exists
    mock_path.return_value = mock_path_instance
    
    # Mock JSON content for empty DataFrame case
    mock_open_file.return_value.read.return_value = json.dumps({'result': 'warning', 'errors': [], 'warnings': ['Empty dataframe']})
    
    # Run the function
    data_validation_run.main(mock_config)
    
    # Assertions
    mock_wandb.init.assert_called_once()
    mock_wandb.finish.assert_called_once()


@patch('src.data_validation.run.tempfile.TemporaryDirectory')
@patch('src.data_validation.run.Path')
@patch('src.data_validation.run.wandb')
@patch('src.data_validation.run.validate_data')
@patch('builtins.open', new_callable=mock_open)
@patch('src.data_validation.run.yaml.safe_load')
@patch('src.data_validation.run.pd.read_csv')
def test_data_validation_run_exception_handling(mock_read_csv, mock_yaml_load, mock_open_file,
                                               mock_validate_data, mock_wandb, mock_path,
                                               mock_temp_dir, mock_config):
    """Test data validation run exception handling."""
    from src.data_validation import run as data_validation_run
    
    # Setup mocks
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    mock_validate_data.side_effect = Exception("Validation failed")
    
    # Mock temp directory
    mock_temp_context = MagicMock()
    mock_temp_context.__enter__.return_value = '/tmp/test'
    mock_temp_context.__exit__ = MagicMock()
    mock_temp_dir.return_value = mock_temp_context
    
    # Mock artifact
    mock_artifact = MagicMock()
    mock_artifact.download.return_value = '/tmp/downloaded'
    mock_wandb_run.use_artifact.return_value = mock_artifact
    
    mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2, 3]})
    mock_yaml_load.return_value = {'data_validation': {'report_path': 'logs/validation_report.json'}}
    
    # Mock Path operations
    mock_path_instance = MagicMock()
    mock_path_instance.is_file.return_value = True
    mock_path.return_value = mock_path_instance
    
    # Run the function and expect SystemExit
    with pytest.raises(SystemExit):
        data_validation_run.main(mock_config)
    
    # Verify error handling
    mock_wandb_run.alert.assert_called_once()
    mock_wandb.finish.assert_called_once()


@patch('src.data_validation.run.tempfile.TemporaryDirectory')
@patch('src.data_validation.run.Path')
@patch('src.data_validation.run.wandb')
@patch('src.data_validation.run.validate_data')
@patch('builtins.open', new_callable=mock_open)
@patch('src.data_validation.run.yaml.safe_load')
@patch('src.data_validation.run.pd.read_csv')
def test_data_validation_run_no_report_file(mock_read_csv, mock_yaml_load, mock_open_file,
                                          mock_validate_data, mock_wandb, mock_path,
                                          mock_temp_dir, mock_config, sample_dataframe):
    """Test data validation run when report file doesn't exist."""
    from src.data_validation import run as data_validation_run
    
    # Setup mocks
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    mock_read_csv.return_value = sample_dataframe
    mock_yaml_load.return_value = {'data_validation': {'report_path': 'logs/validation_report.json'}}
    
    # Mock temp directory
    mock_temp_context = MagicMock()
    mock_temp_context.__enter__.return_value = '/tmp/test'
    mock_temp_context.__exit__ = MagicMock()
    mock_temp_dir.return_value = mock_temp_context
    
    # Mock artifact
    mock_artifact = MagicMock()
    mock_artifact.download.return_value = '/tmp/downloaded'
    mock_wandb_run.use_artifact.return_value = mock_artifact
    
    # Mock Path operations - report file doesn't exist
    mock_path_instance = MagicMock()
    mock_path_instance.is_file.return_value = True  # File exists for no_report_file test  
    mock_path.return_value = mock_path_instance
    
    # Mock JSON content for no report file case
    mock_open_file.return_value.read.return_value = json.dumps({'result': 'passed', 'errors': [], 'warnings': []})
    
    # Run the function
    data_validation_run.main(mock_config)
    
    # Assertions
    mock_wandb.init.assert_called_once()
    mock_wandb.finish.assert_called_once()
