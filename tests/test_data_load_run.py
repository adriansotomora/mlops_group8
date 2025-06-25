"""
Tests for data_load/run.py module
"""
import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from omegaconf import DictConfig
import tempfile
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
        'data_source': {
            'raw_path': 'data/raw/test.csv'
        },
        'data_load': {
            'output_dir': 'data/output',
            'log_sample_artifacts': True,
            'log_summary_stats': True,
            'log_artifacts': True
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


def test_data_load_run_success(mock_config, sample_dataframe):
    """Test successful data load run."""
    with patch('src.data_load.run.Path') as mock_path, \
         patch('src.data_load.run.PROJECT_ROOT') as mock_project_root, \
         patch('src.data_load.run.wandb') as mock_wandb, \
         patch('src.data_load.run.get_raw_data') as mock_get_raw_data:
        
        from src.data_load import run as data_load_run
    
        # Setup mocks
        mock_wandb_run = MagicMock()
        mock_wandb.init.return_value = mock_wandb_run
        mock_get_raw_data.return_value = sample_dataframe
        
        # Mock PROJECT_ROOT
        mock_project_root_instance = MagicMock()
        mock_project_root_instance.__truediv__ = MagicMock()
        mock_project_root_instance.mkdir = MagicMock()
        mock_project_root.__truediv__ = MagicMock(return_value=mock_project_root_instance)
        
        # Mock Path operations
        def side_effect(*args, **kwargs):
            mock_path_instance = MagicMock()
            mock_path_instance.mkdir = MagicMock()
            mock_path_instance.is_absolute.return_value = False
            mock_path_instance.is_file.return_value = True
            mock_path_instance.name = 'test.csv'
            mock_path_instance.__str__ = MagicMock(return_value='/mock/path/test.csv')
            mock_path_instance.__truediv__ = MagicMock(return_value=mock_path_instance)
            return mock_path_instance
        
        mock_path.side_effect = side_effect
        
        # Run the function
        data_load_run.main(mock_config)
        
        # Assertions
        mock_wandb.init.assert_called_once()
        init_call = mock_wandb.init.call_args
        assert init_call[1]['project'] == 'test_project'
        assert init_call[1]['entity'] == 'test_entity'
        assert init_call[1]['job_type'] == 'data_load'
        
        mock_get_raw_data.assert_called_once()
        mock_wandb.log.assert_called()
        mock_wandb.finish.assert_called_once()


@patch('src.data_load.run.Path')
@patch('src.data_load.run.PROJECT_ROOT')
@patch('src.data_load.run.wandb')
@patch('src.data_load.run.get_raw_data')
def test_data_load_run_empty_dataframe(mock_get_raw_data, mock_wandb, mock_project_root, mock_path, mock_config):
    """Test data load run with empty DataFrame."""
    from src.data_load import run as data_load_run
    
    # Setup mocks
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    mock_get_raw_data.return_value = pd.DataFrame()  # Empty DataFrame
    
    # Mock PROJECT_ROOT and its operations
    mock_project_root_instance = MagicMock()
    mock_project_root_instance.mkdir = MagicMock()
    mock_project_root_instance.__str__ = MagicMock(return_value='/mock/project/root')
    mock_project_root.__truediv__ = MagicMock(return_value=mock_project_root_instance)
    
    # Mock Path operations
    def mock_path_constructor(*args, **kwargs):
        mock_path_instance = MagicMock()
        mock_path_instance.mkdir = MagicMock()
        mock_path_instance.is_absolute.return_value = False
        mock_path_instance.is_file.return_value = True
        mock_path_instance.name = 'test.csv'
        mock_path_instance.__truediv__ = MagicMock(return_value=mock_path_instance)
        mock_path_instance.__str__ = MagicMock(return_value='/mock/path/test.csv')
        return mock_path_instance
    
    mock_path.side_effect = mock_path_constructor
    
    # Run the function
    data_load_run.main(mock_config)
    
    # Assertions
    mock_wandb.init.assert_called_once()
    mock_wandb.finish.assert_called_once()


@patch('src.data_load.run.Path')
@patch('src.data_load.run.PROJECT_ROOT')
@patch('src.data_load.run.wandb')
@patch('src.data_load.run.get_raw_data')
def test_data_load_run_with_duplicates(mock_get_raw_data, mock_wandb, mock_project_root, mock_path, mock_config):
    """Test data load run with duplicate rows."""
    from src.data_load import run as data_load_run
    
    # Create DataFrame with duplicates
    df_with_dups = pd.DataFrame({
        'col1': [1, 2, 2, 3],
        'col2': ['a', 'b', 'b', 'c']
    })
    
    # Setup mocks
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    mock_get_raw_data.return_value = df_with_dups
    
    # Mock PROJECT_ROOT
    mock_project_root_instance = MagicMock()
    mock_project_root_instance.mkdir = MagicMock()
    mock_project_root.__truediv__ = MagicMock(return_value=mock_project_root_instance)
    
    # Mock Path operations
    mock_path_instance = MagicMock()
    mock_path_instance.mkdir = MagicMock()
    mock_path_instance.is_absolute.return_value = False
    mock_path_instance.is_file.return_value = True
    mock_path_instance.name = 'test.csv'
    mock_path.return_value = mock_path_instance
    
    # Run the function
    data_load_run.main(mock_config)
    
    # Assertions
    mock_wandb.init.assert_called_once()
    mock_wandb.finish.assert_called_once()


@patch('src.data_load.run.Path')
@patch('src.data_load.run.PROJECT_ROOT')
@patch('src.data_load.run.wandb')
@patch('src.data_load.run.get_raw_data')
def test_data_load_run_file_not_found(mock_get_raw_data, mock_wandb, mock_project_root, mock_path, mock_config):
    """Test data load run when file is not found."""
    from src.data_load import run as data_load_run
    
    # Mock PROJECT_ROOT
    mock_project_root_instance = MagicMock()
    mock_project_root_instance.mkdir = MagicMock()
    mock_project_root.__truediv__ = MagicMock(return_value=mock_project_root_instance)
    
    # Mock Path operations
    mock_path_instance = MagicMock()
    mock_path_instance.mkdir = MagicMock()
    mock_path_instance.is_absolute.return_value = False
    mock_path_instance.is_file.return_value = False  # File not found
    mock_path_instance.name = 'test.csv'
    mock_path.return_value = mock_path_instance
    
    # Run the function and expect FileNotFoundError
    with pytest.raises(SystemExit):
        data_load_run.main(mock_config)


@patch('src.data_load.run.Path')
@patch('src.data_load.run.PROJECT_ROOT')
@patch('src.data_load.run.wandb')
@patch('src.data_load.run.get_raw_data')
def test_data_load_run_exception_handling(mock_get_raw_data, mock_wandb, mock_project_root, mock_path, mock_config):
    """Test data load run exception handling."""
    from src.data_load import run as data_load_run
    
    # Setup mocks
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    mock_get_raw_data.side_effect = Exception("Test exception")
    
    # Mock PROJECT_ROOT
    mock_project_root_instance = MagicMock()
    mock_project_root_instance.mkdir = MagicMock()
    mock_project_root.__truediv__ = MagicMock(return_value=mock_project_root_instance)
    
    # Mock Path operations
    mock_path_instance = MagicMock()
    mock_path_instance.mkdir = MagicMock()
    mock_path_instance.is_absolute.return_value = False
    mock_path_instance.is_file.return_value = True
    mock_path_instance.name = 'test.csv'
    mock_path.return_value = mock_path_instance
    
    # Run the function and expect SystemExit
    with pytest.raises(SystemExit):
        data_load_run.main(mock_config)
    
    # Verify error handling
    mock_wandb_run.alert.assert_called_once()
    mock_wandb.finish.assert_called_once()


@patch('src.data_load.run.Path')
@patch('src.data_load.run.PROJECT_ROOT')
@patch('src.data_load.run.wandb')
@patch('src.data_load.run.get_raw_data')
def test_data_load_run_no_artifacts_logging(mock_get_raw_data, mock_wandb, mock_project_root, mock_path, mock_config, sample_dataframe):
    """Test data load run with artifacts logging disabled."""
    from src.data_load import run as data_load_run
    
    # Modify config to disable logging
    mock_config.data_load.log_sample_artifacts = False
    mock_config.data_load.log_summary_stats = False
    mock_config.data_load.log_artifacts = False
    
    # Setup mocks
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    mock_get_raw_data.return_value = sample_dataframe
    
    # Mock PROJECT_ROOT
    mock_project_root_instance = MagicMock()
    mock_project_root_instance.mkdir = MagicMock()
    mock_project_root.__truediv__ = MagicMock(return_value=mock_project_root_instance)
    
    # Mock Path operations
    mock_path_instance = MagicMock()
    mock_path_instance.mkdir = MagicMock()
    mock_path_instance.is_absolute.return_value = False
    mock_path_instance.is_file.return_value = True
    mock_path_instance.name = 'test.csv'
    mock_path.return_value = mock_path_instance
    
    # Run the function
    data_load_run.main(mock_config)
    
    # Assertions
    mock_wandb.init.assert_called_once()
    mock_wandb.finish.assert_called_once()


@patch('src.data_load.run.Path')
@patch('src.data_load.run.wandb')
@patch('src.data_load.run.get_raw_data')
def test_data_load_run_absolute_path(mock_get_raw_data, mock_wandb, mock_path, mock_config, sample_dataframe):
    """Test data load run with absolute path."""
    from src.data_load import run as data_load_run
    
    # Setup mocks
    mock_wandb_run = MagicMock()
    mock_wandb.init.return_value = mock_wandb_run
    mock_get_raw_data.return_value = sample_dataframe
    
    # Mock Path operations
    mock_path_instance = MagicMock()
    mock_path_instance.mkdir = MagicMock()
    mock_path_instance.is_absolute.return_value = True  # Absolute path
    mock_path_instance.is_file.return_value = True
    mock_path_instance.name = 'test.csv'
    mock_path.return_value = mock_path_instance
    
    # Run the function
    data_load_run.main(mock_config)
    
    # Assertions
    mock_wandb.init.assert_called_once()
    mock_wandb.finish.assert_called_once()
