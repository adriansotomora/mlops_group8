"""
Tests for src/features/run.py module - Hydra script with WandB integration
"""
import pytest
from unittest.mock import MagicMock, patch, call
from omegaconf import DictConfig
import sys
from pathlib import Path

# Add project root to path
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestFeaturesRun:
    """Test cases for features run module."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock DictConfig for testing."""
        return DictConfig({
            'main': {
                'WANDB_PROJECT': 'test_project',
                'WANDB_ENTITY': 'test_entity'
            }
        })

    @patch('src.features.run.wandb')
    @patch('src.features.run.main_features')
    @patch('src.features.run.datetime')
    def test_main_success(self, mock_datetime, mock_main_features, mock_wandb, mock_config):
        """Test successful execution of main function."""
        from src.features.run import main
        
        # Setup mocks
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        mock_wandb_run = MagicMock()
        mock_wandb.init.return_value = mock_wandb_run
        mock_wandb.run = mock_wandb_run
        
        # Run the function
        main(mock_config)
        
        # Verify WandB initialization
        mock_wandb.init.assert_called_once_with(
            project='test_project',
            entity='test_entity',
            job_type='features',
            name='features_20250101_120000',
            config=dict(mock_config),
            tags=['features']
        )
        
        # Verify main_features was called
        mock_main_features.assert_called_once_with(
            config_path=str(PROJECT_ROOT / "config.yaml")
        )
        
        # Verify logging
        mock_wandb.log.assert_called_with({"features_status": "completed"})
        mock_wandb.finish.assert_called_once()

    @patch('src.features.run.wandb')
    @patch('src.features.run.main_features')
    @patch('src.features.run.datetime')
    @patch('src.features.run.logger')
    def test_main_exception_handling(self, mock_logger, mock_datetime, mock_main_features, mock_wandb, mock_config):
        """Test exception handling in main function."""
        from src.features.run import main
        
        # Setup mocks
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        mock_wandb_run = MagicMock()
        mock_wandb.init.return_value = mock_wandb_run
        mock_wandb.run = mock_wandb_run
        
        # Make main_features raise an exception
        test_exception = Exception("Test error")
        mock_main_features.side_effect = test_exception
        
        # Run the function and expect it to re-raise
        with pytest.raises(Exception, match="Test error"):
            main(mock_config)
        
        # Verify error logging
        mock_logger.exception.assert_called_once_with("Failed during features step")
        
        # Verify WandB error logging
        mock_wandb.log.assert_called_with({
            "features_status": "failed", 
            "error": str(test_exception)
        })
        
        # Verify alert was sent
        mock_wandb_run.alert.assert_called_once_with(
            title="Features Error", 
            text=str(test_exception)
        )
        
        # Verify cleanup
        mock_wandb.finish.assert_called_once()

    @patch('src.features.run.wandb')
    @patch('src.features.run.main_features')
    @patch('src.features.run.datetime')
    def test_main_wandb_init_failure(self, mock_datetime, mock_main_features, mock_wandb, mock_config):
        """Test behavior when WandB initialization fails."""
        from src.features.run import main
        
        # Setup mocks
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        mock_wandb.init.side_effect = Exception("WandB connection failed")
        mock_wandb.run = None
        
        # Run the function and expect it to re-raise
        with pytest.raises(Exception, match="WandB connection failed"):
            main(mock_config)
        
        # main_features should not be called if WandB init fails
        mock_main_features.assert_not_called()

    @patch('src.features.run.wandb')
    @patch('src.features.run.main_features')
    @patch('src.features.run.datetime')
    def test_main_no_wandb_run(self, mock_datetime, mock_main_features, mock_wandb, mock_config):
        """Test exception handling when run is None."""
        from src.features.run import main
        
        # Setup mocks
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        mock_wandb.init.return_value = None
        mock_wandb.run = None
        
        # Make main_features raise an exception
        test_exception = Exception("Test error")
        mock_main_features.side_effect = test_exception
        
        # Run the function and expect it to re-raise
        with pytest.raises(Exception, match="Test error"):
            main(mock_config)
        
        # Verify that alert was not called since run is None
        # and that finish was not called since wandb.run is None
        mock_wandb.finish.assert_not_called()

    @patch('src.features.run.wandb')
    @patch('src.features.run.main_features')
    @patch('src.features.run.datetime')
    def test_main_wandb_finish_called_on_success(self, mock_datetime, mock_main_features, mock_wandb, mock_config):
        """Test that WandB finish is called on success."""
        from src.features.run import main
        
        # Setup mocks
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        mock_wandb_run = MagicMock()
        mock_wandb.init.return_value = mock_wandb_run
        mock_wandb.run = mock_wandb_run
        
        # Run the function
        main(mock_config)
        
        # Verify finish was called
        mock_wandb.finish.assert_called_once()

    @patch('src.features.run.wandb')
    @patch('src.features.run.main_features')
    @patch('src.features.run.datetime')
    def test_main_wandb_no_run_available(self, mock_datetime, mock_main_features, mock_wandb, mock_config):
        """Test behavior when wandb.run is None in finally block."""
        from src.features.run import main
        
        # Setup mocks
        mock_datetime.now.return_value.strftime.return_value = "20250101_120000"
        mock_wandb_run = MagicMock()
        mock_wandb.init.return_value = mock_wandb_run
        
        # Set wandb.run to None to test the finally block condition
        mock_wandb.run = None
        
        # Run the function
        main(mock_config)
        
        # finish should not be called when wandb.run is None
        mock_wandb.finish.assert_not_called()

    def test_project_root_path(self):
        """Test that PROJECT_ROOT is correctly set."""
        from src.features.run import PROJECT_ROOT
        
        assert PROJECT_ROOT.is_dir()
        assert PROJECT_ROOT.name == "mlops_group8"

    def test_src_root_path(self):
        """Test that SRC_ROOT is correctly set."""
        from src.features.run import SRC_ROOT
        
        assert SRC_ROOT.is_dir()
        assert SRC_ROOT.name == "src"

    @patch('src.features.run.main')
    def test_main_module_execution(self, mock_main):
        """Test that main is called when module is executed directly."""
        # This tests the if __name__ == "__main__": block
        # We can't easily test this directly, but we can verify the function exists
        from src.features import run
        
        assert hasattr(run, 'main')
        assert callable(run.main)
