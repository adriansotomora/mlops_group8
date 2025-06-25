import os
import pandas as pd
import numpy as np
import yaml
import pytest 
from unittest.mock import MagicMock, patch 
import logging 
from pathlib import Path # Ensure Path is imported

# Ensure the 'src' directory is in PYTHONPATH for imports
import sys
TEST_DIR_FEATURES = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_FEATURES = os.path.abspath(os.path.join(TEST_DIR_FEATURES, '..'))
if PROJECT_ROOT_FEATURES not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FEATURES)

from src.features import features # Import the module to be tested

# Logger for test setup information
test_setup_logger_features = logging.getLogger("test_features_setup")
if not test_setup_logger_features.handlers:
    handler_features = logging.StreamHandler(sys.stdout)
    formatter_features = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_features.setFormatter(formatter_features)
    test_setup_logger_features.addHandler(handler_features)
    test_setup_logger_features.setLevel(logging.INFO)


def minimal_config(tmpdir_path: Path, preprocessed_input_csv_path: str): # tmpdir_path is Path object
    """
    Creates a minimal config for testing features.py.
    It needs to know where the (mock) preprocessed input data is.
    """
    # Use Path objects for constructing paths, then convert to string for YAML
    feature_list_file = tmpdir_path / "feature_list.txt"
    engineered_features_file = tmpdir_path / "features.csv"
    log_file = tmpdir_path / "test_features.log"

    return {
        "data_source": { # Needed by features.main_features to load its input
            "processed_path": str(preprocessed_input_csv_path) 
        },
        "features": {
            "drop": ["dropme_in_features"], 
            "audio_features": ["danceability", "energy"], # Ensure these exist in toy_df
            "genre_features": ["pop", "rock"], # Base names, ensure parse_genres creates these
            "polynomial": {
                "audio": {"degree": 2, "include_bias": False, "interaction_only": False},
                "genre": {"degree": 2, "include_bias": False, "interaction_only": False}
            },
            "exclude": ["year_excluded"],
            "profiling_variables": ["target_col_profiling"]
        },
        "artifacts": {
            "processed_dir": str(tmpdir_path), 
            "feature_list_filename": "feature_list.txt", # Used by features.py
            "engineered_features_filename": "features.csv" # Used by features.py
        },
        "logging": {
            "log_file": str(log_file),
            "level": "DEBUG"
        }
    }

def toy_df_for_features_input():
    """Creates a DataFrame simulating the output of preprocessing.py."""
    return pd.DataFrame({
        "artist_genres": ["[pop; rock]", "[rock]", "[pop]", "[jazz hip-hop]"],
        "danceability": [0.1, 0.2, 0.3, 0.4], 
        "energy": [0.5, 0.6, 0.7, 0.8],       
        "loudness": np.random.rand(4),        
        "year_excluded": [2020, 2021, 2022, 2023], # For testing 'exclude'
        "target_col_profiling": [10, 20, 30, 40],  # For testing 'profiling_variables'
        "dropme_in_features": [1, 2, 3, 4],        # For testing 'features.drop'
        "some_other_numeric_feature": np.random.rand(4)
        # Ensure columns listed in minimal_config's audio_features exist here
    })

# Patch the logger in the 'features' module for unit tests of individual functions
@patch.object(features, 'logger', MagicMock())
def test_parse_genres_creates_columns(): # Removed tmp_path as it's not used directly here
    df = toy_df_for_features_input()
    config_for_parse = {"features": {"genre_features": ["pop", "rock", "hip-hop"]}} 
    
    df2 = features.parse_genres(df.copy(), config_for_parse, features.logger) # Pass the patched logger
    assert "genre_pop" in df2.columns
    assert "genre_rock" in df2.columns
    assert "genre_hip_hop" in df2.columns 
    assert set(df2["genre_pop"]).issubset({0, 1})
    features.logger.info.assert_any_call("Parsing and engineering genre features...")


@patch.object(features, 'logger', MagicMock())
def test_drop_irrelevant_columns_removes(): # Removed tmp_path
    df = toy_df_for_features_input()
    config_for_drop = {"features": {"drop": ["dropme_in_features"]}}
    
    df2 = features.drop_irrelevant_columns(df.copy(), config_for_drop, features.logger)
    assert "dropme_in_features" not in df2.columns
    assert "artist_genres" in df2.columns 
    features.logger.info.assert_any_call("Dropping columns (from features.drop): ['dropme_in_features']")


@patch.object(features, 'logger', MagicMock())
def test_create_polynomial_features_adds_columns(): # Removed tmp_path
    df = toy_df_for_features_input()
    config_for_poly = {
        "features": {
            "audio_features": ["danceability", "energy"], 
            "genre_features": ["pop"], 
            "polynomial": {
                "audio": {"degree": 2, "include_bias": False},
                "genre": {"degree": 2, "include_bias": False}
            }
        }
    }
    # parse_genres needs to be called to create 'genre_pop' for the poly features
    df_with_genres = features.parse_genres(df.copy(), config_for_poly, features.logger) 
    
    df_with_poly = features.create_polynomial_features(df_with_genres, config_for_poly, features.logger)
    assert any(c.startswith("audio_poly_") for c in df_with_poly.columns)
    assert any(c.startswith("genre_poly_") for c in df_with_poly.columns)


@patch.object(features, 'logger', MagicMock())
def test_select_features_excludes_and_profiles(): # Removed tmp_path
    df = toy_df_for_features_input()
    config_for_select = {
        "features": {
            "exclude": ["year_excluded"],
            "profiling_variables": ["target_col_profiling"]
        }
    }
    df_for_select = df.copy()
    df_for_select["numeric_to_keep"] = np.random.rand(len(df)) # Add a numeric column to be kept

    selected_df = features.select_features(df_for_select, config_for_select, features.logger)
    assert "year_excluded" not in selected_df.columns
    assert "target_col_profiling" not in selected_df.columns
    assert "numeric_to_keep" in selected_df.columns 


@patch.object(features, 'logger', MagicMock())
def test_log_feature_list_creates_file(tmp_path: Path): # Use Path type hint
    df = toy_df_for_features_input()[["danceability", "energy"]] 
    feature_list_path = tmp_path / "feature_list.txt" # Use Path object for path construction
    
    features.log_feature_list(df, str(feature_list_path), features.logger) # Convert to string for open()
    
    assert feature_list_path.exists()
    with open(feature_list_path) as f:
        lines = f.read().splitlines()
    assert "danceability" in lines
    assert "energy" in lines


def test_main_features_e2e(tmp_path: Path): # Use Path type hint
    """End-to-end test for the main_features function."""
    preprocessed_df = toy_df_for_features_input()
    
    mock_preprocessed_csv_path = tmp_path / "mock_preprocessed_data.csv"
    preprocessed_df.to_csv(mock_preprocessed_csv_path, index=False)
    test_setup_logger_features.info(f"E2E Test: Saved mock preprocessed data to {mock_preprocessed_csv_path}")

    test_config = minimal_config(tmp_path, str(mock_preprocessed_csv_path)) # Pass string path
    
    config_yaml_path = tmp_path / "test_features_config.yaml"
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(test_config, f)
    test_setup_logger_features.info(f"E2E Test: Saved mock config to {config_yaml_path}")

    # Call main_features with only the config_path
    features.main_features(config_path=str(config_yaml_path))

    # Check for expected output files based on paths in test_config
    engineered_features_output_path = Path(test_config["artifacts"]["processed_dir"]) / test_config["artifacts"]["engineered_features_filename"]
    feature_list_output_path = Path(test_config["artifacts"]["processed_dir"]) / test_config["artifacts"]["feature_list_filename"]

    assert engineered_features_output_path.exists(), "Engineered features CSV not created."
    assert feature_list_output_path.exists(), "Feature list file not created."

    df_out = pd.read_csv(engineered_features_output_path)
    assert "dropme_in_features" not in df_out.columns
    assert any(c.startswith("audio_poly_") for c in df_out.columns), "No audio polynomial features found."
    
    # Check for a specific polynomial genre feature if its base was created
    if "genre_pop" in df_out.columns: 
         assert any(c.startswith("genre_poly_") for c in df_out.columns), "No pop polynomial features found."
    
    with open(feature_list_output_path, 'r') as f:
        logged_features = [line.strip() for line in f.readlines()]
    assert len(logged_features) == len(df_out.columns)
    # Check if at least one known base feature (that wasn't dropped or excluded) is in the list
    assert "danceability" in logged_features or "energy" in logged_features

    test_setup_logger_features.info("test_main_features_e2e passed.")

