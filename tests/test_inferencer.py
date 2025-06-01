import subprocess
import os
import pandas as pd
import pytest
import yaml
import joblib
import pickle
import json
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, List 
import logging 
from pathlib import Path 
import sys 

# Define base paths relative to this test file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = Path(TEST_DIR).parent 

# Setup a logger for test setup diagnostics
test_setup_logger = logging.getLogger("test_inferencer_setup")
if not test_setup_logger.handlers:
    handler = logging.StreamHandler(sys.stdout) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    test_setup_logger.addHandler(handler)
    test_setup_logger.setLevel(logging.INFO)

# Ensure project root is in sys.path for src imports
if str(PROJECT_ROOT) not in sys.path: 
    sys.path.insert(0, str(PROJECT_ROOT))

# Import functions from the main application scripts
try:
    from src.preprocess.preprocessing import drop_columns as preprocess_drop_columns
    from src.features.features import (
        parse_genres as features_parse_genres,
        create_polynomial_features as features_create_polynomial_features,
        drop_irrelevant_columns as features_drop_columns
    )
except ModuleNotFoundError as e:
    test_setup_logger.error(f"Error importing from src: {e}. Ensure PYTHONPATH or run from project root.")

@pytest.fixture(scope="module")
def mock_environment(tmp_path_factory):
    """Sets up a mock environment with data, config, and artifacts for testing inference."""
    base_temp_dir = tmp_path_factory.mktemp("inference_test_env")
    mock_artifacts_dir = base_temp_dir / "mock_artifacts"
    mock_predictions_dir = base_temp_dir / "mock_predictions"
    mock_artifacts_dir.mkdir(exist_ok=True)
    mock_predictions_dir.mkdir(exist_ok=True)

    user_mock_input_csv_path_str = "tests/mock_data/mock_data_spotify.csv"
    mock_input_csv_path = PROJECT_ROOT / user_mock_input_csv_path_str
    
    if not mock_input_csv_path.exists():
        pytest.fail(f"User mock data file not found: {mock_input_csv_path}.")
    try:
        mock_input_df = pd.read_csv(mock_input_csv_path)
    except Exception as e:
        pytest.fail(f"Failed to read user mock data CSV {mock_input_csv_path}: {e}")

    cfg_preprocessing_drop_cols = ['instrumentalness'] 
    cfg_scale_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo', 'duration_ms', 'key']
    cfg_features_drop_cols = ['track_name', 'album', 'artist_name', 'artist_genres', 'year'] 
    cfg_audio_features_for_poly = ['danceability', 'energy', 'loudness'] 
    cfg_genre_names_for_poly = ['pop', 'electronic', 'jazz'] 

    for col_list_name, col_list in [("cfg_scale_cols", cfg_scale_cols), ("cfg_audio_features_for_poly", cfg_audio_features_for_poly)]:
        missing = [col for col in col_list if col not in mock_input_df.columns]
        if missing: pytest.fail(f"Missing columns in '{user_mock_input_csv_path_str}' for {col_list_name}: {missing}")

    temp_df_for_scaler = mock_input_df.drop(columns=cfg_preprocessing_drop_cols, errors='ignore')
    actual_cols_for_scaler_fit = [col for col in cfg_scale_cols if col in temp_df_for_scaler.columns]
    if not actual_cols_for_scaler_fit: pytest.fail("No columns found to fit the mock scaler.")
    scaler_fit_data = temp_df_for_scaler[actual_cols_for_scaler_fit].copy()
    for col in actual_cols_for_scaler_fit: 
        if not pd.api.types.is_numeric_dtype(scaler_fit_data[col]): pytest.fail(f"Column '{col}' for scaler not numeric.")
    
    mock_scaler = MinMaxScaler()
    mock_scaler.fit(scaler_fit_data) 
    mock_scaler_path = mock_artifacts_dir / "mock_scaler.pkl"
    joblib.dump(mock_scaler, mock_scaler_path)

    transformed_df = mock_input_df.copy()
    # Use logger_param for functions from preprocessing.py
    transformed_df = preprocess_drop_columns(transformed_df, cfg_preprocessing_drop_cols, logger_param=test_setup_logger)
    
    dummy_feat_config = {
        'features': {
            'drop': cfg_features_drop_cols, 'audio_features': cfg_audio_features_for_poly, 
            'genre_features': cfg_genre_names_for_poly,
            'polynomial': {
                'audio': {'degree': 2, 'include_bias': False, 'interaction_only': False},
                'genre': {'degree': 2, 'include_bias': False, 'interaction_only': False}
            }}}
    if 'artist_genres' in transformed_df.columns: 
        # Assuming functions from features.py take 'logger' as the keyword argument
        transformed_df = features_parse_genres(transformed_df, dummy_feat_config, logger=test_setup_logger)
    transformed_df = features_drop_columns(transformed_df, dummy_feat_config, logger=test_setup_logger)
    
    actual_audio_poly_inputs = [c for c in cfg_audio_features_for_poly if c in transformed_df.columns]
    dummy_feat_config['features']['audio_features'] = actual_audio_poly_inputs 
    transformed_df = features_create_polynomial_features(transformed_df, dummy_feat_config, logger=test_setup_logger)

    mock_model_feature_names = []
    base_feats_present = [c for c in cfg_scale_cols if c in transformed_df.columns and pd.api.types.is_numeric_dtype(transformed_df[c])]
    if base_feats_present: mock_model_feature_names.extend(base_feats_present[:2]) 
    
    for g_name in cfg_genre_names_for_poly: 
        if f"genre_{g_name}" in transformed_df.columns: mock_model_feature_names.append(f"genre_{g_name}"); break
    
    for audio_f_base in cfg_audio_features_for_poly: 
        poly_candidate_sq = f"poly_audio_{audio_f_base.replace(' ', '_')}^2"
        if poly_candidate_sq in transformed_df.columns: mock_model_feature_names.append(poly_candidate_sq); break
    
    mock_model_feature_names = list(dict.fromkeys(mock_model_feature_names)) 
    if not mock_model_feature_names: 
        numeric_cols = transformed_df.select_dtypes(include=np.number).columns
        mock_model_feature_names = [numeric_cols[0]] if not numeric_cols.empty else ['fallback_dummy_feature']
        if 'fallback_dummy_feature' in mock_model_feature_names and 'fallback_dummy_feature' not in transformed_df.columns:
            transformed_df['fallback_dummy_feature'] = 0 

    mock_selected_features_json = {"selected_features": mock_model_feature_names, "count": len(mock_model_feature_names)}
    mock_selected_features_path = mock_artifacts_dir / "mock_model_selected_features.json"
    with open(mock_selected_features_path, 'w') as f: json.dump(mock_selected_features_json, f)

    missing_feats_for_training = [f_name for f_name in mock_model_feature_names if f_name not in transformed_df.columns]
    if missing_feats_for_training: pytest.fail(f"Mock model training features missing from transformed_df: {missing_feats_for_training}")
    
    X_dummy_model = transformed_df[mock_model_feature_names].astype(float).fillna(0) 
    y_dummy_model = (X_dummy_model.iloc[:, 0] * 0.5 + np.random.rand(len(X_dummy_model)) * 0.05).fillna(0) 
    X_dummy_model_const = sm.add_constant(X_dummy_model, has_constant='add')
    mock_ols_model = sm.OLS(y_dummy_model, X_dummy_model_const).fit()
    mock_model_path = mock_artifacts_dir / "mock_model.pkl"
    with open(mock_model_path, 'wb') as f: pickle.dump(mock_ols_model, f)

    mock_test_config = {
        'data_source': { 'delimiter': ',', 'header': 0, 'encoding': 'utf-8'},
        'logging': {'level': 'DEBUG', 'log_file': str(base_temp_dir / "test_inferencer_run.log")},
        'preprocessing': {'drop_columns': cfg_preprocessing_drop_cols, 'scale': { 'columns': cfg_scale_cols, 'method': 'minmax' }},
        'features': dummy_feat_config['features'], 
        'artifacts': {'preprocessing_pipeline': str(mock_scaler_path), 'processed_dir': str(mock_predictions_dir)},
        'model': {
            'active': 'test_lr', 
            'test_lr': { 'save_path': str(mock_model_path), 'selected_features_path': str(mock_selected_features_path)}
        }}
    mock_config_yaml_path = base_temp_dir / "mock_config_test.yaml"
    with open(mock_config_yaml_path, 'w') as f: yaml.dump(mock_test_config, f)

    return {
        "input_csv": str(mock_input_csv_path), 
        "config_yaml": str(mock_config_yaml_path),
        "output_dir": mock_predictions_dir,
        "project_root": str(PROJECT_ROOT)
    }

def test_run_inference_successful(mock_environment):
    """Test inferencer.py runs successfully and output contains prediction and interval columns."""
    paths = mock_environment
    input_csv = paths["input_csv"] 
    config_file = paths["config_yaml"]
    output_csv = paths["output_dir"] / "output_predictions.csv" 

    command = [sys.executable, "-m", "src.inference.inferencer", input_csv, config_file, str(output_csv)]
    test_setup_logger.info(f"Executing: {' '.join(command)}") 
    result = subprocess.run(command, cwd=paths["project_root"], capture_output=True, text=True, check=False)

    if result.returncode != 0: 
        print("\n--- INFERENCE SCRIPT STDOUT ---\n", result.stdout)
        print("--- INFERENCE SCRIPT STDERR ---\n", result.stderr)
    
    assert result.returncode == 0, f"Inference script failed. Stderr: {result.stderr}"
    assert os.path.exists(output_csv), f"Output CSV not created: {output_csv}"
    
    try:
        predictions_df = pd.read_csv(output_csv)
        original_input_df = pd.read_csv(input_csv)
    except Exception as e: pytest.fail(f"Error reading CSVs for validation: {e}")
    
    assert len(predictions_df) == len(original_input_df), "Output row count mismatch."
    
    expected_cols = ["prediction", "prediction_pi_lower", "prediction_pi_upper"]
    for col in expected_cols:
        assert col in predictions_df.columns, f"'{col}' column missing."
        assert pd.api.types.is_numeric_dtype(predictions_df[col]), f"'{col}' not numeric."
        if predictions_df[col].notnull().any() and predictions_df[col].isnull().all():
             test_setup_logger.warning(f"All values in '{col}' are NaN in output.")
    
    if predictions_df['prediction'].notnull().any(): 
        assert predictions_df['prediction_pi_lower'].notnull().any()
        assert predictions_df['prediction_pi_upper'].notnull().any()
    test_setup_logger.info(f"Inference test passed for {input_csv}.")

