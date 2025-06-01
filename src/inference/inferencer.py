"""
inferencer.py

Performs batch inference using a trained model and preprocessing/feature
engineering pipeline. Includes prediction intervals for regression models.

Usage:
python -m src.inference.inferencer <input_csv_path> <config_path> <output_csv_path>
Example:
python -m src.inference.inferencer data/new_unseen_data.csv config.yaml data/predictions/output_predictions.csv

Having already run the training pipeline:

python -m src.inference.inferencer data/raw_holdout/inference_holdout_data.csv config.yaml data/predictions/holdout_predictions_with_intervals.csv
"""

import argparse
import logging
import pickle
import json
import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yaml
from typing import Dict, Any, List, Optional
import joblib 

# Import paths (ensure these match your project structure)
try:
    from src.preprocess.preprocessing import drop_columns as preprocess_drop_columns
    from src.features.features import (
        parse_genres,
        create_polynomial_features,
        drop_irrelevant_columns as features_drop_columns
    )
except ModuleNotFoundError as e:
    print(f"CRITICAL ERROR: Could not import necessary modules. Ensure PYTHONPATH. Missing: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

def _setup_logging(log_level_str: str = "INFO", log_file: Optional[str] = None, log_format: Optional[str] = None, date_format: Optional[str] = None):
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    effective_log_format = log_format if log_format else "%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s"
    effective_date_format = date_format if date_format else "%Y-%m-%d %H:%M:%S"
    handlers_list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers_list.append(logging.FileHandler(log_file, mode='a'))
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format=effective_log_format,
        datefmt=effective_date_format,
        handlers=handlers_list
    )
    logging.getLogger(__name__).setLevel(log_level)

def _load_standard_pickle(path: str, label: str) -> Any:
    if not os.path.exists(path):
        logger.error(f"{label} not found at path: {path}")
        raise FileNotFoundError(f"{label} not found at path: {path}")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    logger.info(f"Successfully loaded {label} (using pickle) from {path}")
    return artifact

def _load_joblib_pickle(path: str, label: str) -> Any:
    if not os.path.exists(path):
        logger.error(f"{label} not found at path: {path}")
        raise FileNotFoundError(f"{label} not found at path: {path}")
    artifact = joblib.load(path)
    logger.info(f"Successfully loaded {label} (using joblib) from {path}")
    return artifact

def _load_json(path: str, label: str) -> Any:
    if not os.path.exists(path):
        logger.error(f"{label} not found at path: {path}")
        raise FileNotFoundError(f"{label} not found at path: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    logger.info(f"Successfully loaded {label} from {path}")
    return data

def run_inference(input_csv_path: str, config_path: str, output_csv_path: str) -> None:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Config file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"CRITICAL ERROR: Error parsing YAML '{config_path}': {e}.")
        sys.exit(1)

    log_cfg = config.get("logging", {})
    _setup_logging(
        log_level_str=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file", "logs/inference.log"),
        log_format=log_cfg.get("format"),
        date_format=log_cfg.get("datefmt")
    )
    logger.info("Starting batch inference process...")
    logger.info(f"Using configuration from: {config_path}")

    art_cfg = config.get("artifacts", {})
    model_cfg = config.get("model", {})
    active_model_type = model_cfg.get("active", "linear_regression") 

    scaler_path = art_cfg.get("preprocessing_pipeline")
    model_save_path = model_cfg.get(active_model_type, {}).get("save_path")
    
    selected_features_json_path = model_cfg.get(active_model_type, {}).get("selected_features_path")
    if not selected_features_json_path and model_save_path: 
        base, _ = os.path.splitext(model_save_path)
        selected_features_json_path = f"{base}_selected_features.json"
        logger.info(f"No explicit 'selected_features_path' in config. Using derived: {selected_features_json_path}")
    elif not model_save_path and not selected_features_json_path:
         logger.critical("Cannot determine selected features path.")
         return

    if not all([scaler_path, model_save_path, selected_features_json_path]):
        logger.critical("Missing one or more required artifact paths in config.")
        return

    try:
        scaler = _load_joblib_pickle(scaler_path, "Scaler")
        model = _load_standard_pickle(model_save_path, "Trained Model")
        selected_features_data = _load_json(selected_features_json_path, "Selected Features List")
        final_selected_features_list = selected_features_data.get("selected_features", [])
        if not final_selected_features_list:
            logger.critical("Loaded selected features list is empty.")
            return
    except FileNotFoundError:
        logger.critical("A required artifact was not found. Check paths.")
        return
    except Exception as e:
        logger.critical(f"Error loading artifacts: {e}") 
        return

    logger.info(f"Loading new input data from: {input_csv_path}")
    try:
        new_data_df = pd.read_csv(
            input_csv_path,
            delimiter=config.get("data_source", {}).get("delimiter", ","),
            header=config.get("data_source", {}).get("header", 0),
            encoding=config.get("data_source", {}).get("encoding", "utf-8")
        )
        original_data_for_output = new_data_df.copy() 
        logger.info(f"New data loaded. Shape: {new_data_df.shape}")
    except FileNotFoundError:
        logger.critical(f"Input data file not found at '{input_csv_path}'.")
        return
    except Exception as e:
        logger.critical(f"Error loading input data from '{input_csv_path}': {e}")
        return

    current_df = new_data_df.copy()

    logger.info("Applying initial preprocessing transformations...")
    pre_cfg = config.get("preprocessing", {})
    cols_to_drop_initial = pre_cfg.get("drop_columns", [])
    current_df = preprocess_drop_columns(current_df, cols_to_drop_initial, logger_param=logger)

    scale_cfg = pre_cfg.get("scale", {})
    columns_to_scale_from_config = scale_cfg.get("columns", [])
    actual_columns_to_scale = [col for col in columns_to_scale_from_config if col in current_df.columns and pd.api.types.is_numeric_dtype(current_df[col])]
    # Log warnings for columns in config not found or not numeric in current_df
    for col in columns_to_scale_from_config:
        if col not in current_df.columns:
            logger.warning(f"Scaling config: Column '{col}' not found in input data after initial drops.")
        elif col not in actual_columns_to_scale: # Implies it was not numeric
            logger.warning(f"Scaling config: Column '{col}' found but is not numeric (type: {current_df[col].dtype}). Will not be scaled.")

    if actual_columns_to_scale:
        logger.info(f"Applying scaler to columns: {actual_columns_to_scale}")
        try:
            current_df_to_scale = current_df[actual_columns_to_scale]
            if hasattr(scaler, 'feature_names_in_'):
                scaler_expected_features = list(scaler.feature_names_in_)
                # Ensure columns are in the order the scaler expects
                current_df_to_scale = current_df_to_scale[scaler_expected_features]
            transformed_data = scaler.transform(current_df_to_scale)
            current_df[actual_columns_to_scale] = transformed_data
        except Exception as e: # More specific error handling for ValueError was here before
            logger.error(f"Error applying scaler: {e}")
            return
    else:
        logger.info("No valid columns for scaling based on config and input data.")
    
    logger.info("Applying feature engineering transformations...")
    current_df = parse_genres(current_df, config, logger)
    current_df = features_drop_columns(current_df, config, logger) 
    current_df = create_polynomial_features(current_df, config, logger)
    logger.info(f"Shape after feature engineering: {current_df.shape}")

    logger.info(f"Selecting final features for prediction: {final_selected_features_list}")
    missing_model_features = [f for f in final_selected_features_list if f not in current_df.columns]
    if missing_model_features:
        logger.critical(f"Required model features missing from transformed data: {missing_model_features}")
        return
    X_inference = current_df[final_selected_features_list].astype(float) 

    X_inference_const = sm.add_constant(X_inference, has_constant='add')
    if hasattr(model, 'model') and hasattr(model.model, 'exog_names'):
        model_expected_cols = model.model.exog_names
        missing_from_inference = [col for col in model_expected_cols if col not in X_inference_const.columns]
        if missing_from_inference:
            logger.critical(f"Cols expected by model missing after adding const: {missing_from_inference}")
            return
        try:
            X_inference_const = X_inference_const[model_expected_cols] # Reorder
            logger.info("Reordered inference columns to match model's exog_names.")
        except KeyError as e:
            logger.critical(f"KeyError reordering columns for statsmodels: {e}. Expected: {model_expected_cols}, Got: {X_inference_const.columns.tolist()}")
            return
    else:
        logger.warning("Model lacks 'exog_names'; cannot verify/reorder columns. Assuming order is correct.")

    logger.info(f"Prepared data for prediction. Shape: {X_inference_const.shape}")

    # --- MODIFIED SECTION for Prediction and Prediction Intervals ---
    logger.info("Generating predictions and 95% prediction intervals...")
    try:
        pred_results_obj = model.get_prediction(X_inference_const)
        # alpha=0.05 for 95% prediction intervals
        pred_summary_df = pred_results_obj.summary_frame(alpha=0.05) 
        
        # pred_summary_df contains 'mean' (prediction), 'obs_ci_lower', 'obs_ci_upper'
        # It should have the same index as X_inference_const
    except Exception as e:
        logger.critical(f"Error during model prediction or interval generation: {e}")
        if hasattr(model, 'model') and hasattr(model.model, 'exog_names'):
             logger.critical(f"Model exog names: {model.model.exog_names}")
        logger.critical(f"Data columns passed to model: {X_inference_const.columns.tolist()}")
        return
    # --- END OF MODIFIED SECTION ---

    # Append predictions and intervals to original data
    # We use a left merge on the index to ensure all original rows are kept.
    # Predictions/intervals will be NaN for rows that might have been dropped during transformations
    # (though our current preprocessing/feature engineering for inference doesn't drop rows explicitly,
    # errors or all-NaN features could lead to issues if not handled before this stage).
    
    output_df_to_save = original_data_for_output.copy()
    
    # Select and rename columns from pred_summary_df
    predictions_to_add = pred_summary_df[['mean', 'obs_ci_lower', 'obs_ci_upper']].copy()
    predictions_to_add.rename(columns={
        'mean': 'prediction',
        'obs_ci_lower': 'prediction_pi_lower',
        'obs_ci_upper': 'prediction_pi_upper'
    }, inplace=True)

    # Merge based on index
    output_df_to_save = output_df_to_save.merge(
        predictions_to_add,
        left_index=True,
        right_index=True,
        how='left' # Keeps all rows from original_data_for_output
    )
    logger.info(f"Predictions and intervals merged. Output shape: {output_df_to_save.shape}")
    if output_df_to_save['prediction'].isnull().any():
        logger.warning("Some predictions are NaN. This might be due to rows dropped during "
                       "transformations if input data had issues, or if indices didn't align perfectly.")


    logger.info(f"Writing predictions to: {output_csv_path}")
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir): 
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        output_df_to_save.to_csv(output_csv_path, index=False)
        logger.info("Inference complete. Predictions and intervals saved.")
    except Exception as e:
        logger.critical(f"Error saving predictions to '{output_csv_path}': {e}")

def main_cli():
    parser = argparse.ArgumentParser(
        description="Run batch inference on new data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_csv", help="Path to the new raw input CSV data file.")
    parser.add_argument("config", help="Path to the YAML config file.")
    parser.add_argument("output_csv", help="Path to save the output CSV with predictions.")
    args = parser.parse_args()
    run_inference(
        input_csv_path=args.input_csv,
        config_path=args.config,
        output_csv_path=args.output_csv
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", stream=sys.stdout)
    main_cli()
