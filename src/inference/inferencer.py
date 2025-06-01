"""
inferencer.py

Performs batch inference using a trained model and preprocessing/feature
engineering pipeline.

Usage:
python -m src.inference.inferencer <input_csv_path> <config_path> <output_csv_path>
Example:
python -m src.inference.inferencer data/new_unseen_data.csv config.yaml data/predictions/output_predictions.csv
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
import joblib # <<<<------ ADD THIS IMPORT

# It's crucial that these import paths match your project structure.
# Assuming preprocessing.py is at src/preprocess/preprocessing.py
# and features.py is at src/features/features.py
try:
    from src.preprocess.preprocessing import drop_columns as preprocess_drop_columns
    from src.features.features import (
        parse_genres,
        create_polynomial_features,
        drop_irrelevant_columns as features_drop_columns
    )
except ModuleNotFoundError as e:
    print(f"CRITICAL ERROR: Could not import necessary modules. Ensure your PYTHONPATH is set correctly or run from project root using -m. Missing: {e}")
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

def _load_standard_pickle(path: str, label: str) -> Any: # Renamed for clarity
    """Loads a standard pickled object (e.g., statsmodels model)."""
    if not os.path.exists(path):
        logger.error(f"{label} not found at path: {path}")
        raise FileNotFoundError(f"{label} not found at path: {path}")
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    logger.info(f"Successfully loaded {label} (using pickle) from {path}")
    return artifact

def _load_joblib_pickle(path: str, label: str) -> Any: # New function for joblib
    """Loads a joblib-pickled object (e.g., scikit-learn scaler)."""
    if not os.path.exists(path):
        logger.error(f"{label} not found at path: {path}")
        raise FileNotFoundError(f"{label} not found at path: {path}")
    artifact = joblib.load(path) # Use joblib.load()
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
        print(f"CRITICAL ERROR: Configuration file '{config_path}' not found. Cannot proceed.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"CRITICAL ERROR: Error parsing YAML in '{config_path}': {e}. Cannot proceed.")
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
        # Use _load_joblib_pickle for the scaler
        scaler = _load_joblib_pickle(scaler_path, "Scaler")
        # Use _load_standard_pickle for the statsmodels model
        model = _load_standard_pickle(model_save_path, "Trained Model")
        
        selected_features_data = _load_json(selected_features_json_path, "Selected Features List")
        final_selected_features_list = selected_features_data.get("selected_features", [])
        if not final_selected_features_list:
            logger.critical("Loaded selected features list is empty. Cannot proceed.")
            return
    except FileNotFoundError:
        logger.critical("A required artifact was not found during loading. Check paths and previous pipeline steps.")
        return
    except Exception as e:
        logger.critical(f"Error loading artifacts: {e}") # This is where your error was caught
        return

    # ... (rest of the run_inference function remains the same as in inferencer_py_fix_optional) ...
    # 2. Load New Raw Data
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

    # 3. Apply Preprocessing transformations (matching training)
    logger.info("Applying initial preprocessing transformations...")
    pre_cfg = config.get("preprocessing", {})

    cols_to_drop_initial = pre_cfg.get("drop_columns", [])
    current_df = preprocess_drop_columns(current_df, cols_to_drop_initial, logger_param=logger)

    scale_cfg = pre_cfg.get("scale", {})
    columns_to_scale_from_config = scale_cfg.get("columns", [])
    
    actual_columns_to_scale = []
    for col in columns_to_scale_from_config:
        if col not in current_df.columns:
            logger.warning(f"Column '{col}' (for scaling) not found in input data. Skipping scaling for this column.")
            continue
        if not pd.api.types.is_numeric_dtype(current_df[col]): 
            logger.warning(f"Column '{col}' (for scaling) is not numeric in input data (type: {current_df[col].dtype}). Skipping scaling for this column.")
            continue
        actual_columns_to_scale.append(col)

    if actual_columns_to_scale:
        logger.info(f"Applying scaler to columns: {actual_columns_to_scale}")
        try:
            current_df_to_scale = current_df[actual_columns_to_scale]
            if hasattr(scaler, 'feature_names_in_'):
                scaler_expected_features = list(scaler.feature_names_in_)
                current_df_to_scale = current_df_to_scale[scaler_expected_features]
            
            transformed_data = scaler.transform(current_df_to_scale)
            current_df[actual_columns_to_scale] = transformed_data
        except ValueError as e:
            logger.error(f"Error applying scaler: {e}")
            if hasattr(scaler, 'n_features_in_'):
                 logger.error(f"Scaler expected {scaler.n_features_in_} features.")
            if hasattr(scaler, 'feature_names_in_'):
                 logger.error(f"Scaler was fit on features: {list(scaler.feature_names_in_)}")
            return 
        except Exception as e:
            logger.error(f"Unexpected error applying scaler: {e}")
            return
    else:
        logger.info("No valid columns found or specified for scaling in the input data based on config.")
    
    # 4. Apply Feature Engineering transformations (matching features.py)
    logger.info("Applying feature engineering transformations...")
    current_df = parse_genres(current_df, config, logger)
    current_df = features_drop_columns(current_df, config, logger) 
    current_df = create_polynomial_features(current_df, config, logger)
    logger.info(f"Shape after feature engineering: {current_df.shape}")

    # 5. Select the exact features the model was trained on
    logger.info(f"Selecting final features for prediction: {final_selected_features_list}")
    
    missing_model_features = [f for f in final_selected_features_list if f not in current_df.columns]
    if missing_model_features:
        logger.critical(f"One or more features required by the model are missing from the transformed input data: {missing_model_features}")
        return

    X_inference = current_df[final_selected_features_list].astype(float) 

    # 6. Add constant for statsmodels
    X_inference_const = sm.add_constant(X_inference, has_constant='add')
    
    if hasattr(model, 'model') and hasattr(model.model, 'exog_names'):
        model_expected_cols = model.model.exog_names
        missing_from_inference = [col for col in model_expected_cols if col not in X_inference_const.columns]
        if missing_from_inference:
            logger.critical(f"Columns expected by the model are missing from the inference data after adding constant: {missing_from_inference}")
            return
        try:
            X_inference_const = X_inference_const[model_expected_cols]
            logger.info("Reordered inference columns to match model's exog_names.")
        except KeyError as e:
            logger.critical(f"KeyError during column reordering for statsmodels: {e}. "
                           f"Model expected: {model_expected_cols}, Inference data has: {X_inference_const.columns.tolist()}")
            return
    else:
        logger.warning("Model does not have 'exog_names' attribute; cannot verify/reorder columns. Assuming current order is correct.")

    logger.info(f"Prepared data for prediction. Shape: {X_inference_const.shape}")

    # 7. Make Predictions
    logger.info("Generating predictions...")
    try:
        predictions = model.predict(X_inference_const)
    except Exception as e:
        logger.critical(f"Error during model prediction: {e}")
        # ... (error details logging) ...
        return

    # 8. Append predictions to original data and save
    if len(predictions) == len(original_data_for_output):
        original_data_for_output["prediction"] = predictions
        output_df_to_save = original_data_for_output
    else: # Fallback logic
        logger.warning(f"Length of predictions ({len(predictions)}) does not match original data ({len(original_data_for_output)}).")
        if current_df.index.equals(original_data_for_output.index) and X_inference_const.index.equals(current_df.index) : 
            predictions_series = pd.Series(predictions, index=X_inference_const.index)
            original_data_for_output["prediction"] = predictions_series 
            output_df_to_save = original_data_for_output
            logger.info("Aligned predictions to original data using index.")
        else: 
            logger.error("Cannot reliably align predictions. Saving predictions only.")
            output_df_to_save = pd.DataFrame({'prediction': predictions})
            if not X_inference_const.index.empty:
                 output_df_to_save.index = X_inference_const.index 

    logger.info(f"Writing predictions to: {output_csv_path}")
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir): 
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        output_df_to_save.to_csv(output_csv_path, index=False)
        logger.info("Inference complete. Predictions saved.")
    except Exception as e:
        logger.critical(f"Error saving predictions to '{output_csv_path}': {e}")

def main_cli():
    parser = argparse.ArgumentParser(
        description="Run batch inference on new data using a trained model and full preprocessing/feature engineering pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_csv", help="Path to the new raw input CSV data file.")
    parser.add_argument("config", help="Path to the YAML configuration file (e.g., config.yaml).")
    parser.add_argument("output_csv", help="Path to save the output CSV file with predictions.")
    args = parser.parse_args()
    run_inference(
        input_csv_path=args.input_csv,
        config_path=args.config,
        output_csv_path=args.output_csv
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", stream=sys.stdout)
    main_cli()
