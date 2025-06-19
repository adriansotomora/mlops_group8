"""
Regression model training, evaluation, and artifact saving.

This module implements regression model training, evaluation, and artifact saving
using pre-processed and feature-engineered data.
It uses evaluator_regression.py for model evaluation.
"""
import os
import json
import logging
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
# MSE and R2 are calculated by the evaluator, but sklearn.model_selection is still used
from sklearn.model_selection import train_test_split

# Assuming evaluator_regression.py is in src/evaluation/
# Adjust the import path if your evaluator_regression.py is located elsewhere.
from src.evaluation.evaluator import evaluate_statsmodels_model 

# Module-level logger - will be initialized by main_modeling
logger = logging.getLogger(__name__)

def get_logger(logging_config: Dict[str, Any]) -> logging.Logger:
    """Set up and return a logger based on the provided configuration."""
    log_file = logging_config.get("log_file", "logs/modeling.log") 
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_format = logging_config.get(
        "format", "%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s"
    )
    date_format = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_level_str = logging_config.get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    current_handlers = logging.getLogger().handlers
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in current_handlers) and \
       not any(isinstance(h, logging.StreamHandler) for h in current_handlers):
        
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt=date_format,
        )
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        file_handler.setLevel(log_level)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
        console_handler.setLevel(log_level)
        
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().addHandler(console_handler)
        logging.getLogger().setLevel(log_level)

    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(log_level) 
    return module_logger

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    temp_logger = logging.getLogger(f"{__name__}.load_config")
    temp_logger.info(f"Loading configuration from: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    temp_logger.info("Configuration loaded successfully.")
    return config

def validate_modeling_config(config: Dict[str, Any]) -> None:
    """Validate the presence of essential keys in the modeling configuration."""
    logger.info("Validating modeling configuration.")
    required_top_keys = ["artifacts", "data_split", "model", "logging", "target", "data_source"]
    for key in required_top_keys:
        if key not in config:
            raise KeyError(f"Missing required top-level config key: '{key}'")

    if "processed_dir" not in config["artifacts"]:
         raise KeyError("Missing 'processed_dir' in 'artifacts' config for loading features.")
    if config["artifacts"].get("engineered_features_filename") is None and not os.path.exists(os.path.join(config["artifacts"]["processed_dir"], "features.csv")):
         logger.warning("Default 'features.csv' will be used, ensure 'engineered_features_filename' is set in artifacts if different.")
    
    if "processed_path" not in config["data_source"]:
        raise KeyError("Missing 'processed_path' in 'data_source' config for loading target variable.")

    active_model_type = config["model"].get("active")
    if not active_model_type:
        raise KeyError("Missing 'active' model type in 'model' config.")
    if active_model_type not in config["model"] or "save_path" not in config["model"][active_model_type]:
        raise KeyError(f"Missing 'save_path' for active model '{active_model_type}' in 'model' config.")
    
    if "metrics_path" not in config["artifacts"]:
        raise KeyError("Missing 'metrics_path' in 'artifacts' config.")
    logger.info("Modeling configuration validation successful.")


def stepwise_selection(
    X: pd.DataFrame,
    y: pd.Series,
    threshold_in: float = 0.05,
    threshold_out: float = 0.1,
    verbose: bool = True
) -> List[str]:
    """
    Perform forward-backward stepwise selection using p-values.
    
    Returns a list of selected feature names.
    """
    included: List[str] = []
    logger.info(f"Starting stepwise selection: threshold_in={threshold_in}, threshold_out={threshold_out}")
    
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        if not excluded: 
            break

        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            try:
                model_data = X[included + [new_column]].astype(float)
                aligned_y = y.loc[model_data.index] 
                model = sm.OLS(aligned_y, sm.add_constant(model_data)).fit()
                new_pval[new_column] = model.pvalues[new_column]
            except Exception as e:
                logger.warning(f"Could not calculate p-value for {new_column}: {e}")
                new_pval[new_column] = float('inf') 

        if not new_pval.empty and new_pval.min() < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                logger.info(f"Stepwise Add: Feature '{best_feature}', p-value {new_pval.min():.4f}")

        if not included: 
            if not changed: break 
            else: continue 

        try:
            model_data_bwd = X[included].astype(float)
            aligned_y_bwd = y.loc[model_data_bwd.index]
            model = sm.OLS(aligned_y_bwd, sm.add_constant(model_data_bwd)).fit()
            pvalues = model.pvalues.iloc[1:] 
            if not pvalues.empty and pvalues.max() > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                if verbose:
                    logger.info(f"Stepwise Drop: Feature '{worst_feature}', p-value {pvalues.max():.4f}")
        except Exception as e:
            logger.warning(f"Error during backward step with features {included}: {e}")
            
        if not changed:
            break
            
    if not included:
        logger.warning("Stepwise selection resulted in no features. This may indicate issues with data or thresholds.")
    else:
        logger.info(f"Stepwise selection completed. Selected features ({len(included)}): {included}")
    return included

def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_config: Dict[str, Any] 
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, List[str]]:
    """Trains a linear regression model, possibly with stepwise selection."""
    step_cfg = model_config.get("stepwise", {})
    use_stepwise = step_cfg.get("enabled", True) 

    selected_features: List[str] = []

    if use_stepwise:
        logger.info("Performing stepwise feature selection for linear regression.")
        selected_features = stepwise_selection(
            X_train,
            y_train,
            threshold_in=step_cfg.get("threshold_in", 0.05),
            threshold_out=step_cfg.get("threshold_out", 0.1),
            verbose=step_cfg.get("verbose", True)
        )
        if not selected_features:
            logger.warning("Stepwise selection resulted in no features. Fitting on all available features as a fallback.")
            selected_features = list(X_train.columns) 
            if not selected_features: 
                 raise ValueError("No features available for training after stepwise selection and fallback.")
    else:
        logger.info("Skipping stepwise selection. Using all provided features for linear regression.")
        selected_features = list(X_train.columns)
        if not selected_features:
             raise ValueError("No features available for training (stepwise selection disabled).")
    
    X_train_selected = X_train[selected_features].astype(float)
    y_train_aligned = y_train.loc[X_train_selected.index]

    X_train_const = sm.add_constant(X_train_selected)
    model = sm.OLS(y_train_aligned, X_train_const).fit()
    
    logger.info(f"Linear regression fitted using {len(selected_features)} features.")
    logger.debug(f"Model summary:\n{model.summary()}")
    return model, selected_features

# Removed the old evaluate_regression function. It's now in evaluator_regression.py

def save_model_artifacts(
    model: Any,
    selected_features: List[str],
    metrics: Dict[str, float], # These are raw metrics from the evaluator
    config: Dict[str, Any]
) -> None:
    """Save the trained model, selected features, and evaluation metrics based on config."""
    model_cfg = config["model"]
    art_cfg = config["artifacts"]
    active_model_type = model_cfg["active"] 
    
    model_save_path = model_cfg[active_model_type]["save_path"]
    model_dir = os.path.dirname(model_save_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Trained model '{active_model_type}' saved to: {model_save_path}")

    selected_features_path_key = model_cfg[active_model_type].get("selected_features_path")
    if selected_features_path_key:
         features_path = selected_features_path_key
    else: 
        base, ext = os.path.splitext(model_save_path)
        features_path = f"{base}_selected_features.json"
        logger.info(f"No explicit 'selected_features_path' in config for '{active_model_type}'. Saving to derived path: {features_path}")

    features_dir = os.path.dirname(features_path)
    if features_dir and not os.path.exists(features_dir):
        os.makedirs(features_dir, exist_ok=True)
    with open(features_path, "w") as f:
        json.dump({"selected_features": selected_features, "count": len(selected_features)}, f, indent=4)
    logger.info(f"Selected features list saved to: {features_path}")

    metrics_save_path = art_cfg["metrics_path"]
    metrics_dir = os.path.dirname(metrics_save_path)
    if metrics_dir and not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)
        
    # Round metrics before saving for consistent JSON output
    # (or save raw metrics if preferred, evaluator already logs rounded ones)
    # from src.evaluation.evaluator_regression import round_metrics_dict # Import if needed here
    # rounded_metrics_to_save = round_metrics_dict(metrics) 
    # For now, saving the raw metrics dict returned by the evaluator
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4) 
    logger.info(f"Evaluation metrics saved to: {metrics_save_path}")


def main_modeling(config_path: str = "config.yaml") -> None:
    """Run the modeling pipeline."""
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logging.critical(f"CRITICAL: Configuration file '{config_path}' not found. Modeling cannot start.")
        return
    except yaml.YAMLError as e:
        logging.critical(f"CRITICAL: Error parsing YAML in '{config_path}': {e}. Modeling cannot start.")
        return

    global logger 
    logger = get_logger(config.get("logging", {}))
    
    try:
        validate_modeling_config(config)
    except KeyError as e:
        logger.critical(f"Configuration validation failed: {e}. Exiting.")
        return

    art_cfg = config["artifacts"]
    model_section_cfg = config["model"] 
    split_cfg = config["data_split"]
    target_column_name = config["target"]
    data_source_cfg = config["data_source"]

    features_file_name = art_cfg.get("engineered_features_filename", "features.csv")
    features_input_path = os.path.join(art_cfg["processed_dir"], features_file_name)
    logger.info(f"Loading features (X) from: {features_input_path}")
    try:
        X_df = pd.read_csv(features_input_path)
    except FileNotFoundError:
        logger.critical(f"Features file not found at '{features_input_path}'. Exiting.")
        return
    except Exception as e:
        logger.critical(f"Error loading features from '{features_input_path}': {e}. Exiting.")
        return
    logger.info(f"Features (X) loaded. Shape: {X_df.shape}")

    target_data_path = data_source_cfg["processed_path"]
    logger.info(f"Loading data for target variable '{target_column_name}' from: {target_data_path}")
    try:
        target_df = pd.read_csv(target_data_path)
    except FileNotFoundError:
        logger.critical(f"Target data file not found at '{target_data_path}'. Exiting.")
        return
    except Exception as e:
        logger.critical(f"Error loading target data from '{target_data_path}': {e}. Exiting.")
        return

    if target_column_name not in target_df.columns:
        logger.critical(f"Target column '{target_column_name}' not found in '{target_data_path}'. Exiting.")
        return
    y_series = target_df[target_column_name]
    logger.info(f"Target variable (y) '{target_column_name}' loaded. Length: {len(y_series)}")

    if not X_df.index.equals(target_df.index):
        logger.warning("Indices of X (from features.csv) and target_df (from preprocessed.csv) do not match.")
        min_len = min(len(X_df), len(y_series))
        if len(X_df) != len(y_series):
             logger.warning(f"X_df length ({len(X_df)}) and y_series length ({len(y_series)}) differ. Truncating to min_len: {min_len}")
        X_df = X_df.iloc[:min_len].reset_index(drop=True) # Reset index after slicing
        y_series = y_series.iloc[:min_len].reset_index(drop=True) # Reset index after slicing
        logger.info("Applied basic alignment by slicing to minimum length and resetting index. Ensure data order is consistent.")


    logger.info(f"Final X shape: {X_df.shape}, Final y length: {len(y_series)}")
    if X_df.empty or y_series.empty:
        logger.critical("X or y is empty after loading and alignment attempts. Exiting.")
        return

    logger.info(f"Splitting data: test_size={split_cfg['test_size']}, random_state={split_cfg['random_state']}")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series,
            test_size=split_cfg.get("test_size", 0.2),
            random_state=split_cfg.get("random_state", 42)
        )
    except ValueError as e:
        logger.critical(f"Error during train_test_split: {e}. Check data shapes and consistency. X: {X_df.shape}, y: {y_series.shape}")
        return
        
    logger.info(f"Data split complete: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

    active_model_type = model_section_cfg["active"]
    model_params_config = model_section_cfg.get(active_model_type, {})

    trained_model = None
    final_selected_features: List[str] = []

    if active_model_type == "linear_regression":
        try:
            trained_model, final_selected_features = train_linear_regression(X_train, y_train, model_params_config)
        except Exception as e:
            logger.critical(f"Error during linear regression training: {e}")
            return
    else:
        logger.error(f"Model type '{active_model_type}' is not implemented in this script.")
        return 

    if trained_model is None or not final_selected_features:
        logger.critical("Model training failed or no features were selected. Exiting.")
        return

    # --- Evaluate Model using the new evaluator ---
    logger.info(f"Evaluating '{active_model_type}' model on test set...")
    # The evaluate_statsmodels_model function already logs rounded metrics.
    # It returns the raw (unrounded) metrics.
    evaluation_metrics = evaluate_statsmodels_model(
        model=trained_model,
        X_eval=X_test,
        y_eval=y_test,
        selected_features=final_selected_features,
        split_label="test" 
    )
    if not evaluation_metrics or all(pd.isna(v) for v in evaluation_metrics.values()):
        logger.error("Evaluation returned no metrics or all NaN. Check evaluator logs.")
        # Decide if to proceed with saving NaN metrics or exit
    
    logger.info("Saving model artifacts...")
    save_model_artifacts(trained_model, final_selected_features, evaluation_metrics, config)
    
    logger.info(f"Modeling pipeline for '{active_model_type}' completed successfully.")
    # Log final raw metrics that were saved
    logger.info(f"Final saved metrics (raw): {evaluation_metrics}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    
    config_file_path = "config.yaml" 
    
    if not os.path.exists(config_file_path):
        logging.critical(f"CRITICAL: Main - Configuration file '{config_file_path}' not found. Modeling cannot start.")
    else:
        main_modeling(config_path=config_file_path)

