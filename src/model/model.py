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
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import joblib
import mlflow
import numpy as np
import pandas as pd
import statsmodels.api as sm
import wandb
import yaml
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.evaluation.evaluator import evaluate_statsmodels_model


logger = logging.getLogger(__name__)


def get_logger(logging_config: Dict[str, Any], config_dir: str) -> logging.Logger:
    log_file_rel = logging_config.get("log_file", "logs/modeling.log") 
    log_file = os.path.join(config_dir, log_file_rel)
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
    temp_logger = logging.getLogger(f"{__name__}.load_config")
    temp_logger.info(f"Loading configuration from: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    temp_logger.info("Configuration loaded successfully.")
    return config

def validate_modeling_config(config: Dict[str, Any]) -> None:
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

def save_model_artifacts(
    model: Any,
    selected_features: List[str],
    metrics: Dict[str, float],
    config: Dict[str, Any],
    config_dir: str
) -> None:
    model_cfg = config["model"]
    art_cfg = config["artifacts"]
    active_model_type = model_cfg["active"] 

    # --- Feature presence check ---
    features_csv_name = art_cfg.get("engineered_features_filename", "features.csv")
    features_csv_path = features_csv_name if os.path.isabs(features_csv_name) else os.path.join(config_dir, features_csv_name)
    if not os.path.exists(features_csv_path):
        logger.critical(f"Features file not found for verification: {features_csv_path}")
        raise FileNotFoundError(f"Features file not found: {features_csv_path}")
    features_df = pd.read_csv(features_csv_path)
    missing_features = [f for f in selected_features if f not in features_df.columns]
    if missing_features:
        logger.critical(
            f"Selected features not found in features.csv: {missing_features}. Aborting artifact saving."
        )
        raise ValueError(
            f"Selected features not found in features.csv: {missing_features}"
        )
    else:
        logger.info(
            f"All selected features are present in features.csv. Selected features: {selected_features}"
        )

    model_save_path_rel = model_cfg[active_model_type]["save_path"]
    model_save_path = os.path.join(config_dir, model_save_path_rel)
    model_dir = os.path.dirname(model_save_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Trained model '{active_model_type}' saved to: {model_save_path}")

    selected_features_path_key = model_cfg[active_model_type].get("selected_features_path")
    if selected_features_path_key:
         features_path_rel = selected_features_path_key
         features_path = os.path.join(config_dir, features_path_rel)
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

    metrics_save_path_rel = art_cfg["metrics_path"]
    metrics_save_path = os.path.join(config_dir, metrics_save_path_rel)
    metrics_dir = os.path.dirname(metrics_save_path)
    if metrics_dir and not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4) 
    logger.info(f"Evaluation metrics saved to: {metrics_save_path}")

def main_modeling(config_path: str = "config.yaml") -> None:
    try:
        config_abs_path = os.path.abspath(config_path)
        config_dir = os.path.dirname(config_abs_path)
        config = load_config(config_abs_path)
    except FileNotFoundError:
        logging.critical(f"CRITICAL: Configuration file '{config_path}' not found. Modeling cannot start.")
        return
    except yaml.YAMLError as e:
        logging.critical(f"CRITICAL: Error parsing YAML in '{config_path}': {e}. Modeling cannot start.")
        return

    global logger 
    logger = get_logger(config.get("logging", {}), config_dir)
    
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

    # --- Load features and target directly from features.csv and preprocessed data ---
    features_csv_name = art_cfg.get("engineered_features_filename", "features.csv")
    features_csv_path = features_csv_name if os.path.isabs(features_csv_name) else os.path.join(config_dir, features_csv_name)
    if not os.path.exists(features_csv_path):
        logger.critical(f"Features file not found: {features_csv_path}")
        return
    X_df = pd.read_csv(features_csv_path)
    logger.info(f"Loaded features from {features_csv_path} with shape {X_df.shape}")

    # Use only the columns present in features.csv for feature selection and model training
    # Do not expect or generate any engineered feature names (e.g., audio_poly__...)
    # Stepwise selection and model training will only use these columns

    # Load target variable from preprocessed data
    preprocessed_data_path = os.path.join(config_dir, data_source_cfg["processed_path"])
    df = pd.read_csv(preprocessed_data_path)
    if target_column_name not in df.columns:
        logger.critical(f"Target column '{target_column_name}' not found in preprocessed data. Exiting.")
        return
    y_series = df[target_column_name]
    # Align indices if needed
    if len(X_df) != len(y_series):
        logger.warning(f"Row count mismatch: features ({len(X_df)}) vs target ({len(y_series)}). Aligning by index.")
        min_len = min(len(X_df), len(y_series))
        X_df = X_df.iloc[:min_len].reset_index(drop=True)
        y_series = y_series.iloc[:min_len].reset_index(drop=True)
    logger.info(f"Final features shape: {X_df.shape}, target length: {len(y_series)}")

    # Proceed to train/test split and modeling as before, using X_df and y_series
    # REMOVE all pipeline-related code above this point (do not run build_feature_pipeline, do not use X_transformed_df, etc.)

    logger.info(f"Splitting data: test_size={split_cfg['test_size']}, random_state={split_cfg['random_state']}")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_series,
            test_size=split_cfg.get("test_size", 0.2),
            random_state=split_cfg.get("random_state", 42)
        )
        print("Type of X_train:", type(X_train))
        print("Type of X_test:", type(X_test))
        print("Type of y_train:", type(y_train))
        print("Type of y_test:", type(y_test))    
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

    logger.info(f"Evaluating '{active_model_type}' model on test set...")
    evaluation_metrics = evaluate_statsmodels_model(
        model=trained_model,
        X_eval=X_test,
        y_eval=y_test,
        selected_features=final_selected_features,
        split_label="test" 
    )
    if not evaluation_metrics or all(pd.isna(v) for v in evaluation_metrics.values()):
        logger.error("Evaluation returned no metrics or all NaN. Check evaluator logs.")
    
    logger.info("Saving model artifacts...")
    save_model_artifacts(trained_model, final_selected_features, evaluation_metrics, config, config_dir)
    
    logger.info(f"Modeling pipeline for '{active_model_type}' completed successfully.")
    logger.info(f"Final saved metrics (raw): {evaluation_metrics}")

    # MLflow and W&B Integration
    logger.info("Logging parameters and metrics to MLflow and W&B.")

    # Log parameters
    mlflow.log_params(config['data_split'])
    mlflow.log_params(config['model'][active_model_type])

    # Log metrics
    mlflow.log_metrics(evaluation_metrics)

    # Log model
    mlflow.statsmodels.log_model(
        statsmodels_model=trained_model,
        artifact_path="model"
    )

    # W&B Logging
    wandb.log(config['data_split'])
    wandb.log(config['model'][active_model_type])
    wandb.log(evaluation_metrics)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    config_file_path = "config.yaml"
    if not os.path.exists(config_file_path):
        logging.critical(f"CRITICAL: Main - Configuration file '{config_file_path}' not found. Modeling cannot start.")
    else:
        main_modeling(config_path=config_file_path)