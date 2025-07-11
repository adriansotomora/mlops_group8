"""
Data preprocessing pipeline for the MLOps project.

This module handles data cleaning, outlier removal, scaling, and preparation
for the feature engineering stage.
"""
import os
import logging
import sys 
import yaml
import pandas as pd
import joblib 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, Optional, List

# Import get_raw_data from your data_loader module
try:
    from src.data_load.data_loader import get_raw_data
except ModuleNotFoundError:
    logging.basicConfig(level=logging.CRITICAL, format="%(levelname)s: %(message)s")
    logging.critical("CRITICAL ERROR: Could not import get_raw_data from src.data_load.data_loader. Ensure path is correct.")
    sys.exit(1)

# Module-level logger, configured in main_preprocessing
logger = logging.getLogger(__name__)

def get_logger(logging_config: Dict[str, Any], default_log_file: str = "logs/preprocessing.log") -> logging.Logger:
    """Set up and return a logger based on the provided configuration."""
    log_file = logging_config.get("log_file", default_log_file)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_format_str = logging_config.get("format", "%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s")
    date_format_str = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_level_name = logging_config.get("level", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format=log_format_str,
        datefmt=date_format_str,
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(log_level)
    return module_logger

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load the YAML configuration file."""
    _current_logger = logging.getLogger(__name__) 
    _current_logger.info(f"Loading configuration from: {path}")
    if not os.path.isfile(path):
        _current_logger.error(f"Configuration file not found: {path}")
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    _current_logger.info("Configuration loaded successfully.")
    return config

def validate_preprocessing_config(config: Dict[str, Any], logger_param: Optional[logging.Logger] = None) -> None:
    """Validate essential keys in the preprocessing configuration."""
    effective_logger = logger_param if logger_param else logger
    effective_logger.info("Validating preprocessing configuration.")
    required_top_keys = ["preprocessing", "artifacts", "data_source", "logging"]
    for key in required_top_keys:
        if key not in config:
            raise KeyError(f"Missing required top-level config key: '{key}'")

    if "raw_path" not in config["data_source"]: # data_loader will use this, so it's still essential
        raise KeyError("Missing 'raw_path' in 'data_source' config.")
    if "processed_path" not in config["data_source"]:
        raise KeyError("Missing 'processed_path' in 'data_source' config.")
    
    pre_cfg = config.get("preprocessing", {}) 
    if "drop_columns" not in pre_cfg:
        effective_logger.warning("Config: 'preprocessing.drop_columns' not found.")
    if "outlier_removal" not in pre_cfg:
        effective_logger.warning("Config: 'preprocessing.outlier_removal' not found.")
    elif pre_cfg["outlier_removal"].get("enabled") and not pre_cfg["outlier_removal"].get("features"):
        effective_logger.warning("Config: Outlier removal enabled but no features listed.")
    if "scale" not in pre_cfg:
        effective_logger.warning("Config: 'preprocessing.scale' not found.")
    elif not pre_cfg["scale"].get("columns"):
        effective_logger.warning("Config: Scaling configured but no columns listed.")
    effective_logger.info("Preprocessing configuration validation successful.")

def drop_columns(df: pd.DataFrame, columns_to_drop: List[str], logger_param: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Drop specified columns from the DataFrame."""
    effective_logger = logger_param if logger_param else logger
    if not columns_to_drop:
        effective_logger.info("No columns specified for dropping. Skipping.")
        return df
    
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if not existing_columns_to_drop:
        effective_logger.warning(f"None of columns to drop exist in DataFrame: {columns_to_drop}")
        return df

    effective_logger.info(f"Dropping columns: {existing_columns_to_drop}")
    df_dropped = df.drop(columns=existing_columns_to_drop, errors="ignore")
    effective_logger.info(f"Shape after dropping columns: {df_dropped.shape}")
    return df_dropped

def remove_outliers_iqr(df: pd.DataFrame, features: List[str], iqr_multiplier: float, logger_param: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Remove outliers from specified numeric features using the IQR method."""
    effective_logger = logger_param if logger_param else logger
    if not features:
        effective_logger.info("No features specified for outlier removal. Skipping.")
        return df

    df_clean = df.copy()
    effective_logger.info(f"Starting outlier removal for: {features} (IQR multiplier: {iqr_multiplier})")
    initial_rows = df_clean.shape[0]

    for feature in features:
        if feature not in df_clean.columns:
            effective_logger.warning(f"Outlier feature '{feature}' not found. Skipping.")
            continue
        if not pd.api.types.is_numeric_dtype(df_clean[feature]):
            effective_logger.warning(f"Outlier feature '{feature}' is not numeric. Skipping.")
            continue
        
        Q1 = df_clean[feature].quantile(0.25)
        Q3 = df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            effective_logger.info(f"IQR is 0 for '{feature}'. No outliers removed for this feature.")
            continue

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        rows_before_feature_filter = df_clean.shape[0]
        df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]
        rows_after_feature_filter = df_clean.shape[0]
        if rows_before_feature_filter > rows_after_feature_filter:
            effective_logger.info(
                f"Outliers on '{feature}': {rows_before_feature_filter - rows_after_feature_filter} rows removed. Valid range: [{lower_bound:.2f}, {upper_bound:.2f}]"
            )
    
    total_rows_removed = initial_rows - df_clean.shape[0]
    effective_logger.info(f"Total rows removed by outlier processing: {total_rows_removed}. Shape after: {df_clean.shape}")
    return df_clean

def save_data_and_artifact(df: pd.DataFrame, data_path: str, artifact: Optional[Any], artifact_path: Optional[str], logger_param: Optional[logging.Logger] = None) -> None:
    """Save the processed DataFrame and the preprocessing artifact (e.g., scaler)."""
    effective_logger = logger_param if logger_param else logger
    
    data_dir = os.path.dirname(data_path)
    if data_dir and not os.path.exists(data_dir): os.makedirs(data_dir, exist_ok=True)
    df.to_csv(data_path, index=False)
    effective_logger.info(f"Processed data saved to: {data_path} | Shape: {df.shape}")

    if artifact and artifact_path:
        artifact_dir = os.path.dirname(artifact_path)
        if artifact_dir and not os.path.exists(artifact_dir): os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(artifact, artifact_path)
        effective_logger.info(f"Preprocessing artifact (scaler) saved to: {artifact_path}")
    elif artifact_path and not artifact: 
        effective_logger.warning(f"Scaler path '{artifact_path}' provided, but no scaler object to save.")
    elif artifact and not artifact_path: 
        effective_logger.warning("Scaler object generated, but no 'preprocessing_pipeline' path in config. Scaler not saved.")

def main_preprocessing(config_path: str = "config.yaml") -> None:
    """Run the data preprocessing pipeline, including holdout split."""
    try:
        config = load_config(config_path)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        logging.critical(f"Preprocessing: Failed to load config '{config_path}': {e}", exc_info=True)
        return

    global logger
    logger = get_logger(config.get("logging", {}), default_log_file="logs/preprocessing.log")

    try:
        validate_preprocessing_config(config, logger_param=logger)
    except KeyError as e:
        logger.critical(f"Config validation failed: {e}. Exiting preprocessing.", exc_info=True)
        return

    # --- Path Resolution ---
    config_dir = os.path.dirname(os.path.abspath(config_path))

    data_source_cfg = config.get("data_source", {})
    pre_cfg = config.get("preprocessing", {})
    art_cfg = config.get("artifacts", {})

    # --- Load full raw dataset USING data_loader.py ---
    logger.info(f"Attempting to load full raw data using data_loader from config: {config_path}")
    try:
        full_raw_df = get_raw_data(config_path=config_path)
        if full_raw_df is None or full_raw_df.empty:
            logger.critical("Data loading via get_raw_data returned empty or None. Exiting.")
            return
        logger.info(f"Full raw data loaded via data_loader. Shape: {full_raw_df.shape}")
        # Debug: Check liveness/valence after loading
        for col in ["liveness", "valence"]:
            if col in full_raw_df.columns:
                logger.info(f"[DEBUG] After loading: {col} dtype={full_raw_df[col].dtype}, sample={full_raw_df[col].head(5).tolist()}")
    except FileNotFoundError as e:
        logger.critical(f"Failed to load raw data (FileNotFound via data_loader): {e}. Exiting.", exc_info=True)
        return
    except ValueError as e:
        logger.critical(f"Failed to load raw data (ValueError via data_loader): {e}. Exiting.", exc_info=True)
        return
    except Exception as e:
        logger.critical(f"Unexpected error during data loading via data_loader: {e}. Exiting.", exc_info=True)
        return

    df_for_pipeline = full_raw_df
    holdout_path_relative = data_source_cfg.get("inference_holdout_path")
    holdout_size = data_source_cfg.get("inference_holdout_size")
    split_random_state = config.get("data_split", {}).get("random_state", 42)

    if holdout_path_relative and isinstance(holdout_size, float) and 0 < holdout_size < 1:
        logger.info(f"Splitting {holdout_size*100:.1f}% of raw data for inference holdout.")
        try:
            df_for_pipeline, df_holdout = train_test_split(
                full_raw_df, test_size=holdout_size, random_state=split_random_state
            )
            if not df_holdout.empty:
                holdout_path_absolute = os.path.join(config_dir, holdout_path_relative)
                holdout_dir = os.path.dirname(holdout_path_absolute)
                if holdout_dir and not os.path.exists(holdout_dir):
                    os.makedirs(holdout_dir, exist_ok=True)
                df_holdout.to_csv(holdout_path_absolute, index=False)
                logger.info(f"Inference holdout data saved to: {holdout_path_absolute} | Shape: {df_holdout.shape}")
                logger.info(f"Proceeding with main pipeline data. Shape: {df_for_pipeline.shape}")
            else:
                logger.warning("Holdout DataFrame is empty after split. Using full dataset.")
                df_for_pipeline = full_raw_df
        except Exception as e:
            logger.error(f"Failed to split/save raw data for holdout: {e}. Using full dataset.", exc_info=True)
            df_for_pipeline = full_raw_df
    else:
        logger.info("Inference holdout not configured or invalid. Using full raw dataset.")

    if df_for_pipeline.empty:
        logger.critical("Data for pipeline is empty after holdout split. Exiting.")
        return

    current_df = df_for_pipeline.copy()
    # Debug: Check liveness/valence after holdout split
    for col in ["liveness", "valence"]:
        if col in current_df.columns:
            logger.info(f"[DEBUG] After holdout split: {col} dtype={current_df[col].dtype}, sample={current_df[col].head(5).tolist()}")

    logger.info("Data validation (placeholder step)...")
    logger.info("Data validation (placeholder step) complete.")

    current_df = drop_columns(current_df, pre_cfg.get("drop_columns", []))
    # Debug: Check liveness/valence after drop_columns
    for col in ["liveness", "valence"]:
        if col in current_df.columns:
            logger.info(f"[DEBUG] After drop_columns: {col} dtype={current_df[col].dtype}, sample={current_df[col].head(5).tolist()}")

    if pre_cfg.get("outlier_removal", {}).get("enabled", False):
        current_df = remove_outliers_iqr(
            current_df,
            pre_cfg["outlier_removal"].get("features", []),
            pre_cfg["outlier_removal"].get("iqr_multiplier", 1.5)
        )
        # Debug: Check liveness/valence after outlier removal
        for col in ["liveness", "valence"]:
            if col in current_df.columns:
                logger.info(f"[DEBUG] After outlier removal: {col} dtype={current_df[col].dtype}, sample={current_df[col].head(5).tolist()}")

    # --- Build and Apply Preprocessing Pipeline ---
    logger.info("Building preprocessing pipeline (imputation + scaling)...")
    preprocessor_artifact = None
    numeric_features = pre_cfg.get("scale", {}).get("columns", [])

    if not numeric_features:
        logger.info("No numeric features to scale/impute specified in config. Skipping pipeline.")
    else:
        numeric_transformer_steps = []
        imputation_cfg = pre_cfg.get("imputation", {})
        if imputation_cfg.get("enabled", True):  # Default to enabled
            strategy = imputation_cfg.get("strategy", "mean")
            logger.info(f"Pipeline: Adding imputer with strategy '{strategy}'.")
            numeric_transformer_steps.append(('imputer', SimpleImputer(strategy=strategy)))

        scale_cfg = pre_cfg.get("scale", {})
        method = scale_cfg.get("method", "minmax")
        if method.lower() == "minmax":
            scaler = MinMaxScaler()
        elif method.lower() == "standard":
            scaler = StandardScaler()
        else:
            logger.warning(f"Unsupported scaling method '{method}'. Scaler not added.")
            scaler = None

        if scaler:
            logger.info(f"Pipeline: Adding scaler with method '{method}'.")
            numeric_transformer_steps.append(('scaler', scaler))

        numeric_pipeline = Pipeline(steps=numeric_transformer_steps)

        preprocessor_artifact = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features)
            ],
            remainder='passthrough'
        )

        logger.info("Fitting and applying preprocessing pipeline...")
        transformed = preprocessor_artifact.fit_transform(current_df)
        # Get correct column names after transformation
        try:
            feature_names = preprocessor_artifact.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn: manually combine
            num_features = [f'num__{col}' for col in numeric_features]
            passthrough = [col for col in current_df.columns if col not in numeric_features]
            feature_names = num_features + passthrough
        current_df = pd.DataFrame(transformed, columns=feature_names, index=current_df.index)
        # Remove 'num__' and 'remainder__' prefixes from column names for clarity
        def clean_col(col):
            if col.startswith('num__'):
                return col.replace('num__', '', 1)
            if col.startswith('remainder__'):
                return col.replace('remainder__', '', 1)
            return col
        current_df.columns = [clean_col(col) for col in current_df.columns]
        logger.info("Preprocessing pipeline applied. New shape: %s", current_df.shape)
        # Debug: Check liveness/valence after pipeline
        for col in ["liveness", "valence"]:
            if col in current_df.columns:
                logger.info(f"[DEBUG] After preprocessing pipeline: {col} dtype={current_df[col].dtype}, sample={current_df[col].head(5).tolist()}")
        current_df = current_df[current_df.columns]  # No column filtering, just renaming

    processed_output_path_relative = data_source_cfg.get("processed_path")
    scaler_artifact_path_relative = art_cfg.get("preprocessing_pipeline")
    if not processed_output_path_relative:
        logger.error("Config missing 'data_source.processed_path'. Cannot save processed data.")
    else:
        processed_output_path_absolute = os.path.join(config_dir, processed_output_path_relative)
        scaler_artifact_path_absolute = None
        if scaler_artifact_path_relative:
            scaler_artifact_path_absolute = os.path.join(config_dir, scaler_artifact_path_relative)
        save_data_and_artifact(current_df, processed_output_path_absolute, preprocessor_artifact, scaler_artifact_path_absolute)

    logger.info("--- Preprocessing Stage Completed ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger.info("Running preprocessing.py standalone...")
    
    config_file_path = "config.yaml" 
    if not os.path.exists(config_file_path):
        logger.critical(f"Config file '{config_file_path}' not found. Cannot run preprocessing.")
    else:
        try:
            main_preprocessing(config_path=config_file_path)
        except Exception as e:
            logger.critical(f"Error during standalone execution of preprocessing.py: {e}", exc_info=True)
