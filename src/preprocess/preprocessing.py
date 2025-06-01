import os
import logging
import yaml
import pandas as pd
import joblib 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split # <<<--- ADD THIS IMPORT
from typing import Dict, Any, Tuple, Optional, List

# Module-level logger - this will be initialized by main_preprocessing
logger = logging.getLogger(__name__)

# ... (get_logger, load_config, validate_preprocessing_config, drop_columns, remove_outliers_iqr, scale_columns, save_data_and_artifact functions remain the same) ...
# Ensure these functions are present above main_preprocessing

def get_logger(logging_config: Dict[str, Any]) -> logging.Logger:
    """Sets up and returns a logger based on the provided configuration."""
    log_file = logging_config.get("log_file", "logs/preprocessing.log")
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
    _logger = logging.getLogger(__name__) # Use module logger
    _logger.info(f"Loading configuration from: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    _logger.info("Configuration loaded successfully.")
    return config

def validate_preprocessing_config(config: Dict[str, Any], logger_param: Optional[logging.Logger] = None) -> None:
    effective_logger = logger_param if logger_param else logger
    effective_logger.info("Validating preprocessing configuration.")
    required_top_keys = ["preprocessing", "artifacts", "data_source", "logging"]
    for key in required_top_keys:
        if key not in config:
            raise KeyError(f"Missing required top-level config key: '{key}'")

    if "raw_path" not in config["data_source"]:
        raise KeyError("Missing 'raw_path' in 'data_source' config for loading raw data.")
    if "processed_path" not in config["data_source"]:
        raise KeyError("Missing 'processed_path' in 'data_source' config.")
    
    pre_cfg = config["preprocessing"]
    if "drop_columns" not in pre_cfg:
        effective_logger.warning("Missing 'drop_columns' in 'preprocessing' config.")
    if "outlier_removal" not in pre_cfg:
        effective_logger.warning("Missing 'outlier_removal' in 'preprocessing' config.")
    else:
        if pre_cfg["outlier_removal"].get("enabled") and not pre_cfg["outlier_removal"].get("features"):
            effective_logger.warning("Outlier removal enabled but no features specified.")
    if "scale" not in pre_cfg:
        effective_logger.warning("Missing 'scale' in 'preprocessing' config.")
    else:
        if not pre_cfg["scale"].get("columns"):
            effective_logger.warning("Scaling config present but no columns specified for scaling.")
    effective_logger.info("Preprocessing configuration validation successful.")

def drop_columns(df: pd.DataFrame, columns_to_drop: List[str], logger_param: Optional[logging.Logger] = None) -> pd.DataFrame:
    effective_logger = logger_param if logger_param else logger
    if not columns_to_drop:
        effective_logger.info("No columns specified for dropping. Skipping.")
        return df
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if not existing_columns_to_drop:
        effective_logger.warning(f"None of specified columns to drop exist: {columns_to_drop}")
        return df
    effective_logger.info(f"Dropping columns: {existing_columns_to_drop}")
    df_dropped = df.drop(columns=existing_columns_to_drop, errors="ignore")
    effective_logger.info(f"Shape after dropping columns: {df_dropped.shape}")
    return df_dropped

def remove_outliers_iqr(df: pd.DataFrame, features: List[str], iqr_multiplier: float, logger_param: Optional[logging.Logger] = None) -> pd.DataFrame:
    effective_logger = logger_param if logger_param else logger
    if not features:
        effective_logger.info("No features for outlier removal. Skipping.")
        return df
    df_clean = df.copy()
    effective_logger.info(f"Outlier removal for: {features}, IQR multiplier: {iqr_multiplier}")
    initial_rows = df_clean.shape[0]
    for feature in features:
        if feature not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[feature]):
            effective_logger.warning(f"Feature '{feature}' not numeric or not found. Skipping.")
            continue
        Q1, Q3 = df_clean[feature].quantile(0.25), df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            effective_logger.info(f"IQR is 0 for '{feature}'. No outliers removed.")
            continue
        lower_b, upper_b = Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR
        rows_before = df_clean.shape[0]
        df_clean = df_clean[(df_clean[feature] >= lower_b) & (df_clean[feature] <= upper_b)]
        effective_logger.info(f"Outliers on '{feature}': {rows_before - df_clean.shape[0]} rows removed. Range: [{lower_b:.2f}, {upper_b:.2f}]")
    effective_logger.info(f"Total rows removed by outlier processing: {initial_rows - df_clean.shape[0]}")
    return df_clean

def scale_columns(df: pd.DataFrame, columns_to_scale: List[str], method: str, logger_param: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, Optional[Any]]:
    effective_logger = logger_param if logger_param else logger
    if not columns_to_scale:
        effective_logger.info("No columns for scaling. Skipping.")
        return df, None
    df_scaled, scaler = df.copy(), None
    actual_cols = [c for c in columns_to_scale if c in df_scaled.columns and pd.api.types.is_numeric_dtype(df_scaled[c])]
    for c in columns_to_scale: 
        if c not in actual_cols: effective_logger.warning(f"Column '{c}' for scaling not numeric or not found.")
    if not actual_cols:
        effective_logger.warning("No valid numeric columns for scaling. Skipping.")
        return df, None
    effective_logger.info(f"Scaling columns {actual_cols} with '{method}'.")
    if method == "minmax": scaler = MinMaxScaler()
    elif method == "standard": scaler = StandardScaler()
    else:
        effective_logger.error(f"Unsupported scaling: {method}. Choose 'minmax' or 'standard'.")
        return df, None
    df_scaled[actual_cols] = scaler.fit_transform(df_scaled[actual_cols])
    effective_logger.info("Scaling complete.")
    return df_scaled, scaler

def save_data_and_artifact(df: pd.DataFrame, data_path: str, artifact: Optional[Any], artifact_path: Optional[str], logger_param: Optional[logging.Logger] = None) -> None:
    effective_logger = logger_param if logger_param else logger
    data_dir = os.path.dirname(data_path)
    if data_dir and not os.path.exists(data_dir): os.makedirs(data_dir, exist_ok=True)
    df.to_csv(data_path, index=False)
    effective_logger.info(f"Processed data saved to: {data_path} | Shape: {df.shape}")
    if artifact and artifact_path:
        art_dir = os.path.dirname(artifact_path)
        if art_dir and not os.path.exists(art_dir): os.makedirs(art_dir, exist_ok=True)
        joblib.dump(artifact, artifact_path)
        effective_logger.info(f"Scaler saved to: {artifact_path}")
    elif artifact_path and not artifact: effective_logger.warning(f"Path '{artifact_path}' provided, but no scaler to save.")
    elif artifact and not artifact_path: effective_logger.warning("Scaler generated, but no path in config. Not saved.")

def main_preprocessing(config_path: str = "config.yaml") -> None:
    """Main function to run the data preprocessing pipeline."""
    try:
        config = load_config(config_path) 
    except FileNotFoundError:
        logging.error(f"CRITICAL: Config file '{config_path}' not found. Exiting.") 
        return
    except yaml.YAMLError as e:
        logging.error(f"CRITICAL: Error parsing YAML config '{config_path}': {e}. Exiting.")
        return

    global logger 
    logger = get_logger(config.get("logging", {})) 
    
    try:
        validate_preprocessing_config(config, logger_param=logger)
    except KeyError as e:
        logger.error(f"Config validation failed: {e}. Exiting.")
        return

    data_source_cfg = config.get("data_source", {})
    pre_cfg = config.get("preprocessing", {})
    art_cfg = config.get("artifacts", {})

    # --- Original Data Loading ---
    raw_data_path = data_source_cfg.get("raw_path")
    data_type = data_source_cfg.get("type", "csv").lower()
    delimiter = data_source_cfg.get("delimiter", ",")
    header = data_source_cfg.get("header", 0)
    encoding = data_source_cfg.get("encoding", "utf-8")

    logger.info(f"Attempting to load full raw data from: {raw_data_path} (type: {data_type})")
    try:
        if data_type == "csv":
            full_raw_df = pd.read_csv(raw_data_path, delimiter=delimiter, header=header, encoding=encoding)
        elif data_type == "excel":
            sheet_name = data_source_cfg.get("sheet_name", 0)
            full_raw_df = pd.read_excel(raw_data_path, sheet_name=sheet_name, header=header)
        else:
            logger.error(f"Unsupported data type '{data_type}'. Exiting.")
            return
        logger.info(f"Full raw data loaded successfully. Shape: {full_raw_df.shape}")
    except FileNotFoundError:
        logger.error(f"Raw data file not found at '{raw_data_path}'. Exiting.")
        return
    except Exception as e:
        logger.error(f"Error loading raw data from '{raw_data_path}': {e}. Exiting.")
        return

    if full_raw_df.empty:
        logger.error("Loaded raw data is empty. Exiting.")
        return

    # --- NEW: Split off inference holdout set ---
    df_for_pipeline = full_raw_df # Default to using all data if no holdout split
    holdout_path = data_source_cfg.get("inference_holdout_path")
    holdout_size = data_source_cfg.get("inference_holdout_size")
    
    # Use random_state from data_split for consistency, or a new one
    split_cfg = config.get("data_split", {})
    split_random_state = split_cfg.get("random_state", 42) 

    if holdout_path and isinstance(holdout_size, float) and 0 < holdout_size < 1:
        logger.info(f"Splitting off {holdout_size*100:.1f}% of raw data for inference holdout set.")
        try:
            # Stratification target column (optional, if meaningful for this initial split)
            # For now, not stratifying this initial split.
            df_for_pipeline, df_holdout = train_test_split(
                full_raw_df, 
                test_size=holdout_size, 
                random_state=split_random_state
            )
            
            if not df_holdout.empty:
                 holdout_dir = os.path.dirname(holdout_path)
                 if holdout_dir and not os.path.exists(holdout_dir):
                     os.makedirs(holdout_dir, exist_ok=True)
                 df_holdout.to_csv(holdout_path, index=False)
                 logger.info(f"Inference holdout data saved to: {holdout_path} | Shape: {df_holdout.shape}")
                 logger.info(f"Proceeding with main pipeline data. Shape: {df_for_pipeline.shape}")
            else:
                 logger.warning("Holdout DataFrame is empty after split; proceeding with full dataset for pipeline.")
                 df_for_pipeline = full_raw_df # Fallback
        except Exception as e:
            logger.error(f"Failed to split raw data for holdout: {e}. Proceeding with full dataset for preprocessing.")
            df_for_pipeline = full_raw_df # Fallback
    else:
        logger.info("Inference holdout set not configured or size is invalid. Using full raw dataset for pipeline.")
        # df_for_pipeline is already full_raw_df

    if df_for_pipeline.empty:
        logger.error("Data for pipeline is empty after attempting holdout split. Exiting.")
        return
        
    # --- Proceed with df_for_pipeline for all subsequent preprocessing steps ---
    current_df = df_for_pipeline.copy()
    
    logger.info("Starting data validation (placeholder step)...") # Placeholder
    logger.info("Data validation (placeholder step) complete.")
    
    # Step 1: Drop Columns
    columns_to_drop = pre_cfg.get("drop_columns", [])
    current_df = drop_columns(current_df, columns_to_drop, logger_param=logger)

    # Step 2: Outlier Removal
    outlier_cfg = pre_cfg.get("outlier_removal", {})
    if outlier_cfg.get("enabled", False):
        outlier_features = outlier_cfg.get("features", [])
        iqr_multiplier = outlier_cfg.get("iqr_multiplier", 1.5)
        if outlier_features:
             current_df = remove_outliers_iqr(current_df, outlier_features, iqr_multiplier, logger_param=logger)
        else:
            logger.info("Outlier removal enabled but no features specified. Skipping.")
    else:
        logger.info("Outlier removal is disabled in the configuration.")

    # Step 3: Scale Columns
    scale_cfg = pre_cfg.get("scale", {})
    scaler_artifact = None
    if scale_cfg and scale_cfg.get("columns"):
        columns_to_scale = scale_cfg.get("columns", [])
        scaling_method = scale_cfg.get("method", "minmax")
        current_df, scaler_artifact = scale_columns(current_df, columns_to_scale, scaling_method, logger_param=logger)
    else:
        logger.info("Scaling is disabled or no columns specified for scaling.")

    # Save Processed Data (main pipeline portion) and Scaler Artifact
    processed_data_output_path = data_source_cfg.get("processed_path") # This is for the main pipeline data
    scaler_artifact_path = art_cfg.get("preprocessing_pipeline")

    if not processed_data_output_path:
        logger.error("Missing 'processed_path' in config for main pipeline data. Cannot save.")
    else:
        save_data_and_artifact(current_df, processed_data_output_path, scaler_artifact, scaler_artifact_path, logger_param=logger)
    
    logger.info("Preprocessing pipeline finished for main data.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    config_file_path = "config.yaml" 
    if not os.path.exists(config_file_path):
        logging.error(f"CRITICAL: Main - Config file '{config_file_path}' not found. Preprocessing cannot start.")
    else:
        main_preprocessing(config_path=config_file_path)

