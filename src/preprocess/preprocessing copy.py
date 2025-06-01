import os
import logging
import yaml
import pandas as pd
import joblib # For saving the scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Any, Tuple, Optional, List

# Module-level logger - this will be initialized by main_preprocessing
logger = logging.getLogger(__name__)

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

    # Configure root logger if not already configured
    current_handlers = logging.getLogger().handlers
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in current_handlers) and \
       not any(isinstance(h, logging.StreamHandler) for h in current_handlers):
        
        # Clear existing handlers from root to avoid duplication if re-running in same session or if basicConfig was called
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)

        logging.basicConfig( #This basicConfig might not be strictly necessary if we add handlers directly
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # General format
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


    # Return logger for the current module
    module_logger = logging.getLogger(__name__)
    module_logger.setLevel(log_level) 
    return module_logger


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    # Use the effective logger (module-level, as get_logger might not have been called yet if this is imported)
    _logger = logging.getLogger(__name__)
    _logger.info(f"Loading configuration from: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    _logger.info("Configuration loaded successfully.")
    return config

def validate_preprocessing_config(config: Dict[str, Any], logger_param: Optional[logging.Logger] = None) -> None:
    """Validates the presence of essential keys in the preprocessing configuration."""
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
        effective_logger.warning("Missing 'drop_columns' in 'preprocessing' config. No columns will be dropped by default.")
    if "outlier_removal" not in pre_cfg:
        effective_logger.warning("Missing 'outlier_removal' in 'preprocessing' config. Outlier removal will be disabled.")
    else:
        if pre_cfg["outlier_removal"].get("enabled") and not pre_cfg["outlier_removal"].get("features"):
            effective_logger.warning("Outlier removal enabled but no features specified. No outliers will be removed.")
    if "scale" not in pre_cfg:
        effective_logger.warning("Missing 'scale' in 'preprocessing' config. Scaling will be disabled.")
    else:
        if not pre_cfg["scale"].get("columns"):
            effective_logger.warning("Scaling config present but no columns specified for scaling.")

    effective_logger.info("Preprocessing configuration validation successful.")


def drop_columns(df: pd.DataFrame, columns_to_drop: List[str], logger_param: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Drops specified columns from the DataFrame."""
    effective_logger = logger_param if logger_param else logger
    
    if not columns_to_drop:
        effective_logger.info("No columns specified for dropping. Skipping drop_columns step.")
        return df
    
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if not existing_columns_to_drop:
        effective_logger.warning(f"None of the specified columns to drop exist in DataFrame: {columns_to_drop}")
        return df

    effective_logger.info(f"Dropping columns: {existing_columns_to_drop}")
    df_dropped = df.drop(columns=existing_columns_to_drop, errors="ignore")
    effective_logger.info(f"Shape after dropping columns: {df_dropped.shape}")
    return df_dropped


def remove_outliers_iqr(df: pd.DataFrame, features: List[str], iqr_multiplier: float, logger_param: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Removes outliers from specified features using the IQR method."""
    effective_logger = logger_param if logger_param else logger

    if not features:
        effective_logger.info("No features specified for outlier removal. Skipping.")
        return df

    df_clean = df.copy()
    effective_logger.info(f"Starting outlier removal for features: {features} with IQR multiplier: {iqr_multiplier}")
    initial_rows = df_clean.shape[0]

    for feature in features:
        if feature not in df_clean.columns:
            effective_logger.warning(f"Feature '{feature}' for outlier removal not found in DataFrame. Skipping this feature.")
            continue
        if not pd.api.types.is_numeric_dtype(df_clean[feature]):
            effective_logger.warning(f"Feature '{feature}' is not numeric. Skipping outlier removal for this feature.")
            continue
        
        Q1 = df_clean[feature].quantile(0.25)
        Q3 = df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            effective_logger.info(f"IQR is 0 for feature '{feature}'. No outliers will be removed for this feature based on IQR.")
            continue

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        rows_before_feature_filter = df_clean.shape[0]
        df_clean = df_clean[(df_clean[feature] >= lower_bound) & (df_clean[feature] <= upper_bound)]
        rows_after_feature_filter = df_clean.shape[0]
        effective_logger.info(
            f"Outlier removal on '{feature}': "
            f"{rows_before_feature_filter - rows_after_feature_filter} rows removed. "
            f"Range: [{lower_bound:.2f}, {upper_bound:.2f}]"
        )
    
    total_rows_removed = initial_rows - df_clean.shape[0]
    effective_logger.info(f"Total rows removed after outlier processing: {total_rows_removed}")
    effective_logger.info(f"Shape after outlier removal: {df_clean.shape}")
    return df_clean


def scale_columns(df: pd.DataFrame, columns_to_scale: List[str], method: str, logger_param: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, Optional[Any]]:
    """Scales specified columns using the chosen method (minmax or standard)."""
    effective_logger = logger_param if logger_param else logger

    if not columns_to_scale:
        effective_logger.info("No columns specified for scaling. Skipping scaling step.")
        return df, None

    df_scaled = df.copy()
    scaler = None
    
    actual_columns_to_scale = []
    for col in columns_to_scale:
        if col not in df_scaled.columns:
            effective_logger.warning(f"Column '{col}' for scaling not found in DataFrame. Skipping this column.")
            continue
        if not pd.api.types.is_numeric_dtype(df_scaled[col]):
            effective_logger.warning(f"Column '{col}' is not numeric. Skipping scaling for this column.")
            continue
        actual_columns_to_scale.append(col)

    if not actual_columns_to_scale:
        effective_logger.warning("No valid numeric columns found for scaling after checks. Skipping scaling.")
        return df, None

    effective_logger.info(f"Scaling columns {actual_columns_to_scale} with '{method}' method.")
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        effective_logger.error(f"Unsupported scaling method: {method}. Choose 'minmax' or 'standard'.")
        return df, None

    df_scaled[actual_columns_to_scale] = scaler.fit_transform(df_scaled[actual_columns_to_scale])
    effective_logger.info("Scaling complete.")
    return df_scaled, scaler


def save_data_and_artifact(
    df: pd.DataFrame,
    data_path: str,
    artifact: Optional[Any],
    artifact_path: Optional[str],
    logger_param: Optional[logging.Logger] = None
) -> None:
    """Saves the processed DataFrame and the preprocessing artifact (e.g., scaler)."""
    effective_logger = logger_param if logger_param else logger
    # Save DataFrame
    data_dir = os.path.dirname(data_path)
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    df.to_csv(data_path, index=False)
    effective_logger.info(f"Processed data saved to: {data_path} | Shape: {df.shape}")

    # Save artifact (scaler)
    if artifact and artifact_path:
        artifact_dir = os.path.dirname(artifact_path)
        if artifact_dir and not os.path.exists(artifact_dir):
            os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(artifact, artifact_path)
        effective_logger.info(f"Preprocessing artifact (scaler) saved to: {artifact_path}")
    elif artifact_path and not artifact:
        effective_logger.warning(f"Artifact path '{artifact_path}' provided, but no artifact (scaler) was generated/provided to save.")
    elif artifact and not artifact_path:
        effective_logger.warning("Artifact (scaler) was generated, but no 'preprocessing_pipeline' path specified in config. Scaler not saved.")


def main_preprocessing(config_path: str = "config.yaml") -> None:
    """Main function to run the data preprocessing pipeline."""
    try:
        config = load_config(config_path) # Uses its own logger or basicConfig
    except FileNotFoundError:
        logging.error(f"CRITICAL: Configuration file not found at {config_path}. Exiting.") 
        return
    except yaml.YAMLError as e:
        logging.error(f"CRITICAL: Error parsing YAML configuration file: {e}. Exiting.")
        return

    global logger # This rebinds the module-level logger
    logger = get_logger(config.get("logging", {})) # Initialize/update the module-level logger
    
    try:
        # Pass the now configured module-level logger to validate_preprocessing_config
        validate_preprocessing_config(config, logger_param=logger)
    except KeyError as e:
        logger.error(f"Configuration validation failed: {e}. Exiting.")
        return

    pre_cfg = config.get("preprocessing", {})
    art_cfg = config.get("artifacts", {})
    data_source_cfg = config.get("data_source", {})

    # --- Data Loading ---
    raw_data_path = data_source_cfg.get("raw_path")
    data_type = data_source_cfg.get("type", "csv").lower()
    delimiter = data_source_cfg.get("delimiter", ",")
    header = data_source_cfg.get("header", 0)
    encoding = data_source_cfg.get("encoding", "utf-8")

    logger.info(f"Attempting to load raw data from: {raw_data_path} (type: {data_type})")
    try:
        if data_type == "csv":
            df_raw = pd.read_csv(raw_data_path, delimiter=delimiter, header=header, encoding=encoding)
        elif data_type == "excel":
            sheet_name = data_source_cfg.get("sheet_name", 0)
            df_raw = pd.read_excel(raw_data_path, sheet_name=sheet_name, header=header)
        else:
            logger.error(f"Unsupported data type '{data_type}' specified in config. Exiting.")
            return
        logger.info(f"Raw data loaded successfully. Shape: {df_raw.shape}")
    except FileNotFoundError:
        logger.error(f"Raw data file not found at {raw_data_path}. Exiting.")
        return
    except Exception as e:
        logger.error(f"Error loading raw data from '{raw_data_path}': {e}. Exiting.")
        return

    logger.info("Starting data validation (placeholder step)...")
    logger.info("Data validation (placeholder step) complete.")
    
    current_df = df_raw.copy()

    # --- Step 1: Drop Columns ---
    # Called without logger_param, so it will use the module-level logger
    columns_to_drop = pre_cfg.get("drop_columns", [])
    current_df = drop_columns(current_df, columns_to_drop)

    # --- Step 2: Outlier Removal ---
    outlier_cfg = pre_cfg.get("outlier_removal", {})
    if outlier_cfg.get("enabled", False):
        outlier_features = outlier_cfg.get("features", [])
        iqr_multiplier = outlier_cfg.get("iqr_multiplier", 1.5)
        if outlier_features:
             current_df = remove_outliers_iqr(current_df, outlier_features, iqr_multiplier)
        else:
            logger.info("Outlier removal enabled but no features specified. Skipping outlier removal.")
    else:
        logger.info("Outlier removal is disabled in the configuration.")

    # --- Step 3: Scale Columns ---
    scale_cfg = pre_cfg.get("scale", {})
    scaler_artifact = None
    if scale_cfg and scale_cfg.get("columns"):
        columns_to_scale = scale_cfg.get("columns", [])
        scaling_method = scale_cfg.get("method", "minmax")
        current_df, scaler_artifact = scale_columns(current_df, columns_to_scale, scaling_method)
    else:
        logger.info("Scaling is disabled or no columns specified for scaling in the configuration.")

    # --- Save Processed Data and Scaler Artifact ---
    processed_data_output_path = data_source_cfg.get("processed_path")
    scaler_artifact_path = art_cfg.get("preprocessing_pipeline")

    if not processed_data_output_path:
        logger.error("Missing 'processed_path' in 'data_source' config. Cannot save processed data.")
    else:
        save_data_and_artifact(current_df, processed_data_output_path, scaler_artifact, scaler_artifact_path)
    
    logger.info("Preprocessing pipeline finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    
    config_file_path = "config.yaml" 
    
    if not os.path.exists(config_file_path):
        logging.error(f"CRITICAL: Main - Configuration file '{config_file_path}' not found. Preprocessing cannot start.")
        logging.error("Please ensure 'config.yaml' is present and correctly configured.")
    else:
        main_preprocessing(config_path=config_file_path)
