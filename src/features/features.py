"""
Config-driven, modular feature engineering.

Config-driven, modular feature engineering.
- Loads preprocessed data based on config.yaml.
- Applies feature transformations (genre parsing, drops, polynomial features).
- Selects final features for modeling.
- Saves engineered features DataFrame and a list of feature names.
"""
import os
import logging
import sys 
import yaml
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, Any, List, Optional

# Module-level logger - will be initialized by main_features
logger = logging.getLogger(__name__)

def get_logger(logging_config: Dict[str, Any], default_log_file: str = "logs/features.log") -> logging.Logger:
    """Set up and return a logger based on the provided configuration."""
    log_file = logging_config.get("log_file", default_log_file)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_format_str = logging_config.get("format", "%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s")
    date_format_str = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_level_name = logging_config.get("level", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    for handler in logging.root.handlers[:]: # Clear existing root handlers
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
    module_logger = logging.getLogger(__name__) # Get logger for this module
    module_logger.setLevel(log_level)
    return module_logger

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load the YAML configuration file."""
    _temp_logger = logging.getLogger(f"{__name__}.load_config") # Use a temporary logger
    if not os.path.isfile(config_path):
        _temp_logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    _temp_logger.info(f"Configuration loaded from {config_path}")
    return config

def parse_genres(df: pd.DataFrame, config: Dict[str, Any], logger_param: logging.Logger) -> pd.DataFrame:
    """Parse artist_genres and create binary genre columns."""
    logger_param.info("Parsing and engineering genre features...")
    # Define genre patterns; using non-capturing groups (?:...)
    genres_map = {
        'pop': r'(?:pop)', 'rock': r'(?:rock)',
        'electronic': r'(?:house|edm|electro|techno|progressive)',
        'latin': r'(?:latin|reggaeton|salsa|bachata)',
        'hip-hop': r'(?:hip hop|rap|urban|trap)', 'indie': r'(?:indie)',
        'jazz': r'(?:jazz)', 'r&b': r'(?:r&b|soul)',
        'metal': r'(?:metal|punk|hardcore)',
        'classic': r'(?:classical|orchestra|symphony)', 'country': r'(?:country)'
    }
    # Alternative: Load genre list from config for more dynamic mapping
    # genre_list_from_config = config.get("features", {}).get("genre_features", [])
    # genres_map = {g.replace('-', '_'): f"(?:{g})" for g in genre_list_from_config}


    if 'artist_genres' not in df.columns:
        logger_param.error("'artist_genres' column not found. Skipping genre parsing.")
        return df

    df_out = df.copy()
    for genre, pattern in genres_map.items():
        colname = f"genre_{genre.replace('&', 'and').replace('-', '_')}" # Sanitize column name
        try:
            df_out[colname] = df_out['artist_genres'].astype(str).str.contains(
                pattern, flags=re.IGNORECASE, regex=True, na=False).astype(int)
            logger_param.debug(f"Created genre feature: {colname}")
        except Exception as e:
            logger_param.error(f"Error creating genre feature {colname} (pattern '{pattern}'): {e}")
            df_out[colname] = 0 # Default to 0 on error
    logger_param.info("Genre feature creation complete.")
    return df_out

def drop_irrelevant_columns(df: pd.DataFrame, config: Dict[str, Any], logger_param: logging.Logger) -> pd.DataFrame:
    """Drop columns specified in config['features']['drop']."""
    drop_cols_list = config.get("features", {}).get("drop", [])
    if not drop_cols_list:
        logger_param.info("No columns specified for dropping in 'features.drop' config.")
        return df

    existing_cols_to_drop = [col for col in drop_cols_list if col in df.columns]
    if not existing_cols_to_drop:
        logger_param.warning(f"None of columns to drop ({drop_cols_list}) exist in DataFrame.")
        return df
        
    logger_param.info(f"Dropping columns (from features.drop): {existing_cols_to_drop}")
    df_dropped = df.drop(columns=existing_cols_to_drop, errors="ignore")
    logger_param.info(f"Shape after dropping (features.drop): {df_dropped.shape}")
    return df_dropped

def create_polynomial_features(df: pd.DataFrame, config: Dict[str, Any], logger_param: logging.Logger) -> pd.DataFrame:
    """Create polynomial features based on 'audio' and 'genre' configurations."""
    poly_cfg_main = config.get("features", {}).get("polynomial", {})
    if not poly_cfg_main:
        logger_param.info("No polynomial feature configuration. Skipping.")
        return df

    df_out = df.copy()
    features_section = config.get("features", {})

    for poly_type in ["audio", "genre"]:
        poly_type_cfg = poly_cfg_main.get(poly_type, {})
        if not poly_type_cfg:
            logger_param.debug(f"No polynomial config for type '{poly_type}'. Skipping.")
            continue

        base_feature_names_config = []
        prefix = ""
        if poly_type == "audio":
            base_feature_names_config = features_section.get("audio_features", [])
            prefix = "poly_audio_"
        elif poly_type == "genre":
            genre_base_names = features_section.get("genre_features", [])
            base_feature_names_config = [f"genre_{g.replace('&', 'and').replace('-', '_')}" for g in genre_base_names]
            prefix = "poly_genre_"
        
        actual_base_features = [
            name for name in base_feature_names_config 
            if name in df_out.columns and pd.api.types.is_numeric_dtype(df_out[name])
        ]
        for name in base_feature_names_config: # Log missing/non-numeric
            if name not in df_out.columns: logger_param.warning(f"Poly base '{name}' ({poly_type}) not found.")
            elif name not in actual_base_features: logger_param.warning(f"Poly base '{name}' ({poly_type}) not numeric.")

        if not actual_base_features:
            logger_param.info(f"No valid base features for {poly_type} polynomial creation.")
            continue
        
        logger_param.info(f"Creating {poly_type} polynomial features using: {actual_base_features}")
        poly_transformer = PolynomialFeatures(
            degree=poly_type_cfg.get("degree", 2),
            include_bias=poly_type_cfg.get("include_bias", False),
            interaction_only=poly_type_cfg.get("interaction_only", False)
        )
        
        try:
            poly_data = poly_transformer.fit_transform(df_out[actual_base_features])
            poly_names_raw = poly_transformer.get_feature_names_out(actual_base_features)
            
            poly_df = pd.DataFrame(
                poly_data,
                columns=[f"{prefix}{c.replace(' ', '_').replace('^', 'pow')}" for c in poly_names_raw],
                index=df_out.index
            )

            if poly_type_cfg.get("degree", 2) > 1 and not poly_type_cfg.get("include_bias", False):
                # Remove 1st degree terms to avoid duplicating original features
                cols_to_drop = [f"{prefix}{feat.replace(' ', '_').replace('^', 'pow')}" for feat in actual_base_features]
                poly_df = poly_df.drop(columns=cols_to_drop, errors='ignore')
            
            logger_param.info(f"Created {poly_df.shape[1]} new {prefix}features.")
            df_out = pd.concat([df_out, poly_df], axis=1)
            df_out = df_out.loc[:,~df_out.columns.duplicated()] # Remove fully duplicated columns
        except Exception as e:
            logger_param.error(f"Error creating {poly_type} polynomial features: {e}", exc_info=True)
            
    return df_out

def select_features(df: pd.DataFrame, config: Dict[str, Any], logger_param: logging.Logger) -> pd.DataFrame:
    """Select final numeric features, excluding specified columns."""
    features_cfg = config.get("features", {})
    exclude_cols = features_cfg.get("exclude", [])
    profiling_cols = features_cfg.get("profiling_variables", [])
    
    cols_to_exclude = list(set(exclude_cols + profiling_cols))
    logger_param.info(f"Columns to exclude from final features: {cols_to_exclude}")
    
    numeric_df = df.select_dtypes(include=[np.number])
    logger_param.info(f"Initial numeric features count: {len(numeric_df.columns)}")
    
    final_feature_cols = [col for col in numeric_df.columns if col not in cols_to_exclude]
    selected_df = numeric_df[final_feature_cols]
    
    logger_param.info(f"Selected {len(selected_df.columns)} final features: {list(selected_df.columns)}")
    return selected_df

def log_feature_list(df_features: pd.DataFrame, path: str, logger_param: logging.Logger):
    """Save the list of final features to a text file."""
    dir_ = os.path.dirname(path)
    if dir_ and not os.path.exists(dir_):
        os.makedirs(dir_, exist_ok=True)
    
    features_list = list(df_features.columns)
    try:
        with open(path, "w") as f:
            for feat_name in features_list: f.write(f"{feat_name}\n")
        logger_param.info(f"Final feature list ({len(features_list)}) saved to {path}")
    except Exception as e:
        logger_param.error(f"Error saving feature list to {path}: {e}")

def main_features(config_path: str = "config.yaml"):
    """Execute feature engineering stage. Load preprocessed data and apply transformations."""
    try:
        config = load_config(config_path)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s") # Fallback logger
        logging.critical(f"Features Stage: Failed to load config '{config_path}': {e}", exc_info=True)
        return

    global logger # Rebind module-level logger with specific config
    logger = get_logger(config.get("logging", {}), default_log_file="logs/features.log")
    logger.info(f"--- Starting Feature Engineering Stage (config: {config_path}) ---")

    # Load input data (output of preprocessing.py)
    data_source_cfg = config.get("data_source", {})
    processed_input_path = data_source_cfg.get("processed_path")
    if not processed_input_path:
        logger.critical("Config missing 'data_source.processed_path'. Cannot load input data.")
        return
    
    logger.info(f"Loading preprocessed data from: {processed_input_path}")
    try:
        df_input = pd.read_csv(processed_input_path)
        logger.info(f"Preprocessed data loaded. Shape: {df_input.shape}")
    except FileNotFoundError:
        logger.critical(f"Input file not found: '{processed_input_path}'. Ensure preprocessing ran.")
        return
    except Exception as e:
        logger.critical(f"Error loading data from '{processed_input_path}': {e}", exc_info=True)
        return
    if df_input.empty:
        logger.critical("Loaded preprocessed data is empty. Aborting feature engineering.")
        return

    # --- Apply Feature Engineering Steps ---
    df_feat = df_input.copy()
    df_feat = parse_genres(df_feat, config, logger)
    df_feat = drop_irrelevant_columns(df_feat, config, logger) # Uses features.drop from config
    df_feat = create_polynomial_features(df_feat, config, logger)
    selected_features_df = select_features(df_feat, config, logger) # Selects final numeric features

    # --- Save Outputs ---
    artifacts_cfg = config.get("artifacts", {})
    processed_dir = artifacts_cfg.get("processed_dir", "data/processed") # Default output dir

    # Save list of final feature names
    feature_list_filename = artifacts_cfg.get("feature_list_filename", "feature_list.txt")
    feature_list_output_path = os.path.join(processed_dir, feature_list_filename)
    log_feature_list(selected_features_df, feature_list_output_path, logger)
    
    # Save the DataFrame with all engineered features (before final selection for modeling)
    # This is often useful for EDA or if different models use different subsets.
    # However, the current select_features already filters. If you want all engineered features,
    # save df_feat before select_features, or adjust select_features.
    # For now, saving the 'selected_features_df' which is ready for modeling.
    engineered_features_filename = artifacts_cfg.get("engineered_features_filename", "features.csv")
    engineered_features_output_path = os.path.join(processed_dir, engineered_features_filename)
    try:
        selected_features_df.to_csv(engineered_features_output_path, index=False)
        logger.info(f"Engineered features DataFrame saved to {engineered_features_output_path} | Shape: {selected_features_df.shape}")
    except Exception as e:
        logger.error(f"Error saving engineered features to {engineered_features_output_path}: {e}", exc_info=True)
        
    logger.info("--- Feature Engineering Stage Completed ---")
    # This function doesn't need to return the DataFrame if main.py orchestrates file I/O
    # return selected_features_df 


if __name__ == "__main__":
    # This allows running features.py directly, assuming preprocessing.py has already run
    # and its output (e.g., data/processed/Songs_2025_processed.csv) exists.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger.info("Running features.py as a standalone script...")
    
    # Default config path, can be overridden by an orchestrator if needed
    CONFIG_PATH = "config.yaml" 
    if not os.path.exists(CONFIG_PATH):
        logger.critical(f"Configuration file '{CONFIG_PATH}' not found. Cannot run features.py standalone.")
    else:
        try:
            main_features(config_path=CONFIG_PATH)
        except Exception as e:
            logger.critical(f"Error during standalone execution of features.py: {e}", exc_info=True)

