"""
data_loader.py

Utility for loading the initial raw CSV dataset for the MLOps pipeline,
driven by a YAML configuration file.
"""

import os
import logging
import pandas as pd
import yaml
from typing import Dict, Any

# Module-level logger. Assumes configuration by the calling script or __main__.
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    if not os.path.isfile(config_path):
        logger.error(f"Configuration file not found: {config_path}") 
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.debug(f"Configuration loaded from {config_path}")
    return config

def _read_csv_data(
    path: str,
    delimiter: str = ",",
    header: int = 0,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """Helper function to read data from a CSV file with error handling."""
    if not path or not isinstance(path, str):
        logger.error("No valid data path provided for CSV loading.")
        raise ValueError("No valid data path specified for CSV loading.")
    if not os.path.isfile(path):
        logger.error(f"Data file does not exist: {path}")
        raise FileNotFoundError(f"Data file not found: {path}")
    
    try:
        df = pd.read_csv(path, delimiter=delimiter, header=header, encoding=encoding)
        logger.info(f"Loaded data from CSV: {path}, shape={df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from CSV '{path}': {e}", exc_info=True)
        raise # Re-raise after logging

def get_raw_data(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Loads the raw dataset based on specifications in the configuration file.
    Primarily intended for loading the initial raw CSV data for the pipeline.
    """
    try:
        config = load_config(config_path)
    except FileNotFoundError: # Handled by load_config
        raise 
    except yaml.YAMLError as e: # Specific error for YAML parsing
        logger.error(f"Error parsing YAML configuration from '{config_path}': {e}", exc_info=True)
        raise ValueError(f"Error parsing YAML configuration from '{config_path}'.") from e

    data_cfg = config.get("data_source", {})
    
    file_type = data_cfg.get("type", "csv").lower()
    if file_type != "csv":
        logger.error(f"Unsupported file type in config: '{file_type}'. Only 'csv' supported.")
        raise ValueError(f"Unsupported file type: '{file_type}'. This loader only supports 'csv'.")

    raw_data_path = data_cfg.get("raw_path")
    if not raw_data_path or not isinstance(raw_data_path, str):
        logger.error("Config missing 'data_source.raw_path'.")
        raise ValueError("Config must specify 'data_source.raw_path' for raw data.")

    logger.info(f"Attempting to load raw data from: {raw_data_path}")
    
    df = _read_csv_data(
        path=raw_data_path,
        delimiter=data_cfg.get("delimiter", ","),
        header=data_cfg.get("header", 0),
        encoding=data_cfg.get("encoding", "utf-8"),
    )
    return df

if __name__ == "__main__":
    # Basic logging setup for standalone execution.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.info("Running data_loader.py standalone...")
    
    try:
        # Assumes config.yaml is in the current working directory when run standalone.
        raw_df = get_raw_data() 
        logger.info(f"Raw data loaded successfully via __main__. Shape: {raw_df.shape}")
        logger.info("First 5 rows of loaded data:\n%s", raw_df.head().to_string())
    except Exception as e: 
        logger.error(f"__main__: Failed to load data: {e}", exc_info=True)
