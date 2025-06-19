"""
Main entry point and orchestrator for the MLOps pipeline.

Project entry point and orchestrator for the MLOps pipeline.
Supports running preprocessing, feature engineering, model training, 
and batch inference stages.
"""

import argparse
import sys
import logging
import os
import yaml
from typing import Dict, Any 

# Import main functions from each pipeline stage module
try:
    from src.preprocess.preprocessing import main_preprocessing
    from src.features.features import main_features
    from src.model.model import main_modeling 
    from src.inference.inferencer import run_inference 
except ModuleNotFoundError as e:
    print(f"CRITICAL ERROR: Failed to import a pipeline stage module: {e}. "
          f"Ensure scripts are in 'src' subdirectories and PYTHONPATH is set, or run from project root.")
    sys.exit(1)

# Module-level logger; configured by setup_logging
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load the YAML configuration file."""
    if not os.path.isfile(config_path):
        # Use basic logging for this critical error as main logger might not be set up
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
        logging.critical(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(logging_config: Dict[str, Any], default_log_file: str = "logs/main_orchestrator.log"):
    """Set up logging for the application based on configuration."""
    log_file = logging_config.get("log_file", default_log_file)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_format_str = logging_config.get("format", "%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(message)s")
    date_format_str = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_level_name = logging_config.get("level", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Clear existing root handlers to prevent duplicate logs
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
    logging.getLogger(__name__).setLevel(log_level) # Set level for this module's logger
    logger.info(f"Logging configured. Level: {log_level_name}, File: {log_file}")


def main():
    """Parse arguments and run selected pipeline stages."""
    parser = argparse.ArgumentParser(
        description="Main MLOps Pipeline Orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the main YAML configuration file."
    )
    parser.add_argument(
        "--stage", type=str, default="all_training",
        choices=["preprocess", "features", "model", "all_training", "inference"],
        help="Pipeline stage to execute."
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Path to input CSV for inference (used with --stage inference)."
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Path to save output CSV with predictions (used with --stage inference)."
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e: # Catches FileNotFoundError and YAMLError from load_config
        # Error is already logged by load_config or basicConfig is used for critical errors
        sys.exit(1)

    try:
        setup_logging(config.get("logging", {}), default_log_file="logs/main_orchestrator.log")
    except Exception as e:
        print(f"CRITICAL: Failed to set up logging: {e}", file=sys.stderr)
        sys.exit(1)
    
    global logger # Rebind module-level logger to the one configured by setup_logging
    logger = logging.getLogger(__name__)

    logger.info(f"Pipeline execution started. Stage: '{args.stage}', Config: '{args.config}'")

    try:
        if args.stage == "preprocess" or args.stage == "all_training":
            logger.info("--- Running: Preprocessing Stage ---")
            main_preprocessing(config_path=args.config)
            logger.info("--- Completed: Preprocessing Stage ---")

        if args.stage == "features" or args.stage == "all_training":
            logger.info("--- Running: Feature Engineering Stage ---")
            main_features(config_path=args.config) # Assumes main_features loads its own data
            logger.info("--- Completed: Feature Engineering Stage ---")

        if args.stage == "model" or args.stage == "all_training":
            logger.info("--- Running: Model Training Stage ---")
            main_modeling(config_path=args.config) # Assumes main_modeling loads its own data
            logger.info("--- Completed: Model Training Stage ---")
        
        if args.stage == "inference":
            logger.info("--- Running: Inference Stage ---")
            if not args.input_file or not args.output_file:
                logger.critical("For --stage inference, --input_file and --output_file are required.")
                sys.exit(1)
            run_inference(
                input_csv_path=args.input_file,
                config_path=args.config, 
                output_csv_path=args.output_file
            )
            logger.info("--- Completed: Inference Stage ---")

    except FileNotFoundError as e: 
        logger.critical(f"Pipeline failed (stage '{args.stage}'): File not found. {e}", exc_info=True)
        logger.critical("Ensure previous stages ran successfully and created expected outputs.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Pipeline failed (stage '{args.stage}'): {e}", exc_info=True)
        sys.exit(1)

    logger.info("Pipeline execution completed successfully for stage(s): '%s'.", args.stage)


if __name__ == "__main__":
    # Basic logging for initial errors if main() or setup_logging() fails very early
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    main()
