import os
import logging
import yaml
from sklearn.preprocessing import MinMaxScaler
import joblib


def validate_preprocessing_config(config):
    required_keys = ["preprocessing", "artifacts"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    pre = config["preprocessing"]
    for k in ["drop_columns", "outlier_removal", "scale"]:
        if k not in pre:
            logging.warning(f"Missing preprocessing config key: {k}")


def get_logger(logging_config):
    log_file = logging_config.get("log_file", "logs/main.log")
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_format = logging_config.get(
        "format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    date_format = logging_config.get("datefmt", "%Y-%m-%d %H:%M:%S")
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, logging_config.get("level", "INFO")),
        format=log_format,
        datefmt=date_format,
        filemode="a"
    )
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, logging_config.get("level", "INFO")))
    formatter = logging.Formatter(log_format, date_format)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    return logging.getLogger(__name__)


def drop_columns(df, columns, logger):
    logger.info(f"Dropping columns: {columns}")
    return df.drop(columns=columns, errors="ignore")


def remove_outliers_iqr(df, features, iqr_multiplier, logger):
    df_clean = df.copy()
    for feature in features:
        Q1 = df_clean[feature].quantile(0.25)
        Q3 = df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        before = df_clean.shape[0]
        df_clean = df_clean[(df_clean[feature] >= lower) &
                            (df_clean[feature] <= upper)]
        after = df_clean.shape[0]
        logger.info(
            f"Outlier removal on {feature}: "
            f"{before-after} rows dropped"
        )
    return df_clean


def scale_columns(df, columns, method, logger):
    logger.info(f"Scaling columns {columns} with {method}")
    scaler = MinMaxScaler() if method == "minmax" else None
    if scaler is None:
        raise ValueError(f"Unsupported scaling method: {method}")
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
    return df_scaled, scaler


def log_and_save(df, path, logger, msg):
    dir_ = os.path.dirname(path)
    if dir_ and not os.path.exists(dir_):
        os.makedirs(dir_, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"{msg} Saved to: {path} | Shape: {df.shape}")


def main_preprocessing(df, config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    validate_preprocessing_config(config)
    logger = get_logger(config.get("logging", {}))
    pre_cfg = config["preprocessing"]
    art_cfg = config["artifacts"]

    # Drop columns
    df = drop_columns(df, pre_cfg.get("drop_columns", []), logger)

    # Remove outliers
    out_cfg = pre_cfg.get("outlier_removal", {})
    if out_cfg.get("enabled", False):
        df = remove_outliers_iqr(
            df,
            out_cfg.get("features", []),
            out_cfg.get("iqr_multiplier", 1.5),
            logger
        )

    # Scale columns (fit on all data, since no split yet)
    scale_cfg = pre_cfg.get("scale", {})
    columns = scale_cfg.get("columns", [])
    method = scale_cfg.get("method", "minmax")
    df_scaled, scaler = scale_columns(df, columns, method, logger)

    # Save processed data and scaler
    processed_dir = art_cfg.get(
        "processed_dir", "data/processed"
    )
    log_and_save(
        df_scaled,
        os.path.join(processed_dir, "processed.csv"),
        logger,
        "Processed data."
    )
    pipeline_path = art_cfg.get(
        "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
    )
    joblib.dump(scaler, pipeline_path)
    logger.info(f"Saved scaler to {pipeline_path}")


if __name__ == "__main__":
    from src.data_load.data_loader import get_data
    from src.data_validation.data_validator import validate_data
    config_path = "config.yaml"
    df = get_data(config_path=config_path, data_stage="raw")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    validate_data(df, config)
    main_preprocessing(df, config_path=config_path)
