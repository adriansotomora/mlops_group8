"""
features.py

Config-driven, modular, auditable feature engineering for MLOps.
- All logic and steps are parameterized by config.yaml (lines 33-79)
- Modular, never overwrites raw data, logs every step
- Tracks all feature changes for reproducibility
"""
import os
import logging
import yaml
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import PolynomialFeatures


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


def parse_genres(df, config, logger):
    # Parse artist_genres and create top genre columns
    logger.info("Parsing and engineering genre features...")
    genres_map = {
        'pop': r'(pop)',
        'rock': r'(rock)',
        'electronic': r'(house|edm|electro|progressive)',
        'latin': r'(latin|puerto|reggaeton)',
        'hip-hop': r'(hip|rap|urban)',
        'indie': r'(indie)',
        'jazz': r'(jazz)',
        'r&b': r'(r&b)',
        'soul': r'(soul)',
        'metal': r'(metal|punk)',
        'classic': r'(classic)',
        'country': r'(country)'
    }
    for genre, pattern in genres_map.items():
        colname = f"genre_{genre}"
        df[colname] = df['artist_genres'].str.contains(
            pattern, flags=re.IGNORECASE, regex=True).astype(int)
        logger.info(f"Created genre feature: {colname}")
    return df


def drop_irrelevant_columns(df, config, logger):
    drop_cols = config.get("features", {}).get("drop", [])
    logger.info(f"Dropping irrelevant columns: {drop_cols}")
    return df.drop(columns=drop_cols, errors="ignore")


def create_polynomial_features(df, config, logger):
    poly_cfg = config.get("features", {}).get("polynomial", {})
    audio_features = config["features"].get("audio_features", [])
    genre_features = [
        f"genre_{g}"
        for g in config["features"].get("genre_features", [])
    ]
    dfs = [df]
    # Audio polynomial features
    if "audio" in poly_cfg and audio_features:
        poly = PolynomialFeatures(
            degree=poly_cfg["audio"].get("degree", 2),
            include_bias=poly_cfg["audio"].get("include_bias", False)
        )
        audio_poly = poly.fit_transform(df[audio_features])
        audio_poly_cols = poly.get_feature_names_out(audio_features)
        audio_poly_df = pd.DataFrame(
            audio_poly,
            columns=[f"poly_audio_{c}" for c in audio_poly_cols],
            index=df.index
        )
        logger.info(
            f"Created polynomial audio features: {list(audio_poly_df.columns)}"
        )
        dfs.append(audio_poly_df)
    # Genre polynomial features
    if "genre" in poly_cfg and genre_features:
        poly = PolynomialFeatures(
            degree=poly_cfg["genre"].get("degree", 2),
            include_bias=poly_cfg["genre"].get("include_bias", False)
        )
        genre_poly = poly.fit_transform(df[genre_features])
        genre_poly_cols = poly.get_feature_names_out(genre_features)
        genre_poly_df = pd.DataFrame(
            genre_poly,
            columns=[f"poly_genre_{c}" for c in genre_poly_cols],
            index=df.index
        )
        logger.info(
            f"Created polynomial genre features: {list(genre_poly_df.columns)}"
        )
        dfs.append(genre_poly_df)
    df_out = pd.concat(dfs, axis=1)
    return df_out


def select_features(df, config, logger):
    # Select features for modeling (by config)
    exclude = config["features"].get("exclude", [])
    profiling = config["features"].get("profiling_variables", [])
    selected = df.select_dtypes(include=[np.number]).drop(
        columns=exclude + profiling, errors="ignore"
    )
    logger.info(f"Selected features for modeling: {list(selected.columns)}")
    return selected


def log_feature_list(df, path, logger):
    dir_ = os.path.dirname(path)
    if dir_ and not os.path.exists(dir_):
        os.makedirs(dir_, exist_ok=True)
    features = list(df.columns)
    with open(path, "w") as f:
        for feat in features:
            f.write(f"{feat}\n")
    logger.info(f"Feature list saved to {path}")


def main_features(df, config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger = get_logger(config.get("logging", {}))
    logger.info("Starting feature engineering stage...")
    # 1. Parse genres and create genre features
    df_feat = parse_genres(df.copy(), config, logger)
    # 2. Drop irrelevant columns
    df_feat = drop_irrelevant_columns(df_feat, config, logger)
    # 3. Create polynomial features
    df_feat = create_polynomial_features(df_feat, config, logger)
    # 4. Select features for modeling
    selected = select_features(df_feat, config, logger)
    # 5. Log feature list
    features_path = os.path.join(
        config["artifacts"].get("processed_dir", "data/processed"),
        "feature_list.txt"
    )
    log_feature_list(selected, features_path, logger)
    # 6. Save engineered features
    feat_path = os.path.join(
        config["artifacts"].get("processed_dir", "data/processed"),
        "features.csv"
    )
    selected.to_csv(feat_path, index=False)
    logger.info(f"Engineered features saved to {feat_path}")
    return selected


if __name__ == "__main__":
    from src.data_load.data_loader import get_data
    from src.data_validation.data_validator import validate_data
    from src.preprocess.preprocessing import (
        drop_columns,
        remove_outliers_iqr,
        scale_columns
    )
    config_path = "config.yaml"
    df = get_data(config_path=config_path, data_stage="raw")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    validate_data(df, config)
    # Run preprocessing first
    df = drop_columns(
        df,
        config["preprocessing"].get("drop_columns", []),
        logging.getLogger(__name__)
    )
    out_cfg = config["preprocessing"].get("outlier_removal", {})
    if out_cfg.get("enabled", False):
        df = remove_outliers_iqr(
            df,
            out_cfg.get("features", []),
            out_cfg.get("iqr_multiplier", 1.5),
            logging.getLogger(__name__)
        )
    scale_cfg = config["preprocessing"].get("scale", {})
    columns = scale_cfg.get("columns", [])
    method = scale_cfg.get("method", "minmax")
    df, _ = scale_columns(
        df,
        columns,
        method,
        logging.getLogger(__name__)
    )
    main_features(df, config_path=config_path)
