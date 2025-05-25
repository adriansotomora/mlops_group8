import os
import pandas as pd
import numpy as np
import yaml
from unittest.mock import MagicMock
from src.features import features

def minimal_config(tmpdir):
    return {
        "features": {
            "drop": ["dropme"],
            "audio_features": ["danceability", "energy"],
            "genre_features": ["pop", "rock"],
            "polynomial": {
                "audio": {"degree": 2, "include_bias": False},
                "genre": {"degree": 2, "include_bias": False}
            },
            "exclude": ["year"],
            "profiling_variables": ["track_popularity"]
        },
        "artifacts": {
            "processed_dir": str(tmpdir)
        },
        "logging": {
            "log_file": os.path.join(str(tmpdir), "test.log"),
            "level": "INFO"
        }
    }

def toy_df():
    return pd.DataFrame({
        "artist_genres": ["[pop; rock]", "[rock]", "[pop]", "[jazz]"],
        "danceability": [0.1, 0.2, 0.3, 0.4],
        "energy": [0.5, 0.6, 0.7, 0.8],
        "year": [2020, 2021, 2022, 2023],
        "track_popularity": [10, 20, 30, 40],
        "dropme": [1, 2, 3, 4]
    })

def test_parse_genres_creates_columns(tmp_path):
    df = toy_df()
    config = minimal_config(tmp_path)
    logger = MagicMock()
    df2 = features.parse_genres(df.copy(), config, logger)
    assert "genre_pop" in df2.columns
    assert "genre_rock" in df2.columns
    assert set(df2["genre_pop"]).issubset({0, 1})

def test_drop_irrelevant_columns_removes(tmp_path):
    df = toy_df()
    config = minimal_config(tmp_path)
    logger = MagicMock()
    df2 = features.drop_irrelevant_columns(df, config, logger)
    assert "dropme" not in df2.columns
    assert "artist_genres" in df2.columns

def test_create_polynomial_features_adds_columns(tmp_path):
    df = toy_df()
    config = minimal_config(tmp_path)
    logger = MagicMock()
    df2 = features.parse_genres(df.copy(), config, logger)
    df3 = features.create_polynomial_features(df2, config, logger)
    assert any("poly_audio_" in c for c in df3.columns)
    assert any("poly_genre_" in c for c in df3.columns)

def test_select_features_excludes_and_profiles(tmp_path):
    df = toy_df()
    config = minimal_config(tmp_path)
    logger = MagicMock()
    df2 = features.parse_genres(df.copy(), config, logger)
    df3 = features.create_polynomial_features(df2, config, logger)
    selected = features.select_features(df3, config, logger)
    assert "year" not in selected.columns
    assert "track_popularity" not in selected.columns

def test_log_feature_list_creates_file(tmp_path):
    df = toy_df()
    config = minimal_config(tmp_path)
    logger = MagicMock()
    path = os.path.join(tmp_path, "feature_list.txt")
    features.log_feature_list(df, path, logger)
    assert os.path.exists(path)
    with open(path) as f:
        lines = f.read().splitlines()
    assert "artist_genres" in lines or "danceability" in lines

def test_main_features_e2e(tmp_path):
    df = toy_df()
    config = minimal_config(tmp_path)
    config_path = os.path.join(tmp_path, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    out = features.main_features(df, config_path=config_path)
    # Check output file
    features_path = os.path.join(tmp_path, "features.csv")
    assert os.path.exists(features_path)
    df_out = pd.read_csv(features_path)
    # Should not have 'dropme', should have polynomial features
    assert "dropme" not in df_out.columns
    assert any("poly_audio_" in c for c in df_out.columns)
    assert any("poly_genre_" in c for c in df_out.columns)
