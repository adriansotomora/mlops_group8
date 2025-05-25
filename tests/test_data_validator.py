import os
import sys
import pytest
import pandas as pd
import json

# Ensure src is in path for import
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "data_validation"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.data_validation.data_validator import validate_data

# Minimal config/schema for Spotify data, based on config.yaml
SPOTIFY_SCHEMA = [
    {"name": "year", "dtype": "int", "required": True, "min": 2000, "max": 2025},
    {"name": "track_name", "dtype": "str", "required": True},
    {"name": "track_popularity", "dtype": "int", "required": True, "min": 0, "max": 100},
    {"name": "album", "dtype": "str", "required": True},
    {"name": "artist_name", "dtype": "str", "required": True},
    {"name": "artist_genres", "dtype": "str", "required": True},
    {"name": "artist_popularity", "dtype": "int", "required": True, "min": 0, "max": 100},
    {"name": "energy", "dtype": "float", "required": False, "min": 0.0519, "max": 0.999},
    {"name": "key", "dtype": "float", "required": True, "min": 0.0, "max": 11.0},
    {"name": "loudness", "dtype": "float", "required": True, "min": -56.0, "max": 132.0},
    {"name": "mode", "dtype": "float", "required": True, "min": 0.0, "max": 1.0},
    {"name": "speechiness", "dtype": "float", "required": True, "min": 0.0225, "max": 0.576},
    {"name": "acousticness", "dtype": "float", "required": True, "min": 0.0000129, "max": 0.978},
    {"name": "instrumentalness", "dtype": "float", "required": True, "min": 0.0, "max": 0.985},
    {"name": "liveness", "dtype": "float", "required": True, "min": 0.021, "max": 0.843},
    {"name": "valence", "dtype": "float", "required": True, "min": 0.0377, "max": 0.974},
    {"name": "tempo", "dtype": "float", "required": True, "min": 60.019, "max": 210.857},
    {"name": "duration_ms", "dtype": "float", "required": True, "min": 97393.0, "max": 688453.0},
    {"name": "danceability", "dtype": "float", "required": True, "min": 0.0, "max": 1.0},
]

SPOTIFY_CONFIG = {
    "data_validation": {
        "enabled": True,
        "action_on_error": "raise",
        "report_path": "logs/test_validation_report.json",
        "schema": {"columns": SPOTIFY_SCHEMA}
    }
}

@pytest.fixture
def valid_spotify_df():
    """A valid DataFrame matching the Spotify schema."""
    return pd.DataFrame({
        "year": [2020, 2021],
        "track_name": ["Song A", "Song B"],
        "track_popularity": [80, 90],
        "album": ["Album A", "Album B"],
        "artist_name": ["Artist A", "Artist B"],
        "artist_genres": ["['pop']", "['rock']"],
        "artist_popularity": [70, 85],
        "energy": [0.5, 0.7],
        "key": [5.0, 7.0],
        "loudness": [-5.0, -7.0],
        "mode": [1.0, 0.0],
        "speechiness": [0.05, 0.07],
        "acousticness": [0.1, 0.2],
        "instrumentalness": [0.0, 0.1],
        "liveness": [0.2, 0.3],
        "valence": [0.5, 0.6],
        "tempo": [120.0, 130.0],
        "duration_ms": [200000.0, 250000.0],
        "danceability": [0.8, 0.9],
    })

@pytest.fixture
def df_missing_required(valid_spotify_df):
    """Missing required column 'track_name'."""
    return valid_spotify_df.drop(columns=["track_name"])

@pytest.fixture
def df_invalid_dtype(valid_spotify_df):
    """'year' as string instead of int."""
    valid_spotify_df["year"] = ["2020", "2021"]
    return valid_spotify_df

@pytest.fixture
def df_out_of_range(valid_spotify_df):
    """'track_popularity' above 100, 'duration_ms' below min."""
    valid_spotify_df.loc[0, "track_popularity"] = 150
    valid_spotify_df.loc[1, "duration_ms"] = 90000.0
    return valid_spotify_df

def test_validate_data_happy_path(valid_spotify_df, tmp_path):
    config = dict(SPOTIFY_CONFIG)
    report_path = tmp_path / "validation_report.json"
    config["data_validation"]["report_path"] = str(report_path)
    validate_data(valid_spotify_df, config)
    assert os.path.exists(report_path)
    with open(report_path) as f:
        report = json.load(f)
        assert report["result"] == "pass"
        assert not report["errors"]

def test_validate_data_missing_required_column(df_missing_required, tmp_path):
    config = dict(SPOTIFY_CONFIG)
    config["data_validation"]["report_path"] = str(tmp_path / "validation_report.json")
    with pytest.raises(ValueError):
        validate_data(df_missing_required, config)
    with open(config["data_validation"]["report_path"]) as f:
        report = json.load(f)
    assert any("Missing required column" in err for err in report["errors"])

def test_validate_data_invalid_dtype(df_invalid_dtype, tmp_path):
    config = dict(SPOTIFY_CONFIG)
    config["data_validation"]["report_path"] = str(tmp_path / "validation_report.json")
    with pytest.raises(ValueError):
        validate_data(df_invalid_dtype, config)
    with open(config["data_validation"]["report_path"]) as f:
        report = json.load(f)
    assert any("has dtype" in err for err in report["errors"])

def test_validate_data_out_of_range(df_out_of_range, tmp_path):
    config = dict(SPOTIFY_CONFIG)
    config["data_validation"]["report_path"] = str(tmp_path / "validation_report.json")
    with pytest.raises(ValueError):
        validate_data(df_out_of_range, config)
    with open(config["data_validation"]["report_path"]) as f:
        report = json.load(f)
    assert any("above max" in err or "below min" in err for err in report["errors"])

def test_validate_data_disabled(valid_spotify_df, tmp_path):
    config = dict(SPOTIFY_CONFIG)
    config["data_validation"]["enabled"] = False
    config["data_validation"]["report_path"] = str(tmp_path / "validation_report.json")
    validate_data(valid_spotify_df, config)
    assert not os.path.exists(config["data_validation"]["report_path"])


def test_validate_data_missing_schema_key(valid_spotify_df):
    config = {"data_validation": {"enabled": True}}
    # Should not raise
    validate_data(valid_spotify_df, config)