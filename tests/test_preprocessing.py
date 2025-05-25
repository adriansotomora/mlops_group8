import os
import pandas as pd
import numpy as np
import yaml
import joblib
from unittest.mock import MagicMock
from src.preprocess import preprocessing


def minimal_config(tmpdir):
    return {
        "preprocessing": {
            "drop_columns": ["dropme"],
            "outlier_removal": {
                "enabled": True,
                "features": ["A"],
                "iqr_multiplier": 1.5
            },
            "scale": {
                "columns": ["A", "B"],
                "method": "minmax"
            }
        },
        "artifacts": {
            "processed_dir": str(tmpdir),
            "preprocessing_pipeline": os.path.join(str(tmpdir), "scaler.pkl")
        },
        "logging": {
            "log_file": os.path.join(str(tmpdir), "test.log"),
            "level": "INFO"
        }
    }


def test_validate_preprocessing_config_warns_missing_keys(caplog):
    config = {"preprocessing": {}, "artifacts": {}}
    with caplog.at_level("WARNING"):
        preprocessing.validate_preprocessing_config(config)
        assert any(
            "Missing preprocessing config key" in m for m in caplog.messages
        )


def test_drop_columns_removes_specified():
    df = pd.DataFrame({"A": [1, 2], "dropme": [3, 4]})
    logger = MagicMock()
    df2 = preprocessing.drop_columns(df, ["dropme"], logger)
    assert "dropme" not in df2.columns
    assert "A" in df2.columns


def test_remove_outliers_iqr_removes_rows():
    df = pd.DataFrame({"A": [1, 2, 100, 3, 4]})
    logger = MagicMock()
    df2 = preprocessing.remove_outliers_iqr(df, ["A"], 1.5, logger)
    assert df2["A"].max() < 100
    assert df2.shape[0] < df.shape[0]


def test_scale_columns_minmax():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
    logger = MagicMock()
    df2, scaler = preprocessing.scale_columns(df, ["A", "B"], "minmax", logger)
    np.testing.assert_almost_equal(df2["A"].min(), 0.0)
    np.testing.assert_almost_equal(df2["A"].max(), 1.0)
    np.testing.assert_almost_equal(df2["B"].min(), 0.0)
    np.testing.assert_almost_equal(df2["B"].max(), 1.0)


def test_main_preprocessing_e2e(tmp_path):
    # Create toy data
    df = pd.DataFrame({
        "A": [1, 2, 100, 3, 4],
        "B": [10, 20, 30, 40, 50],
        "dropme": [0, 0, 0, 0, 0]
    })
    config = minimal_config(tmp_path)
    config_path = os.path.join(tmp_path, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    # Run preprocessing
    preprocessing.main_preprocessing(df, config_path=config_path)
    # Check processed file
    processed_path = os.path.join(tmp_path, "processed.csv")
    assert os.path.exists(processed_path)
    df_proc = pd.read_csv(processed_path)
    # Should not have 'dropme'
    assert "dropme" not in df_proc.columns
    # Should be scaled
    assert np.isclose(df_proc["A"].min(), 0.0)
    assert np.isclose(df_proc["A"].max(), 1.0)
    # Check scaler file
    scaler_path = os.path.join(tmp_path, "scaler.pkl")
    assert os.path.exists(scaler_path)
    scaler = joblib.load(scaler_path)
    arr = scaler.transform([[2, 20]])
    assert arr.shape == (1, 2)
