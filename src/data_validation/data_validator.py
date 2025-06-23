
"""
Data validation utilities for the MLOps pipeline.

This module provides functions to validate data against defined schemas
and generate validation reports.
"""
import logging
import os
import json
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


def _is_dtype_compatible(series, expected_dtype: str) -> bool:
    kind = series.dtype.kind
    if expected_dtype == "int":
        return kind in ("i", "u")
    elif expected_dtype == "float":
        return kind == "f"
    elif expected_dtype == "str":
        return kind in ("O", "U", "S")
    elif expected_dtype == "bool":
        return kind == "b"
    return False


def _validate_column(
    dataframe: pd.DataFrame,
    col_schema: Dict[str, Any],
    errors: List[str],
    warnings: List[str],
    report: Dict[str, Any]
) -> None:
    """Validate a single column in the dataframe according to the schema."""
    col = col_schema["name"]
    col_report = {}
    if col not in dataframe.columns:
        if col_schema.get("required", True):
            msg = f"Missing required column: {col}"
            errors.append(msg)
            print(f"ERROR ADDED: {msg}")
            col_report["status"] = "missing"
            col_report["error"] = msg
        else:
            col_report["status"] = "not present (optional)"
        report[col] = col_report
        return

    dataframe = dataframe.dropna(subset=[col])  # Drop rows with NaN in this column
    col_series = dataframe[col]
    col_report["status"] = "present"

    # Type check
    dtype_expected = col_schema.get("dtype")
    if dtype_expected and not _is_dtype_compatible(col_series, dtype_expected):
        msg = (
                f"Column '{col}' has dtype '{col_series.dtype}', expected '{dtype_expected}'"
            )
        errors.append(msg)
        col_report["dtype"] = str(col_series.dtype)
        col_report["dtype_expected"] = dtype_expected
        col_report["error"] = msg
        report[col] = col_report
        return  # Stop further checks if dtype is wrong

    # Missing values check
    missing_count = col_series.isnull().sum()
    if missing_count > 0:
        if col_schema.get("required", True):
            msg = (
                f"Column '{col}' had {missing_count} missing values (required); rows dropped."
            )
            errors.append(msg)
        else:
            msg = (
                f"Column '{col}' had {missing_count} missing values (optional); rows dropped."
            )
            warnings.append(msg)
        col_report["missing_count"] = int(missing_count)
        col_report["rows_dropped"] = int(missing_count)
    # Value checks: min, max, allowed_values
    if "min" in col_schema:
        min_val = col_schema["min"]
        below = (col_series < min_val).sum()
        if below > 0:
            msg = (
                f"Column '{col}' has {below} values below min ({min_val})"
            )
            errors.append(msg)
            col_report["below_min"] = int(below)
    if "max" in col_schema:
        max_val = col_schema["max"]
        above = (col_series > max_val).sum()
        if above > 0:
            msg = (
                f"Column '{col}' has {above} values above max ({max_val})"
            )
            errors.append(msg)
            col_report["above_max"] = int(above)

def validate_data(dataframe, config_dict):
    """Validate dataframe against schema defined in config."""
    dv_cfg = config_dict.get("data_validation", {})
    report_path = str(dv_cfg.get("report_path", "logs/validation_report.json"))
    enabled = dv_cfg.get("enabled", True)
    errors, warnings = [], []
    report = {}

    if not enabled:
        logger.info("Data validation is disabled in config.")
        return

    schema = dv_cfg.get("schema", {}).get("columns", [])
    dir_ = os.path.dirname(report_path)

    if not schema:
        logger.warning(
            "No data_validation.schema.columns defined in config. "
            "Skipping validation."
        )
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump({
                "result": "pass",
                "errors": [],
                "warnings": [],
                "details": {}
            }, report_file, indent=2)
        return

    # Validate each column as per schema
    for col_schema in schema:
        _validate_column(dataframe, col_schema, errors, warnings, report)

    action_on_error = dv_cfg.get("action_on_error", "raise").lower()

    # Log summary
    if errors:
        logger.error(
            "Data validation failed with %d errors. See %s",
            len(errors), report_path
        )
        for err in errors:
            logger.error("%s", err)
    if warnings:
        logger.warning(
            "Data validation warnings: %d", len(warnings)
        )
        for warn in warnings:
            logger.warning("%s", warn)

    # --- WRITE THE REPORT HERE, ALWAYS ---
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump({
            "result": "fail" if errors else "pass",
            "errors": errors,
            "warnings": warnings,
            "details": report
        }, report_file, indent=2)

    # Teaching: You want strict validation in prod, warnings for research
    if errors:
        if action_on_error == "raise":
            raise ValueError(
                f"Data validation failed with errors. See {report_path} for details"
            )
        elif action_on_error == "warn":
            logger.warning(
                "Data validation errors detected but proceeding as per config."
            )

if __name__ == "__main__":
    import sys
    import yaml
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python -m src.data_validation.data_validation "
            "<data.csv> <config.yaml>"
        )
        sys.exit(1)
    data_path, config_path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(data_path)
    # Drop rows with missing values before validation
    df = df.dropna()
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = yaml.safe_load(config_file)
    validate_data(df, config_dict)
