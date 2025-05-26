"""Modeling utilities for Spotify popularity analysis.
This module implements regression models using features.csv.
"""

import os
import json
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(cfg: dict) -> None:
    logging.basicConfig(
        filename=cfg.get("log_file", "logs/main.log"),
        level=getattr(logging, cfg.get("level", "INFO")),
        format=cfg.get("format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s"),
        datefmt=cfg.get("datefmt", "%Y-%m-%d %H:%M:%S"),
        filemode="a",
    )
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, cfg.get("level", "INFO")))
    console.setFormatter(logging.Formatter(cfg.get("format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s"), cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")))
    logging.getLogger().addHandler(console)

def stepwise_selection(X: pd.DataFrame, y: pd.Series, threshold_in: float = 0.05, threshold_out: float = 0.1) -> List[str]:
    included: List[str] = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        if not new_pval.empty:
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                logger.info("Add feature %s p-value %.4f", best_feature, best_pval)
        if included:
            model = sm.OLS(y, sm.add_constant(X[included])).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                changed = True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                logger.info("Drop feature %s p-value %.4f", worst_feature, worst_pval)      
        if not changed:
            break
    return included

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict):
    step_cfg = cfg.get("model", {}).get("linear_regression", {}).get("stepwise", {})
    selected = stepwise_selection(
        X_train,
        y_train,
        threshold_in=step_cfg.get("threshold_in", 0.05),
        threshold_out=step_cfg.get("threshold_out", 0.1),
    )
    model = sm.OLS(y_train, sm.add_constant(X_train[selected])).fit()
    logger.info("Linear regression fitted with %d features", len(selected))
    return model, selected

def evaluate_regression(model, X_test: pd.DataFrame, y_test: pd.Series, selected: List[str]) -> dict:
    y_pred = model.predict(sm.add_constant(X_test[selected]))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info("Regression evaluation MSE=%.4f R2=%.4f", mse, r2)
    return {"mse": mse, "r2": r2}

if __name__ == "__main__":
    # Load configuration and set up logging
    cfg = load_config("config.yaml")
    setup_logging(cfg.get("logging", {}))

    # Load features (X) from features.csv
    X = pd.read_csv("data/processed/features.csv")
    # Load target variable (y) from processed.csv
    y = pd.read_csv("data/processed/processed.csv")["track_popularity"]

    # Get train/test split parameters from config
    split_cfg = cfg.get("data_split", {})
    test_size = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train linear regression model using stepwise feature selection
    lr_model, selected = train_linear_regression(X_train, y_train, cfg)

    # Evaluate the model on the test set
    metrics = evaluate_regression(lr_model, X_test, y_test, selected)
    print(metrics)

    # Save the trained model to a file for later reuse
    import pickle
    os.makedirs("models", exist_ok=True)
    with open("models/linear_regression.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    # Save the evaluation metrics to a JSON file for comparison/visualization
    import json
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)