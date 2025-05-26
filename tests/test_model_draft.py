import pandas as pd
import numpy as np
from src.model import model_draft


def test_stepwise_selection_basic():
    # Test that stepwise_selection returns a non-empty list of selected features
    X = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],
        "C": [5, 3, 6, 9, 12],
    })
    y = pd.Series([1, 2, 3, 4, 5])
    selected = model_draft.stepwise_selection(X, y, threshold_in=0.5, threshold_out=0.5)
    assert isinstance(selected, list)
    assert len(selected) > 0


def test_train_linear_regression_and_evaluate():
    # Test that train_linear_regression returns a model and selected features, and evaluation returns metrics
    X = pd.DataFrame({
        "A": np.random.rand(100),
        "B": np.random.rand(100),
        "C": np.random.rand(100),
    })
    y = 2 * X["A"] + 3 * X["B"] + np.random.normal(0, 0.1, 100)
    cfg = {"model": {"linear_regression": {"stepwise": {"threshold_in": 0.05, "threshold_out": 0.1}}}}
    model, selected = model_draft.train_linear_regression(X, y, cfg)
    assert hasattr(model, "predict")
    assert isinstance(selected, list)
    # Evaluate on the same data (just for test)
    metrics = model_draft.evaluate_regression(model, X, y, selected)
    assert "mse" in metrics
    assert "r2" in metrics
    assert isinstance(metrics["mse"], float)
    assert isinstance(metrics["r2"], float)