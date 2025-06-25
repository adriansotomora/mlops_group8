import wandb
import sys
import logging
from datetime import datetime
from pathlib import Path
import hydra
from omegaconf import DictConfig
import pickle
import json
import pandas as pd
import os
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import evaluate_statsmodels_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("evaluation")


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"evaluation_{dt_str}"
    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="evaluation",
            name=run_name,
            config=dict(cfg),
            tags=["evaluation"]
        )
        logger.info("Started WandB run: %s", run_name)

        # Load model
        model_path = str(PROJECT_ROOT / "models/linear_regression.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Load selected features
        features_path = str(PROJECT_ROOT / "models/linear_regression_selected_features.json")
        with open(features_path, "r") as f:
            selected_features = json.load(f)["selected_features"]

        # Load evaluation data
        features_csv = str(PROJECT_ROOT / "features.csv")
        logger.info(f"Loading features from {features_csv}")
        X = pd.read_csv(features_csv)

        processed_csv = cfg.data_source.processed_path
        logger.info(f"Loading target from {processed_csv}")
        y = pd.read_csv(processed_csv)[cfg.target]

        # Align X and y if needed
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len].reset_index(drop=True)
        y = y.iloc[:min_len].reset_index(drop=True)

        # Evaluate
        metrics = evaluate_statsmodels_model(
            model=model,
            X_eval=X[selected_features],
            y_eval=y,
            selected_features=selected_features,
            split_label="test"
        )
        logger.info(f"Evaluation metrics: {metrics}")
        wandb.log(metrics)
        wandb.log({"evaluation_status": "completed"})
    except Exception as e:
        logger.exception("Failed during evaluation step")
        if run is not None:
            wandb.log({"evaluation_status": "failed", "error": str(e)})
            run.alert(title="Evaluation Error", text=str(e))
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
