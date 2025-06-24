import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import wandb
import logging
from datetime import datetime
import hydra
from omegaconf import DictConfig
from src.model.model import main_modeling

SRC_ROOT = PROJECT_ROOT / "src"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model")


@hydra.main(config_path=str(PROJECT_ROOT),
            config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"model_{dt_str}"
    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="model",
            name=run_name,
            config=dict(cfg),
            tags=["model"]
        )
        logger.info("Started WandB run: %s", run_name)
        main_modeling(config_path=str(PROJECT_ROOT / "config.yaml"))
        try:
            import json
            with open(str(PROJECT_ROOT / "models/metrics.json"), "r") as f:
                metrics = json.load(f)
            wandb.log(metrics)
        except FileNotFoundError:
            logger.warning("metrics.json not found for logging.")
        wandb.log({"model_status": "completed"})
    except Exception as e:
        logger.exception("Failed during model step")
        if run is not None:
            wandb.log({"model_status": "failed", "error": str(e)})
            run.alert(title="Model Error", text=str(e))
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
