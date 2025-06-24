import wandb
import sys
import logging
from datetime import datetime
from pathlib import Path
import hydra
from omegaconf import DictConfig

from src.evaluation.evaluator import evaluate_statsmodels_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

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
        evaluate_statsmodels_model(
            config_path=str(PROJECT_ROOT / "config.yaml")
        )
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
