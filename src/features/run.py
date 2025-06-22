import wandb
import os
from datetime import datetime
from .features import main_features

def main():
    run_name = f"features_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mlops_group8_spotify_prediction"),
        entity=os.environ.get("WANDB_ENTITY", "hiroinie"),
        job_type="features",
        name=run_name,
    )
    
    try:
        main_features(config_path="../../config.yaml")
        wandb.log({"features_status": "completed"})
    except Exception as e:
        wandb.log({"features_status": "failed", "error": str(e)})
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
