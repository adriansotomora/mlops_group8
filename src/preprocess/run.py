import wandb
import os
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocess.preprocessing import main_preprocessing

def main():
    run_name = f"preprocess_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mlops_group8_spotify_prediction"),
        entity=os.environ.get("WANDB_ENTITY", "hiroinie"),
        job_type="preprocess",
        name=run_name,
    )
    
    try:
        main_preprocessing(config_path="../../config.yaml")
        wandb.log({"preprocess_status": "completed"})
    except Exception as e:
        wandb.log({"preprocess_status": "failed", "error": str(e)})
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
