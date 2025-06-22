import wandb
import os
import json
from datetime import datetime
from .model import main_modeling

def main():
    run_name = f"model_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mlops_group8_spotify_prediction"),
        entity=os.environ.get("WANDB_ENTITY", "hiroinie"),
        job_type="model",
        name=run_name,
    )
    
    try:
        main_modeling(config_path="../../config.yaml")
        
        try:
            with open("../../models/metrics.json", "r") as f:
                metrics = json.load(f)
            wandb.log(metrics)
        except FileNotFoundError:
            pass
            
        wandb.log({"model_status": "completed"})
    except Exception as e:
        wandb.log({"model_status": "failed", "error": str(e)})
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
