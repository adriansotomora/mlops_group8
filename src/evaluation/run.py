import wandb
import os
from datetime import datetime

def main():
    run_name = f"evaluation_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mlops_group8_spotify_prediction"),
        entity=os.environ.get("WANDB_ENTITY", "hiroinie"),
        job_type="evaluation",
        name=run_name,
    )
    
    try:
        wandb.log({"evaluation_status": "completed"})
    except Exception as e:
        wandb.log({"evaluation_status": "failed", "error": str(e)})
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
