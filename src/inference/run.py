import wandb
import os
from datetime import datetime

def main():
    run_name = f"inference_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mlops_group8_spotify_prediction"),
        entity=os.environ.get("WANDB_ENTITY", "hiroinie"),
        job_type="inference",
        name=run_name,
    )
    
    try:
        wandb.log({"inference_status": "completed"})
    except Exception as e:
        wandb.log({"inference_status": "failed", "error": str(e)})
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
