import wandb
import os
from datetime import datetime
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.inference.inferencer import run_inference

def main():
    run_name = f"inference_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mlops_group8_spotify_prediction"),
        entity=os.environ.get("WANDB_ENTITY", "hiroinie"),
        job_type="inference",
        name=run_name,
    )
    
    try:
        run_inference(
            input_csv_path="data/raw/Songs_2025.csv",
            config_path="../../config.yaml", 
            output_csv_path="data/predictions/inference_output.csv"
        )
        wandb.log({"inference_status": "completed"})
    except Exception as e:
        wandb.log({"inference_status": "failed", "error": str(e)})
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()
