import pandas as pd
from pathlib import Path
import mlflow
import warnings
import yaml  # <-- New import to read YOLO's config file

warnings.simplefilter(action='ignore', category=FutureWarning)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# We use v3 to avoid the "deleted experiment" error again!
experiment_name = "AerialGuard_Final_Evaluation_v3"
mlflow.set_experiment(experiment_name)

runs_dir = Path("runs/train")
print(f"🔍 Scanning {runs_dir} for YOLO results...\n")

key_map = {
    "metrics/precision(B)": "val/precision",
    "metrics/recall(B)": "val/recall",
    "metrics/mAP50(B)": "val/mAP50",
    "metrics/mAP50-95(B)": "val/mAP50_95"
}

# ⚠️ Edit these fallback values to match your actual AI447 configs!
# This is used if args.yaml is missing (like for Faster R-CNN)
fallback_params = {
    "HP1": {"epochs": 100, "lr": 0.01,  "batch": 16, "optimizer": "SGD"},
    "HP2": {"epochs": 100, "lr": 0.001, "batch": 16, "optimizer": "AdamW"},
    "HP3": {"epochs": 100, "lr": 0.001, "batch": 8,  "optimizer": "AdamW"} # Batch 8 for 1280px resolution
}

for run_folder in runs_dir.iterdir():
    if not run_folder.is_dir():
        continue
        
    csv_path = run_folder / "results.csv"
    if not csv_path.exists():
        continue

    print(f"📦 Pushing data & parameters for: {run_folder.name}")
    
    arch = "YOLOv11-S" if "11" in run_folder.name else ("Faster R-CNN" if "rcnn" in run_folder.name.lower() else "YOLOv8-S")
    hp = "HP1" if "hp1" in run_folder.name else ("HP2" if "hp2" in run_folder.name else "HP3")

    with mlflow.start_run(run_name=run_folder.name):
        
        mlflow.log_param("architecture", arch)
        mlflow.log_param("hp_combination", hp)
        
        # ---------------------------------------------------------
        # NEW: Parse Hyperparameters
        # ---------------------------------------------------------
        args_path = run_folder / "args.yaml"
        if args_path.exists():
            # If YOLO's args.yaml exists, read parameters directly from it
            with open(args_path, 'r') as f:
                args = yaml.safe_load(f)
                mlflow.log_param("epochs", args.get("epochs", "?"))
                mlflow.log_param("lr", args.get("lr0", "?"))
                mlflow.log_param("batch", args.get("batch", "?"))
                
                # Optimizer sometimes saves as 'auto', so handle gracefully
                mlflow.log_param("optimizer", args.get("optimizer", "auto"))
        else:
            # If args.yaml is missing, use our hardcoded dictionary above
            params = fallback_params.get(hp, {})
            for key, value in params.items():
                mlflow.log_param(key, value)
        # ---------------------------------------------------------
        
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        for index, row in df.iterrows():
            epoch = int(row.get('epoch', index))
            metrics = {}
            for k, v in row.items():
                if k != 'epoch':
                    clean_key = key_map.get(k, k.replace('(', '_').replace(')', ''))
                    metrics[clean_key] = float(v)
                    
            mlflow.log_metrics(metrics, step=epoch)
            
        mlflow.log_artifact(str(csv_path))

print("\n✅ Retroactive logging v3 complete! Refresh your MLflow UI.")