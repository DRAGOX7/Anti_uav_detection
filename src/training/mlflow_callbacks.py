import mlflow
from ultralytics.utils.callbacks import add_integration_callbacks

def start_mlflow_run(experiment_name, run_name, tracking_uri, tags=None):
    """Initializes an MLflow run and returns the run_id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    run = mlflow.start_run(run_name=run_name, tags=tags)
    return run.info.run_id

def add_mlflow_callbacks(model, run_id, architecture, hp_combo, extra_params=None):
    """Attaches Ultralytics callbacks to log directly to an active MLflow run."""
    # Ensure Ultralytics has MLflow integrated
    add_integration_callbacks(model)
    
    # Custom metadata logging
    with mlflow.start_run(run_id=run_id, nested=True):
        params = {
            "architecture": architecture,
            "hp_combination": hp_combo,
            "model_variant": model.ckpt_path if hasattr(model, 'ckpt_path') else "yolo11s"
        }
        if extra_params:
            params.update(dict(extra_params))
        mlflow.log_params(params)