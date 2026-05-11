"""
src/training/train.py
======================
CLI entry point for training any architecture with any HP configuration.

Every run is fully tracked in MLflow:
  - Hyperparameters logged at start
  - Per-epoch metrics logged during training
  - Best weights saved as artifacts
  - Model registered in MLflow Model Registry

Usage
-----
    # Train YOLOv11 with the best HP config
    python -m src.training.train --config configs/yolov11_hp2.yaml

    # Train RT-DETR, log to remote MLflow server
    python -m src.training.train \\
        --config configs/rtdetr_hp1.yaml \\
        --mlflow-uri http://localhost:5000 \\
        --experiment rtdetr-anti-uav

    # Quick smoke test (5 epochs, tiny batch)
    python -m src.training.train \\
        --config configs/yolov11_hp1.yaml \\
        --epochs 5 --batch 4

    # Via Makefile shorthand:
    make train-yolo
    make train-rtdetr
    make train-all
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict

import mlflow
import torch
import yaml

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.mlflow_callbacks import add_mlflow_callbacks, start_mlflow_run
from src.training.hyperparams import load_config, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture dispatch
# ---------------------------------------------------------------------------

SUPPORTED_ARCHS = {"yolov11", "yolov8", "rtdetr"}


RTDETR_MODEL_MAP = {
    "rtdetr-r18": "rtdetr-resnet18.pt",
    "rtdetr-r34": "rtdetr-resnet34.pt",
    "rtdetr-r50": "rtdetr-resnet50.pt",
    "rtdetr-r101": "rtdetr-resnet101.pt",
}


def resolve_rtdetr_weights(model_name: str, *, default: str = "rtdetr-r50.pt") -> str:
    """Resolve a friendly RT-DETR model key (e.g. 'rtdetr-r18') to a weights filename.

    If `model_name` is not a known key, it is returned as-is. This allows passing
    a direct Ultralytics weights string/path.
    """
    if not model_name:
        return default
    return RTDETR_MODEL_MAP.get(model_name, model_name)


def train_ultralytics(cfg: TrainingConfig, run_id: str) -> Dict[str, float]:
    """
    Train a YOLOv11 or YOLOv8 model using the Ultralytics API.
    Returns dict of final validation metrics.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    project_root = Path(__file__).resolve().parents[2]

    # Map config arch name to Ultralytics model string
    MODEL_MAP = {
        "yolov11s": "yolo11s.pt",
        "yolov11m": "yolo11m.pt",
        "yolov11l": "yolo11l.pt",
        "yolov8s":  "yolov8s.pt",
        "yolov8m":  "yolov8m.pt",
        "yolov8l":  "yolov8l.pt",
        # Custom lightweight Transformer-style backbone
        "yolov11-mobilevit-s": str(project_root / "configs" / "models" / "yolo11_mobilevit_s.yaml"),
    }

    model_weights = MODEL_MAP.get(cfg.model_name, cfg.model_name)
    if str(model_weights).endswith(".yaml") and "mobilevit" in str(model_weights).lower():
        # Register custom modules so Ultralytics can eval() them while parsing YAML
        try:
            import ultralytics.nn.tasks as tasks  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Unable to import ultralytics.nn.tasks for YAML parsing") from e

        from src.models.mobilevit_yolo import MobileViTFeatures  # noqa: PLC0415

        setattr(tasks, "MobileViTFeatures", MobileViTFeatures)

    log.info("Loading model: %s", model_weights)

    model = YOLO(model_weights)

    # Attach MLflow callbacks BEFORE model.train() is called
    add_mlflow_callbacks(
        model=model,
        run_id=run_id,
        architecture=cfg.architecture,
        hp_combo=cfg.hp_combination,
        extra_params={
            "model_name":  cfg.model_name,
            "git_sha":     _get_git_sha(),
            "python":      platform.python_version(),
            "torch":       torch.__version__,
            "cuda":        torch.version.cuda or "cpu",
            "gpu_name":    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    )

    # Train
    log.info("=" * 60)
    log.info("Starting %s training — HP combo: %s", cfg.architecture.upper(), cfg.hp_combination)
    log.info("  Epochs:     %d", cfg.epochs)
    log.info("  Batch:      %d", cfg.batch_size)
    log.info("  Image size: %d", cfg.imgsz)
    log.info("  Optimizer:  %s", cfg.optimizer)
    log.info("  LR:         %g", cfg.lr)
    log.info("=" * 60)

    train_kwargs = {
        "data":         cfg.dataset_yaml,
        "fraction":     cfg.fraction,
        "epochs":       cfg.epochs,
        "imgsz":        cfg.imgsz,
        "batch":        cfg.batch_size,
        "optimizer":    cfg.optimizer,
        "lr0":          cfg.lr,
        "lrf":          cfg.lrf,
        "weight_decay": cfg.weight_decay,
        "momentum":     cfg.momentum,
        "warmup_epochs":cfg.warmup_epochs,
        "mosaic":       cfg.mosaic,
        "mixup":        cfg.mixup,
        "degrees":      cfg.degrees,
        "translate":    cfg.translate,
        "scale":        cfg.scale,
        "flipud":       cfg.flipud,
        "fliplr":       cfg.fliplr,
        "hsv_h":        cfg.hsv_h,
        "hsv_s":        cfg.hsv_s,
        "hsv_v":        cfg.hsv_v,
        "device":       cfg.device,
        "workers":      cfg.workers,
        "patience":     cfg.patience,
        "save":         True,
        "save_period":  cfg.save_period,
        "project":      cfg.output_dir,
        "name":         f"{cfg.architecture}_{cfg.hp_combination}",
        "exist_ok":     True,
        "pretrained":   cfg.pretrained,
        "verbose":      True,
        "plots":        True,
        "val":          True,
        "amp":          cfg.amp,
        "cache":        cfg.cache,
        "seed":         cfg.seed,
    }

    results = model.train(**train_kwargs)

    # Extract final metrics from results
    metrics = {}
    if results and hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                pass

    log.info("Training complete. Best mAP50: %.4f", metrics.get("metrics/mAP50(B)", 0))
    return metrics


def train_rtdetr(cfg: TrainingConfig, run_id: str) -> Dict[str, float]:
    """
    Train RT-DETR using the Ultralytics RT-DETR implementation.
    Ultralytics ≥ 8.1 supports RT-DETR natively with the same API.
    """
    from ultralytics import RTDETR  # noqa: PLC0415
    from src.training.mlflow_callbacks import RTDETRMLflowLogger  # noqa: PLC0415

    model_weights = resolve_rtdetr_weights(cfg.model_name)

    log.info("Loading RT-DETR model: %s", model_weights)
    model = RTDETR(model_weights)

    # RT-DETR uses the same Ultralytics add_callback interface
    add_mlflow_callbacks(
        model=model,
        run_id=run_id,
        architecture=cfg.architecture,
        hp_combo=cfg.hp_combination,
        extra_params={
            "model_name": cfg.model_name,
            "git_sha":    _get_git_sha(),
            "torch":      torch.__version__,
        },
    )

    log.info("=" * 60)
    log.info("Starting RT-DETR training — HP combo: %s", cfg.hp_combination)
    log.info("=" * 60)

    results = model.train(
        data=cfg.dataset_yaml,
        fraction=cfg.fraction,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch_size,
        optimizer=cfg.optimizer,
        lr0=cfg.lr,
        lrf=cfg.lrf,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        device=cfg.device,
        workers=cfg.workers,
        patience=cfg.patience,
        save=True,
        project=cfg.output_dir,
        name=f"rtdetr_{cfg.hp_combination}",
        exist_ok=True,
        pretrained=cfg.pretrained,
        plots=True,
        amp=cfg.amp,
        seed=cfg.seed,
    )

    metrics = {}
    if results and hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                pass

    return metrics


# ---------------------------------------------------------------------------
# Post-training: log final test metrics to MLflow
# ---------------------------------------------------------------------------

def log_final_metrics(
    run_id:    str,
    metrics:   Dict[str, float],
    save_dir:  Path,
    cfg:       TrainingConfig,
) -> None:
    """Log final training metrics and model size to the MLflow run."""
    with mlflow.start_run(run_id=run_id):
        # Training summary metrics
        final = {}
        for k, v in metrics.items():
            try:
                final[f"final/{k.replace('metrics/', '')}"] = float(v)
            except (TypeError, ValueError):
                pass

        # Model file size
        best_pt = save_dir / f"{cfg.architecture}_{cfg.hp_combination}" / "weights" / "best.pt"
        if best_pt.exists():
            size_mb = best_pt.stat().st_size / (1024 ** 2)
            final["model_size_mb"] = round(size_mb, 2)
            log.info("Model size: %.1f MB", size_mb)

        if final:
            mlflow.log_metrics(final)

        # Tag run as completed
        mlflow.set_tags({
            "status":   "completed",
            "arch":     cfg.architecture,
            "hp_combo": cfg.hp_combination,
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_git_sha() -> str:
    """Return short git commit SHA, or 'unknown' if not in a repo."""
    import subprocess  # noqa: PLC0415
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _build_run_name(cfg: TrainingConfig) -> str:
    return f"{cfg.architecture}_{cfg.hp_combination}_{cfg.epochs}ep"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Anti-UAV detector with full MLflow tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file (e.g. configs/yolov11_hp2.yaml)",
    )
    p.add_argument("--mlflow-uri",    default=None, help="MLflow tracking URI")
    p.add_argument("--experiment",    default=None, help="MLflow experiment name (overrides config)")
    p.add_argument("--run-name",      default=None, help="MLflow run name (overrides auto-generated)")
    p.add_argument("--epochs",        type=int,  default=None, help="Override epochs from config")
    p.add_argument("--batch",         type=int,  default=None, help="Override batch size from config")
    p.add_argument("--device",        default=None, help="Override device (e.g. 0, cpu, mps)")
    p.add_argument("--data",          default=None, help="Override dataset YAML path")
    p.add_argument(
        "--fraction",
        type=float,
        default=None,
        help="Train on a fraction of the dataset (Ultralytics 'fraction', in (0, 1]). Useful for smoke tests.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    cfg = load_config(Path(args.config))

    # Apply CLI overrides
    if args.epochs  is not None: cfg.epochs      = args.epochs
    if args.batch   is not None: cfg.batch_size  = args.batch
    if args.device  is not None: cfg.device      = args.device
    if args.data    is not None: cfg.dataset_yaml = args.data
    if args.fraction is not None: cfg.fraction   = args.fraction

    # MLflow setup
    tracking_uri = args.mlflow_uri or os.environ.get(
        "MLFLOW_TRACKING_URI",
        str(Path(__file__).resolve().parents[2] / "mlflow" / "mlruns"),
    )
    # MLflow expects a URI scheme. On Windows, passing a raw path like
    # 'C:\\path\\to\\mlruns' is interpreted as scheme='c' and will fail.
    if "://" not in tracking_uri:
        p = Path(tracking_uri)
        # Only coerce existing local paths; allow empty string or other MLflow schemes.
        if p.exists() or (len(tracking_uri) >= 3 and tracking_uri[1:3] == ":\\"):
            tracking_uri = p.resolve().as_uri()
    experiment_name = args.experiment or cfg.experiment_name
    run_name        = args.run_name   or _build_run_name(cfg)

    log.info("MLflow URI:    %s", tracking_uri)
    log.info("Experiment:    %s", experiment_name)
    log.info("Run name:      %s", run_name)

    # Start MLflow run (returns run_id, run stays open via callbacks)
    run_id = start_mlflow_run(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags={
            "architecture": cfg.architecture,
            "hp_combo":     cfg.hp_combination,
        },
    )

    # Dispatch to correct training function
    arch = cfg.architecture.lower().replace("-", "")
    t0 = time.time()

    try:
        if "rtdetr" in arch:
            metrics = train_rtdetr(cfg, run_id)
        elif "yolo" in arch:
            metrics = train_ultralytics(cfg, run_id)
        else:
            log.error("Unsupported architecture: %s. Choose from: %s", cfg.architecture, SUPPORTED_ARCHS)
            sys.exit(1)

        # Log final summary
        output_dir = Path(cfg.output_dir)
        log_final_metrics(run_id, metrics, output_dir, cfg)

        elapsed = time.time() - t0
        log.info("=" * 60)
        log.info("DONE  |  %.1f min  |  Run ID: %s", elapsed / 60, run_id)
        log.info("View in MLflow: mlflow ui --backend-store-uri %s", tracking_uri)
        log.info("=" * 60)

    except KeyboardInterrupt:
        log.warning("Training interrupted by user.")
        if mlflow.active_run():
            mlflow.set_tag("status", "interrupted")
        else:
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("status", "interrupted")
        sys.exit(130)

    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=True)
        if mlflow.active_run():
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(exc)[:500])
        else:
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", str(exc)[:500])
        sys.exit(1)

    finally:
        # Always end the MLflow run
        if mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    main()
