"""
src/evaluation/evaluate.py
============================
Test-set evaluation for all trained architectures.

Computes and logs to MLflow:
  - mAP@50, mAP@50-95
  - Precision, Recall, F1-score (per-class and overall)
  - Inference FPS
  - Model parameter count and file size
  - Confusion matrix

Usage
-----
    # Evaluate all best models on the test set
    python -m src.evaluation.evaluate

    # Evaluate a specific model
    python -m src.evaluation.evaluate \
        --weights runs/train/yolov11_hp2/weights/best.pt \
        --arch yolov11 --split test

    # Via Makefile
    make evaluate
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


def model_stats(weights_path: Path) -> dict[str, float]:
    """Return model size (MB) and parameter count."""
    stats: dict[str, float] = {}
    if weights_path.exists():
        stats["model_size_mb"] = round(weights_path.stat().st_size / (1024**2), 2)
    try:
        from ultralytics import RTDETR, YOLO  # noqa: PLC0415

        if "rtdetr" in weights_path.stem.lower():
            model = RTDETR(str(weights_path))
        else:
            model = YOLO(str(weights_path))
        n_params = sum(p.numel() for p in model.model.parameters())
        stats["n_parameters_M"] = round(n_params / 1e6, 2)
    except Exception as exc:
        log.warning("Could not count parameters: %s", exc)
    return stats


def measure_fps(
    weights_path: Path,
    imgsz: int = 640,
    n_warmup: int = 10,
    n_bench: int = 100,
    device: str = "0",
) -> float:
    """
    Measure inference FPS on a random image.
    Returns FPS (float). Returns 0.0 on failure.
    """
    try:
        from ultralytics import RTDETR, YOLO  # noqa: PLC0415

        if "rtdetr" in weights_path.stem.lower():
            model = RTDETR(str(weights_path))
        else:
            model = YOLO(str(weights_path))

        dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
        # Warmup
        for _ in range(n_warmup):
            model.predict(dummy, verbose=False, device=device)
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(n_bench):
            model.predict(dummy, verbose=False, device=device)
        elapsed = time.perf_counter() - t0
        return round(n_bench / elapsed, 1)
    except Exception as exc:
        log.warning("FPS measurement failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    weights_path: Path,
    dataset_yaml: Path,
    arch: str,
    split: str = "test",
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = "0",
    save_plots: bool = True,
    output_dir: Path | None = None,
) -> dict[str, float]:
    """
    Evaluate a trained model on the test set.
    Returns dictionary of metric_name -> value.
    """
    from ultralytics import RTDETR, YOLO  # noqa: PLC0415

    output_dir = output_dir or (
        PROJECT_ROOT / "runs" / "evaluation" / weights_path.parent.parent.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Evaluating: %s", weights_path.name)
    log.info("  Split:   %s", split)
    log.info("  ImgSz:   %d", imgsz)
    log.info("  Conf:    %.3f", conf)
    log.info("  IoU:     %.3f", iou)
    log.info("=" * 60)

    # Load model
    if "rtdetr" in arch.lower():
        model = RTDETR(str(weights_path))
    else:
        model = YOLO(str(weights_path))

    # Run validation on the specified split
    metrics_result = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        plots=save_plots,
        save_json=True,
        project=str(output_dir),
        name="val_results",
        exist_ok=True,
        verbose=True,
    )

    # Extract metrics from Ultralytics results object
    metrics: dict[str, float] = {}
    if metrics_result is not None:
        rd = metrics_result.results_dict if hasattr(metrics_result, "results_dict") else {}
        metrics = {
            "test/mAP50": rd.get("metrics/mAP50(B)", 0.0),
            "test/mAP50_95": rd.get("metrics/mAP50-95(B)", 0.0),
            "test/precision": rd.get("metrics/precision(B)", 0.0),
            "test/recall": rd.get("metrics/recall(B)", 0.0),
            "test/F1": compute_f1(
                rd.get("metrics/precision(B)", 0.0),
                rd.get("metrics/recall(B)", 0.0),
            ),
        }

    # Model size and parameter count
    stats = model_stats(weights_path)
    metrics.update(stats)

    # FPS benchmark
    fps = measure_fps(weights_path, imgsz=imgsz, device=device)
    metrics["inference_fps"] = fps

    # Log results
    log.info("Test set results:")
    for k, v in metrics.items():
        log.info("  %-25s = %.4f", k, v)

    return metrics


# ---------------------------------------------------------------------------
# Batch evaluation — all architectures
# ---------------------------------------------------------------------------


def evaluate_all(
    runs_dir: Path,
    dataset_yaml: Path,
    mlflow_uri: str,
    split: str = "test",
) -> list[dict]:
    """
    Find all best.pt files in runs_dir, evaluate each, and log to MLflow.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("final-evaluation")

    # Find all trained models
    weight_files = sorted(runs_dir.rglob("best.pt"))
    if not weight_files:
        log.error("No best.pt files found in %s", runs_dir)
        return []

    all_results = []

    for weights_path in weight_files:
        run_name = weights_path.parent.parent.name  # e.g. yolov11_hp2
        arch = (
            "rtdetr" if "rtdetr" in run_name else "yolov11" if "yolov11" in run_name else "yolov8"
        )

        # Read imgsz from the run's training args if available
        args_yaml = weights_path.parent.parent / "args.yaml"
        imgsz = 640
        if args_yaml.exists():
            with open(args_yaml) as f:
                args = yaml.safe_load(f) or {}
            imgsz = args.get("imgsz", 640)

        with mlflow.start_run(run_name=f"eval-{run_name}") as run:
            mlflow.log_params(
                {
                    "model": run_name,
                    "architecture": arch,
                    "split": split,
                    "imgsz": imgsz,
                    "weights": str(weights_path),
                }
            )

            metrics = evaluate_model(
                weights_path=weights_path,
                dataset_yaml=dataset_yaml,
                arch=arch,
                split=split,
                imgsz=imgsz,
                device="0" if torch.cuda.is_available() else "cpu",
                save_plots=True,
                output_dir=PROJECT_ROOT / "runs" / "evaluation" / run_name,
            )

            mlflow.log_metrics(metrics)

            # Log confusion matrix and plots if they were saved
            eval_dir = PROJECT_ROOT / "runs" / "evaluation" / run_name / "val_results"
            for plot in eval_dir.rglob("*.png") if eval_dir.exists() else []:
                mlflow.log_artifact(str(plot), "evaluation_plots")

            mlflow.set_tag("weights_path", str(weights_path))
            log.info("Logged to MLflow run: %s", run.info.run_id)

        record = {"model": run_name, "arch": arch, **metrics}
        all_results.append(record)

    return all_results


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table — Table 4 in the report."""
    if not results:
        return

    import pandas as pd  # noqa: PLC0415

    df = pd.DataFrame(results)
    cols = [
        "model",
        "arch",
        "test/mAP50",
        "test/mAP50_95",
        "test/precision",
        "test/recall",
        "test/F1",
        "inference_fps",
        "model_size_mb",
        "n_parameters_M",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    df_display = df[existing_cols].copy()

    # Round numerics
    for c in df_display.select_dtypes("float").columns:
        df_display[c] = df_display[c].round(4)

    print("\n" + "=" * 90)
    print("TABLE 4: Test Set Evaluation Results — All Best Models")
    print("=" * 90)
    print(df_display.to_string(index=False))
    print("=" * 90)

    # Save to CSV
    csv_path = PROJECT_ROOT / "reports" / "table4_test_set_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info("Saved results table -> %s", csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained Anti-UAV detection models.")
    p.add_argument("--weights", default=None, help="Path to specific best.pt")
    p.add_argument("--arch", default=None, help="Architecture name (yolov11|rtdetr|yolov8)")
    p.add_argument("--split", default="test", help="Dataset split to evaluate (test|val)")
    p.add_argument("--data", default=None, help="Path to dataset.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs" / "train"))
    p.add_argument("--mlflow-uri", default=None)
    p.add_argument(
        "--evaluate-all", action="store_true", help="Evaluate all best.pt files in runs-dir"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mlflow_uri = args.mlflow_uri or str(PROJECT_ROOT / "mlflow" / "mlruns")
    dataset_yaml = Path(args.data) if args.data else PROJECT_ROOT / "data" / "dataset.yaml"

    if not dataset_yaml.exists():
        log.error("dataset.yaml not found at %s — run split_data.py first.", dataset_yaml)
        sys.exit(1)

    if args.evaluate_all or args.weights is None:
        # Evaluate every trained model
        results = evaluate_all(
            runs_dir=Path(args.runs_dir),
            dataset_yaml=dataset_yaml,
            mlflow_uri=mlflow_uri,
            split=args.split,
        )
        print_comparison_table(results)

    else:
        # Evaluate a specific model
        weights = Path(args.weights)
        if not weights.exists():
            log.error("Weights not found: %s", weights)
            sys.exit(1)

        metrics = evaluate_model(
            weights_path=weights,
            dataset_yaml=dataset_yaml,
            arch=args.arch or "yolov11",
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device="0" if torch.cuda.is_available() else "cpu",
        )
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
