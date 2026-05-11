"""
src/training/hyperparams.py
============================
Dataclass definitions for all hyperparameter configurations,
plus a YAML loader that validates and returns a TrainingConfig.

Each HP combination is stored as a YAML file in configs/:
    configs/yolov11_hp1.yaml  — baseline (SGD, lr=1e-2, standard aug)
    configs/yolov11_hp2.yaml  — best     (AdamW, lr=1e-3, Mosaic+Mixup)
    configs/yolov11_hp3.yaml  — high-res (AdamW, lr=1e-3, img=1280)
    configs/rtdetr_hp1.yaml   — baseline
    configs/rtdetr_hp2.yaml   — best
    configs/rtdetr_hp3.yaml   — larger input
    configs/yolov8_hp1.yaml   — baseline
    configs/yolov8_hp2.yaml   — best
    configs/yolov8_hp3.yaml   — high-res
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    Unified configuration for all supported architectures.
    Loaded from a YAML file — each field maps to a YAML key.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    architecture:    str  = "yolov11"      # yolov11 | yolov8 | rtdetr
    model_name:      str  = "yolov11s"     # exact model variant
    hp_combination:  str  = "hp1"          # hp1 | hp2 | hp3 (for report tables)
    experiment_name: str  = "anti-uav"     # MLflow experiment

    # ── Data ──────────────────────────────────────────────────────────────
    dataset_yaml:    str  = "data/dataset.yaml"
    imgsz:           int  = 640
    workers:         int  = 8
    cache:           bool = False          # cache images to RAM (needs ~8GB+)
    fraction:        float = 1.0           # Ultralytics: fraction of dataset to train on (0, 1]

    # ── Training schedule ─────────────────────────────────────────────────
    epochs:          int   = 100
    batch_size:      int   = 16
    patience:        int   = 50           # early stopping patience (epochs)
    save_period:     int   = 10           # save checkpoint every N epochs

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer:       str   = "SGD"        # SGD | Adam | AdamW
    lr:              float = 1e-2         # initial learning rate (lr0)
    lrf:             float = 0.01         # final lr = lr0 * lrf
    momentum:        float = 0.937        # SGD momentum / Adam beta1
    weight_decay:    float = 5e-4
    warmup_epochs:   float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr:  float = 0.1
    amp:             bool  = True         # automatic mixed precision

    # ── Augmentation ──────────────────────────────────────────────────────
    mosaic:      float = 1.0    # mosaic augmentation probability
    mixup:       float = 0.0    # mixup augmentation probability
    copy_paste:  float = 0.0    # copy-paste augmentation probability
    degrees:     float = 0.0    # rotation degrees
    translate:   float = 0.1    # translation fraction
    scale:       float = 0.5    # scale gain
    shear:       float = 0.0    # shear degrees
    perspective: float = 0.0    # perspective distortion
    flipud:      float = 0.0    # vertical flip probability
    fliplr:      float = 0.5    # horizontal flip probability
    hsv_h:       float = 0.015  # HSV hue augmentation
    hsv_s:       float = 0.7    # HSV saturation augmentation
    hsv_v:       float = 0.4    # HSV value augmentation
    erasing:     float = 0.4    # random erasing probability (YOLOv11)

    # ── Hardware ──────────────────────────────────────────────────────────
    device:      str = "0"      # GPU index, "cpu", or "mps"
    seed:        int = 42

    # ── Output ────────────────────────────────────────────────────────────
    output_dir:  str = "runs/train"
    pretrained:  bool = True    # use COCO-pretrained weights

    # ── Validation override ───────────────────────────────────────────────
    conf_threshold: float = 0.001  # detection confidence threshold for val
    iou_threshold:  float = 0.6    # NMS IoU threshold

    def __post_init__(self) -> None:
        """Resolve relative paths to absolute."""
        if not os.path.isabs(self.dataset_yaml):
            # Try to resolve relative to project root
            project_root = Path(__file__).resolve().parents[2]
            resolved = project_root / self.dataset_yaml
            if resolved.exists():
                self.dataset_yaml = str(resolved)

    @property
    def summary(self) -> str:
        """Human-readable one-line summary for logging."""
        return (
            f"{self.architecture} | {self.hp_combination} | "
            f"lr={self.lr} batch={self.batch_size} "
            f"imgsz={self.imgsz} opt={self.optimizer} "
            f"mosaic={self.mosaic} mixup={self.mixup}"
        )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_config(path: Path) -> TrainingConfig:
    """
    Load a TrainingConfig from a YAML file.
    Unknown keys are silently ignored (forward compatibility).
    Missing keys use dataclass defaults.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Only pass known fields to the dataclass
    known = {f.name for f in fields(TrainingConfig)}
    filtered = {k: v for k, v in raw.items() if k in known}

    cfg = TrainingConfig(**filtered)

    # Validate critical fields
    _validate_config(cfg, path)

    return cfg


def _validate_config(cfg: TrainingConfig, path: Path) -> None:
    """Raise ValueError for obviously invalid configurations."""
    if cfg.architecture.lower() not in {"yolov11", "yolov8", "rtdetr", "fasterrcnn"}:
        raise ValueError(
            f"Unknown architecture '{cfg.architecture}' in {path}. "
            f"Choose from: yolov11, yolov8, rtdetr,fasterrcnn"
        )
    if not (0 < cfg.lr < 1):
        raise ValueError(f"lr={cfg.lr} is out of sensible range (0, 1)")
    if cfg.epochs < 1:
        raise ValueError(f"epochs must be ≥ 1, got {cfg.epochs}")
    if cfg.batch_size < 1:
        raise ValueError(f"batch_size must be ≥ 1, got {cfg.batch_size}")
    if cfg.imgsz not in {320, 416, 512, 640, 800, 1024, 1280}:
        raise ValueError(
            f"imgsz={cfg.imgsz} is unusual. Typical values: 416, 640, 1280."
        )
    if not (0 < float(cfg.fraction) <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {cfg.fraction}")


def dump_config(cfg: TrainingConfig, path: Path) -> None:
    """Save a TrainingConfig to YAML (useful for logging reproducibility)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=True)