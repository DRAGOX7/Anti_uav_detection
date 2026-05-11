---
title: Anti UAV Detection
sdk: docker
app_port: 8501
---

# 🛡️ Anti-UAV Drone Detection & Tracking System

**Course:** AI447 Computer Vision — Spring 2025-2026
**Task:** Developing a Computer Vision Deep Learning-based System

---

## Overview

Real-time detection, classification, and tracking of hostile UAVs (drones) in RGB video and images using YOLOv11, RT-DETR, and ByteTrack. Full MLOps pipeline with MLflow experiment tracking, CI/CD via GitHub Actions, and a 4-page Streamlit interface.

| Component                  | Detail                                                   |
| -------------------------- | -------------------------------------------------------- |
| **Primary architecture**   | YOLOv11-S (one-stage CNN)                                |
| **Secondary architecture** | RT-DETR-R50 (transformer)                                |
| **Baseline**               | YOLOv8-M                                                 |
| **Tracker**                | ByteTrack (bonus +5 marks)                               |
| **Training data**          | DUT Anti-UAV (10K) + Roboflow (2.5–5K)                   |
| **Experiment tracking**    | MLflow                                                   |
| **CI/CD**                  | GitHub Actions (lint → security → test → build → deploy) |

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/your-org/anti-uav-detection.git
cd anti-uav-detection
make setup                      # pip install -r requirements.txt + pre-commit
```

### 2. Download dataset

```bash
# DUT Anti-UAV only (free, no API key needed)
make download

# DUT + Roboflow (needs API key)
export ROBOFLOW_API_KEY=your_key_here
make download-all
```

> **Manual download:** If the automated download fails, visit
> https://github.com/wangdongdut/DUT-Anti-UAV and follow their instructions.
> Place the zip at `data/raw/dut_anti_uav/DUT-Anti-UAV-Detection.zip`
> then re-run `make download`.

### 3. Create data splits

```bash
make split                      # 70/15/15 stratified split, seed=42
```

### 4. Train models

```bash
# Generate all 9 YAML configs
python generate_configs.py

# Train YOLOv11 (all 3 HP combos sequentially)
make train-yolo-all

# Train RT-DETR (all 3 HP combos)
make train-rtdetr-all

# Or train a single config
python -m src.training.train --config configs/yolov11_hp2.yaml
```

### 5. Launch MLflow UI

```bash
make mlflow                     # http://localhost:5000
```

### 6. Launch Streamlit app

```bash
make app                        # http://localhost:8501
```

### 7. Run the full stack with Docker

```bash
docker-compose -f docker/docker-compose.yml up
# → Streamlit: http://localhost:8501
# → MLflow:    http://localhost:5000
```

---

## Project structure

```
anti-uav-detection/
│
├── notebooks/
│   ├── 01_EDA_and_DataPrep.ipynb       ← Start here
│   ├── 02_Preprocessing.ipynb
│   ├── 03_YOLOv11_Training.ipynb
│   ├── 04_RTDETR_Training.ipynb
│   ├── 05_YOLOv8_Baseline.ipynb
│   ├── 06_Tracking_ByteTrack.ipynb
│   ├── 07_Hyperparameter_Tuning.ipynb
│   └── 08_Final_Evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── download_dataset.py         Dataset download + format conversion
│   │   └── split_data.py               Stratified 70/15/15 split
│   ├── training/
│   │   ├── train.py                    CLI entry point (all architectures)
│   │   ├── mlflow_callbacks.py         Per-epoch MLflow logging
│   │   └── hyperparams.py              TrainingConfig dataclass + YAML loader
│   ├── models/
│   │   ├── yolov11.py                  YOLOv11 wrapper
│   │   ├── rtdetr.py                   RT-DETR wrapper
│   │   └── tracker/bytetrack.py        ByteTrack integration
│   └── evaluation/
│       ├── evaluate.py                 Test set evaluation
│       └── metrics.py                  mAP, precision, recall, F1
│
├── app/
│   ├── streamlit_app.py                Main app with sidebar navigation
│   └── pages/
│       ├── image_detection.py          Page 1: image upload + inference
│       ├── video_tracking.py           Page 2: video + ByteTrack (+5 bonus)
│       ├── model_comparison.py         Page 3: side-by-side MLflow comparison
│       └── mlflow_dashboard.py         Page 4: full experiment browser
│
├── configs/                            9 YAML files (3 archs × 3 HP combos)
│   ├── yolov11_hp1.yaml  hp2.yaml  hp3.yaml
│   ├── rtdetr_hp1.yaml   hp2.yaml  hp3.yaml
│   └── yolov8_hp1.yaml   hp2.yaml  hp3.yaml
│
├── tests/
│   ├── unit/test_data_pipeline.py      12 unit tests (offline, no GPU)
│   └── integration/                    End-to-end pipeline tests
│
├── docker/
│   ├── Dockerfile.serve                Streamlit serving image
│   └── docker-compose.yml              App + MLflow local stack
│
├── .github/workflows/
│   ├── ci.yml                          Lint → security → test → build
│   ├── cd.yml                          Deploy on merge to main
│   └── model-eval.yml                  Auto-promote best model
│
├── Makefile                            All commands in one place
├── pyproject.toml                      ruff + pytest + bandit config
├── requirements.txt                    Production dependencies
├── requirements-dev.txt                Dev + CI dependencies
└── generate_configs.py                 Writes all 9 YAML configs
```

---

## Hyperparameter configurations

All 3 HP combos per architecture are defined in `configs/`. Generate them with:

```bash
python generate_configs.py
```

| Combo | Optimizer | LR   | Batch | ImgSz | Mosaic | Mixup | Purpose  |
| ----- | --------- | ---- | ----- | ----- | ------ | ----- | -------- |
| HP1   | SGD       | 1e-2 | 16    | 640   | ✓      | ✗     | Baseline |
| HP2   | AdamW     | 1e-3 | 16    | 640   | ✓      | ✓     | **Best** |
| HP3   | AdamW     | 1e-3 | 8     | 1280  | ✓      | ✓     | High-res |

---

## Dataset

| Split         | Source                     | Images    |
| ------------- | -------------------------- | --------- |
| Train         | DUT Anti-UAV + Roboflow    | ~8,400    |
| Val           | Merged pool (stratified)   | ~1,800    |
| Test          | Merged pool (held-out)     | ~1,800    |
| Tracking eval | DUT 20 RGB video sequences | 20 videos |

---

## MLOps pipeline

```
Code push → GitHub Actions CI:
  ① ruff lint + format check
  ② bandit SAST + safety CVE scan + pip-audit
  ③ pytest unit tests (60%+ coverage required)
  ④ Docker build check

Merge to main → CD:
  ⑤ Pull best model from MLflow Registry
  ⑥ docker build + push to GHCR
  ⑦ Deploy Streamlit container
  ⑧ Health check

After training → model-eval.yml:
  ⑨ Compare new run vs current Production model
  ⑩ Auto-promote if mAP50 improves > 1%
```

---

## Running tests

```bash
make test                   # full test suite with coverage
make test-unit              # unit tests only (no GPU required)
make security               # bandit + safety + pip-audit
make lint                   # ruff check
```

---

## Limitations

- Single-class detection (drone vs background). Multi-class (drone type) is left as future work.
- RGB-only training; IR/thermal footage requires domain adaptation or re-training.
- ByteTrack requires consistent frame rate — degraded performance on very low FPS video.
- Tiny targets (<0.5% image area) remain challenging even at 1280px resolution.

---

## References

[1] Zhao et al., "Vision-Based Anti-UAV Detection and Tracking," IEEE TITS 2022.
[2] Wang et al., "YOLOv11: Advances in Real-Time Object Detection," Ultralytics 2024.
[3] Lv et al., "RT-DETR: DETRs Beat YOLOs on Real-time Object Detection," CVPR 2023.
[4] Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022.
[5] Jiang et al., "Anti-UAV: A Large-Scale Benchmark for Vision-Based UAV Tracking," IEEE TMM 2023.
