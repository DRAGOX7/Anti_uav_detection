# ============================================================
# Anti-UAV Drone Detection — Makefile
# ============================================================
# Usage examples:
#   make setup          — install all dependencies
#   make download       — download datasets (DUT only, no API key needed)
#   make download-all   — download DUT + Roboflow (needs ROBOFLOW_API_KEY)
#   make split          — create 70/15/15 data splits
#   make train-yolo     — train YOLOv11 with best HP config
#   make train-rtdetr   — train RT-DETR with best HP config
#   make train-all      — train all architectures × all HP configs
#   make evaluate       — run final test set evaluation
#   make app            — launch Streamlit UI
#   make mlflow         — launch MLflow tracking UI
#   make test           — run unit + integration tests
#   make lint           — run ruff linter
#   make security       — run bandit + safety + pip-audit
#   make clean          — remove generated files
# ============================================================

PYTHON      := python3
DATA_ROOT   := data
MLFLOW_URI  := mlflow/mlruns
CONFIG_DIR  := configs

.PHONY: all setup download download-all split \
        train-yolo train-rtdetr train-yolov8 train-all \
        evaluate track app mlflow \
        test lint format security \
        notebooks clean help

# ── Default ─────────────────────────────────────────────────────────────
all: help

# ── Setup ───────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install
	@echo "✅ Setup complete."

# ── Data ────────────────────────────────────────────────────────────────
download:
	$(PYTHON) -m src.data.download_dataset --dut-only --merge --data-root $(DATA_ROOT)

download-all:
	$(PYTHON) -m src.data.download_dataset --all --data-root $(DATA_ROOT)

download-tracking:
	$(PYTHON) -m src.data.download_dataset --tracking --data-root $(DATA_ROOT)

split:
	$(PYTHON) -m src.data.split_data \
	    --data-root $(DATA_ROOT) \
	    --train 0.70 --val 0.15 --seed 42

split-dut-fixed:
	$(PYTHON) -m src.data.split_data \
	    --data-root $(DATA_ROOT) \
	    --use-dut-fixed-splits \
	    --seed 42

verify-data:
	$(PYTHON) -m src.data.download_dataset --verify --data-root $(DATA_ROOT)

# ── Training ────────────────────────────────────────────────────────────
train-yolo:
	$(PYTHON) -m src.training.train \
	    --config $(CONFIG_DIR)/yolov11_hp2.yaml \
	    --mlflow-uri $(MLFLOW_URI) \
	    --experiment yolov11-anti-uav

train-yolo-all:
	for cfg in yolov11_hp1 yolov11_hp2 yolov11_hp3; do \
	    $(PYTHON) -m src.training.train \
	        --config $(CONFIG_DIR)/$$cfg.yaml \
	        --mlflow-uri $(MLFLOW_URI) \
	        --experiment yolov11-anti-uav; \
	done

train-rtdetr:
	$(PYTHON) -m src.training.train \
	    --config $(CONFIG_DIR)/rtdetr_hp2.yaml \
	    --mlflow-uri $(MLFLOW_URI) \
	    --experiment rtdetr-anti-uav

train-rtdetr-all:
	for cfg in rtdetr_hp1 rtdetr_hp2 rtdetr_hp3; do \
	    $(PYTHON) -m src.training.train \
	        --config $(CONFIG_DIR)/$$cfg.yaml \
	        --mlflow-uri $(MLFLOW_URI) \
	        --experiment rtdetr-anti-uav; \
	done

train-yolov8:
	$(PYTHON) -m src.training.train \
	    --config $(CONFIG_DIR)/yolov8_hp2.yaml \
	    --mlflow-uri $(MLFLOW_URI) \
	    --experiment yolov8-anti-uav

train-all: train-yolo-all train-rtdetr-all train-yolov8

# ── Evaluation ──────────────────────────────────────────────────────────
evaluate:
	$(PYTHON) -m src.evaluation.evaluate \
	    --data-root $(DATA_ROOT) \
	    --mlflow-uri $(MLFLOW_URI) \
	    --split test

evaluate-best:
	$(PYTHON) -m src.evaluation.evaluate \
	    --data-root $(DATA_ROOT) \
	    --mlflow-uri $(MLFLOW_URI) \
	    --split test \
	    --load-from-registry \
	    --stage Production

# ── Tracking ────────────────────────────────────────────────────────────
track:
	$(PYTHON) -m src.models.tracker.bytetrack \
	    --model-path runs/train/best.pt \
	    --sequences-dir $(DATA_ROOT)/raw/dut_tracking/DUT-Anti-UAV-Tracking \
	    --output-dir runs/tracking

# ── Hyperparameter optimisation ─────────────────────────────────────────
hpo-yolo:
	$(PYTHON) -m src.training.hyperparams \
	    --arch yolov11 \
	    --n-trials 20 \
	    --mlflow-uri $(MLFLOW_URI)

# ── Application ─────────────────────────────────────────────────────────
app:
	streamlit run app/streamlit_app.py \
	    --server.port 8501 \
	    --server.headless true

mlflow:
	mlflow ui \
	    --backend-store-uri $(MLFLOW_URI) \
	    --host 0.0.0.0 \
	    --port 5000

# ── Testing ─────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v --timeout=300

# ── Code quality ────────────────────────────────────────────────────────
lint:
	ruff check src/ app/ tests/

format:
	ruff format src/ app/ tests/

format-check:
	ruff format --check src/ app/ tests/

security:
	@echo "--- bandit (SAST) ---"
	bandit -r src/ app/ -c pyproject.toml
	@echo ""
	@echo "--- safety (CVE scan) ---"
	safety check -r requirements.txt
	@echo ""
	@echo "--- pip-audit ---"
	pip-audit -r requirements.txt

pre-commit-run:
	pre-commit run --all-files

# ── Notebooks ───────────────────────────────────────────────────────────
notebooks:
	jupyter nbconvert --to notebook --execute \
	    notebooks/01_EDA_and_DataPrep.ipynb \
	    --output notebooks/01_EDA_and_DataPrep_executed.ipynb

# ── Clean ───────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	@echo "Clean complete."

clean-data:
	rm -rf data/merged data/splits data/dataset.yaml
	@echo "Merged data and splits removed."

# ── Help ────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Anti-UAV Drone Detection — Available commands"
	@echo "  ──────────────────────────────────────────────"
	@echo "  make setup           Install all dependencies"
	@echo "  make download        Download DUT Anti-UAV dataset"
	@echo "  make download-all    Download DUT + Roboflow datasets"
	@echo "  make split           Create 70/15/15 data splits"
	@echo "  make train-yolo      Train YOLOv11 (best HP config)"
	@echo "  make train-rtdetr    Train RT-DETR (best HP config)"
	@echo "  make train-all       Train all architectures × all HP configs"
	@echo "  make evaluate        Run test set evaluation + MLflow logging"
	@echo "  make track           Run ByteTrack on DUT tracking sequences"
	@echo "  make app             Launch Streamlit UI (port 8501)"
	@echo "  make mlflow          Launch MLflow UI (port 5000)"
	@echo "  make test            Run test suite with coverage"
	@echo "  make lint            Run ruff linter"
	@echo "  make security        Run bandit + safety + pip-audit"
	@echo "  make clean           Remove cache files"
	@echo ""
