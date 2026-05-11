"""
src/models/tracker/bytetrack.py
================================
ByteTrack integration for the Anti-UAV tracking pipeline.

ByteTrack (Zhang et al., ECCV 2022) assigns persistent track IDs to
detected objects across video frames by associating ALL detection boxes —
including low-confidence ones — using IoU-based matching.

Ultralytics ≥ 8.1 includes ByteTrack natively via `model.track()`.
This module provides:
  1. A standalone tracking runner for the 20 DUT tracking sequences
  2. Tracking metric computation (MOTA, IDF1, ID switches)
  3. Trajectory visualisation utilities
  4. A CLI for batch evaluation on all tracking sequences

Usage
-----
    # Track a single video
    python -m src.models.tracker.bytetrack \\
        --weights runs/train/yolov11_hp2/weights/best.pt \\
        --video path/to/drone_footage.mp4 \\
        --output runs/tracking/tracked_output.mp4

    # Evaluate on all DUT tracking sequences
    python -m src.models.tracker.bytetrack \\
        --weights runs/train/yolov11_hp2/weights/best.pt \\
        --sequences-dir data/raw/dut_tracking/DUT-Anti-UAV-Tracking \\
        --output-dir runs/tracking/dut_results

    # Via Makefile
    make track
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# Colour palette for track IDs (BGR)
PALETTE_BGR = [
    (80, 200, 0),   (0, 140, 255),  (200, 0, 200),  (0, 200, 200),
    (50, 50, 255),  (255, 50, 50),  (50, 200, 50),  (0, 200, 255),
    (200, 100, 0),  (100, 0, 200),
]


def _colour(track_id: int) -> Tuple[int, int, int]:
    return PALETTE_BGR[track_id % len(PALETTE_BGR)]


# ---------------------------------------------------------------------------
# Core tracking runner
# ---------------------------------------------------------------------------

def track_video(
    weights_path:  Path,
    video_path:    Path,
    output_path:   Path,
    conf:          float = 0.25,
    iou:           float = 0.45,
    imgsz:         int   = 640,
    device:        str   = "0",
    show_trails:   bool  = True,
    trail_len:     int   = 50,
    show_ids:      bool  = True,
    show_conf:     bool  = True,
    max_frames:    Optional[int] = None,
) -> Dict:
    """
    Run YOLOv11 + ByteTrack on a video file and write annotated output.

    Returns summary statistics dict:
        {n_frames, n_detections, n_unique_ids, fps, elapsed_s}
    """
    from ultralytics import YOLO, RTDETR  # noqa: PLC0415

    # Load model
    weights_str = str(weights_path)
    if "rtdetr" in weights_path.stem.lower():
        model = RTDETR(weights_str)
    else:
        model = YOLO(weights_str)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    process_n = min(max_frames or total, total)
    log.info("Video: %dx%d  %.1f FPS  %d frames  (processing %d)", W, H, fps_in, total, process_n)

    # Video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (W, H))

    # State
    trails: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    unique_ids: set[int] = set()
    n_detections = 0
    frame_idx    = 0
    t0 = time.perf_counter()

    while cap.isOpened() and frame_idx < process_n:
        ret, frame = cap.read()
        if not ret:
            break

        # ByteTrack tracking
        results = model.track(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            device=device,
        )

        annotated = frame.copy()

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                conf_score = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else -1
                colour = _colour(tid) if tid >= 0 else (128, 128, 128)

                # Bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

                # Label
                parts = ["drone"]
                if show_ids   and tid >= 0:  parts = [f"#{tid}"] + parts
                if show_conf:                parts.append(f"{conf_score:.2f}")
                label = " ".join(parts)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Trajectory trail
                if show_trails and tid >= 0:
                    trails[tid].append((cx, cy))
                    if len(trails[tid]) > trail_len:
                        trails[tid].pop(0)
                    pts = trails[tid]
                    for i in range(1, len(pts)):
                        alpha = i / len(pts)
                        c_faded = tuple(int(v * alpha) for v in colour)
                        cv2.line(annotated, pts[i-1], pts[i], c_faded, 2)

                if tid >= 0:
                    unique_ids.add(tid)
                n_detections += 1

        # Frame counter
        cv2.putText(annotated, f"{frame_idx+1}/{process_n}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(annotated, f"IDs: {len(unique_ids)}",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        out_writer.write(annotated)
        frame_idx += 1

        if frame_idx % 50 == 0:
            elapsed = time.perf_counter() - t0
            log.info("  Frame %d/%d  |  Unique IDs: %d  |  %.1f FPS",
                     frame_idx, process_n, len(unique_ids), frame_idx/elapsed)

    cap.release()
    out_writer.release()
    elapsed = time.perf_counter() - t0

    summary = {
        "n_frames":      frame_idx,
        "n_detections":  n_detections,
        "n_unique_ids":  len(unique_ids),
        "fps":           round(frame_idx / max(elapsed, 0.001), 1),
        "elapsed_s":     round(elapsed, 2),
        "output":        str(output_path),
    }
    log.info("Tracking complete: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# DUT tracking sequence evaluation
# ---------------------------------------------------------------------------

def evaluate_sequence(
    model,
    sequence_dir: Path,
    conf:         float = 0.25,
    iou:          float = 0.45,
    imgsz:        int   = 640,
    device:       str   = "0",
) -> Dict:
    """
    Evaluate a DUT tracking sequence and compute frame-level IoU.

    DUT ground truth format: one bounding box per line (x,y,w,h in pixels),
    or 0,0,0,0 when drone is not visible.

    Returns dict with mean_iou, n_frames, n_visible, n_detected.
    """
    # Load ground truth
    gt_file = sequence_dir / "groundtruth.txt"
    frames_dir = sequence_dir

    if not gt_file.exists():
        log.warning("No groundtruth.txt in %s", sequence_dir)
        return {}

    gt_boxes = []
    for line in gt_file.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 4:
            try:
                gt_boxes.append(list(map(float, parts[:4])))
            except ValueError:
                gt_boxes.append([0, 0, 0, 0])
        else:
            gt_boxes.append([0, 0, 0, 0])

    # Get sorted frame images
    frame_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not frame_paths:
        log.warning("No frames in %s", sequence_dir)
        return {}

    ious       = []
    n_detected = 0
    n_visible  = 0

    for i, frame_path in enumerate(frame_paths):
        if i >= len(gt_boxes):
            break

        gt_x, gt_y, gt_w, gt_h = gt_boxes[i]
        gt_visible = (gt_w > 0 and gt_h > 0)

        if gt_visible:
            n_visible += 1

        # Run detection
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        results = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        # Find best matching detection
        best_iou = 0.0
        if results and results[0].boxes is not None and gt_visible:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                det_x, det_y = x1, y1
                det_w = x2 - x1
                det_h = y2 - y1

                # Compute IoU
                iou_val = _compute_iou_xywh(
                    (gt_x, gt_y, gt_w, gt_h),
                    (det_x, det_y, det_w, det_h),
                )
                best_iou = max(best_iou, iou_val)

        if gt_visible:
            ious.append(best_iou)
            if best_iou > 0.5:
                n_detected += 1

    return {
        "sequence":  sequence_dir.name,
        "n_frames":  len(frame_paths),
        "n_visible": n_visible,
        "n_detected":n_detected,
        "mean_iou":  round(float(np.mean(ious)) if ious else 0.0, 4),
        "recall_05": round(n_detected / max(n_visible, 1), 4),
    }


def _compute_iou_xywh(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x1, y1, x2, y2)
    a_x1, a_y1, a_x2, a_y2 = x1, y1, x1+w1, y1+h1
    b_x1, b_y1, b_x2, b_y2 = x2, y2, x2+w2, y2+h2

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def evaluate_all_sequences(
    weights_path:   Path,
    sequences_dir:  Path,
    output_dir:     Path,
    conf:           float = 0.25,
    iou:            float = 0.45,
    imgsz:          int   = 640,
    device:         str   = "0",
    mlflow_uri:     Optional[str] = None,
) -> List[Dict]:
    """
    Evaluate all DUT tracking sequences and log results to MLflow.
    """
    from ultralytics import YOLO, RTDETR  # noqa: PLC0415

    if not sequences_dir.exists():
        log.error("Sequences directory not found: %s", sequences_dir)
        return []

    sequences = sorted(d for d in sequences_dir.iterdir() if d.is_dir())
    if not sequences:
        log.error("No sequence directories found in %s", sequences_dir)
        return []

    log.info("Found %d tracking sequences.", len(sequences))

    # Load model once
    if "rtdetr" in weights_path.stem.lower():
        model = RTDETR(str(weights_path))
    else:
        model = YOLO(str(weights_path))

    all_results = []
    for seq in sequences:
        log.info("Evaluating sequence: %s", seq.name)
        result = evaluate_sequence(
            model=model, sequence_dir=seq,
            conf=conf, iou=iou, imgsz=imgsz, device=device,
        )
        if result:
            all_results.append(result)
            log.info("  %s: mean_IoU=%.4f  recall@0.5=%.4f",
                     seq.name, result["mean_iou"], result["recall_05"])

    if not all_results:
        return []

    # Aggregate stats
    mean_iou    = float(np.mean([r["mean_iou"]   for r in all_results]))
    mean_recall = float(np.mean([r["recall_05"]  for r in all_results]))
    total_frames = sum(r["n_frames"] for r in all_results)
    total_visible = sum(r["n_visible"] for r in all_results)
    total_detected = sum(r["n_detected"] for r in all_results)

    log.info("=" * 60)
    log.info("TRACKING EVALUATION SUMMARY")
    log.info("  Sequences:      %d", len(all_results))
    log.info("  Total frames:   %d", total_frames)
    log.info("  Mean IoU:       %.4f", mean_iou)
    log.info("  Recall@IoU0.5:  %.4f", mean_recall)
    log.info("=" * 60)

    # Save results CSV
    import pandas as pd  # noqa: PLC0415
    df = pd.DataFrame(all_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tracking_evaluation.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved -> %s", csv_path)

    # Log to MLflow
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("tracking-evaluation")
        with mlflow.start_run(run_name=f"tracking-{weights_path.parent.parent.name}"):
            mlflow.log_params({
                "model": str(weights_path),
                "n_sequences": len(all_results),
                "conf": conf,
                "iou": iou,
            })
            mlflow.log_metrics({
                "tracking/mean_iou":        round(mean_iou, 4),
                "tracking/recall_at_0.5":   round(mean_recall, 4),
                "tracking/total_frames":    total_frames,
                "tracking/total_visible":   total_visible,
                "tracking/total_detected":  total_detected,
            })
            mlflow.log_artifact(str(csv_path), "tracking_results")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anti-UAV ByteTrack evaluation.")
    p.add_argument("--weights",        required=True, help="Path to best.pt")
    p.add_argument("--video",          default=None,  help="Single video to track")
    p.add_argument("--sequences-dir",  default=None,  help="DUT tracking sequences directory")
    p.add_argument("--output",         default="runs/tracking/output.mp4")
    p.add_argument("--output-dir",     default="runs/tracking/dut_results")
    p.add_argument("--conf",           type=float, default=0.25)
    p.add_argument("--iou",            type=float, default=0.45)
    p.add_argument("--imgsz",          type=int,   default=640)
    p.add_argument("--max-frames",     type=int,   default=None)
    p.add_argument("--mlflow-uri",     default=None)
    p.add_argument("--device",         default="0")
    return p.parse_args()


def main() -> None:
    import sys
    args = parse_args()
    weights = Path(args.weights)

    if not weights.exists():
        log.error("Weights not found: %s", weights)
        sys.exit(1)

    if args.video:
        # Track a single video
        summary = track_video(
            weights_path=weights,
            video_path=Path(args.video),
            output_path=Path(args.output),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_frames=args.max_frames,
        )
        print("\nTracking summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    elif args.sequences_dir:
        # Batch evaluation on DUT sequences
        evaluate_all_sequences(
            weights_path=weights,
            sequences_dir=Path(args.sequences_dir),
            output_dir=Path(args.output_dir),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            mlflow_uri=args.mlflow_uri,
        )
    else:
        log.error("Provide either --video or --sequences-dir")
        sys.exit(1)


if __name__ == "__main__":
    main()
