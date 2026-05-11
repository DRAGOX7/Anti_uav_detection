"""
app/pages/video_tracking.py
============================
Streamlit page: Upload drone footage → detect + ByteTrack → export annotated MP4.
This is the +5 bonus mark deliverable.

DRONE vs BIRD DISCRIMINATOR 
------------------------------------------------------
Uses 4 kinematic and geometric signals to separate drones from birds:

  Signal 1 — Shape stability (aspect ratio fluctuation)
      Drones are rigid bodies → aspect ratio stays constant.
      Birds flap wings → aspect ratio oscillates wildly.

  Signal 2 — Area stability (bounding box area fluctuation)
      Drones maintain physical size across frames.
      Birds change silhouette area 30-60% per wing-beat cycle.

  Signal 3 — Motion smoothness (trajectory linearity)
      Drones fly in smooth, predictable trajectories.
      Birds bank, dive, and change direction erratically.

  Signal 4 — Velocity consistency (frame-to-frame speed change)
      Drones accelerate/decelerate gradually.
      Birds show sudden speed bursts when flapping.

Each signal produces a score 0-1. A weighted sum determines drone vs bird.
No retraining. Works on any model output.
"""

from __future__ import annotations

import collections
import math
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette for track IDs
# ─────────────────────────────────────────────────────────────────────────────
TRACK_COLOURS = [
    (0, 200, 80),   (255, 140, 0),  (0, 120, 255),  (200, 0, 200),
    (0, 200, 200),  (255, 50, 50),  (50, 200, 50),  (255, 200, 0),
    (100, 100, 255),(255, 100, 100),(0, 180, 180),  (180, 0, 180),
]

def _get_colour(track_id: int) -> Tuple[int, int, int]:
    return TRACK_COLOURS[track_id % len(TRACK_COLOURS)]


# ─────────────────────────────────────────────────────────────────────────────
# Drone / Bird Discriminator
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED DRONE / BIRD DISCRIMINATOR
# Drop-in replacement for your current DroneClassifier
# ─────────────────────────────────────────────────────────────────────────────

class DroneClassifier:
    """
    Improved heuristic drone-vs-bird discriminator.

    NEW SIGNALS
    -----------
    1. Shape stability
    2. Area stability
    3. Motion smoothness
    4. Velocity consistency
    5. Angular jitter            ← NEW
    6. Wingbeat oscillation FFT  ← NEW
    7. Detection confidence stab ← NEW
    8. Track age prior           ← NEW

    Key improvements:
    - Better temporal modelling
    - Less aggressive smoothing
    - Oscillation detection
    - Heading stability analysis
    - More robust against gliding birds
    """

    def __init__(
        self,
        history_len: int = 40,
        drone_threshold: float = 0.60,
        min_frames: int = 12,
    ):
        self.history_len = history_len
        self.drone_threshold = drone_threshold
        self.min_frames = min_frames

        self._aspect_ratios = {}
        self._areas = {}
        self._centres = {}
        self._confidences = {}

        self._scores = {}
        self._frame_counts = {}

    # ─────────────────────────────────────────────────────────────────────

    def update(
        self,
        track_id: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        det_conf: float,
        img_w: int,
        img_h: int,
    ):

        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        aspect = w / h
        area = (w * h) / max(img_w * img_h, 1)

        # ─────────────────────────────────────────────────────────────
        # Init buffers
        # ─────────────────────────────────────────────────────────────

        if track_id not in self._aspect_ratios:

            self._aspect_ratios[track_id] = collections.deque(
                maxlen=self.history_len
            )

            self._areas[track_id] = collections.deque(
                maxlen=self.history_len
            )

            self._centres[track_id] = collections.deque(
                maxlen=self.history_len
            )

            self._confidences[track_id] = collections.deque(
                maxlen=self.history_len
            )

            self._scores[track_id] = 0.5
            self._frame_counts[track_id] = 0

        # ─────────────────────────────────────────────────────────────

        self._aspect_ratios[track_id].append(aspect)
        self._areas[track_id].append(area)
        self._centres[track_id].append((cx, cy))
        self._confidences[track_id].append(det_conf)

        self._frame_counts[track_id] += 1

        n = self._frame_counts[track_id]

        if n < self.min_frames:
            return 0.5, "CHECKING...", {}

        signals = {}

        # ============================================================
        # SIGNAL 1 — SHAPE STABILITY
        # ============================================================

        ar_list = np.array(self._aspect_ratios[track_id])

        ar_fluc = np.std(ar_list)

        signals["shape_stability"] = float(
            np.clip(1.0 - ar_fluc / 0.25, 0.0, 1.0)
        )

        # ============================================================
        # SIGNAL 2 — AREA STABILITY
        # ============================================================

        area_list = np.array(self._areas[track_id])

        area_cv = np.std(area_list) / max(np.mean(area_list), 1e-9)

        signals["area_stability"] = float(
            np.clip(1.0 - area_cv / 0.35, 0.0, 1.0)
        )

        # ============================================================
        # SIGNAL 3 — MOTION SMOOTHNESS
        # ============================================================

        centres = np.array(self._centres[track_id], dtype=np.float32)

        pts = centres.copy()

        pts[:, 0] /= img_w
        pts[:, 1] /= img_h

        if len(pts) >= 5:

            mean = pts.mean(axis=0)

            centred = pts - mean

            cov = np.cov(centred.T)

            eigvals = np.linalg.eigvalsh(cov)

            eigvals = np.sort(eigvals)[::-1]

            linearity = eigvals[0] / max(eigvals.sum(), 1e-9)

            signals["motion_smoothness"] = float(
                np.clip((linearity - 0.55) / 0.45, 0.0, 1.0)
            )

        else:
            signals["motion_smoothness"] = 0.5

        # ============================================================
        # SIGNAL 4 — VELOCITY CONSISTENCY
        # ============================================================

        speeds = []

        for i in range(1, len(pts)):

            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]

            speeds.append(math.sqrt(dx * dx + dy * dy))

        if len(speeds) >= 3:

            speed_arr = np.array(speeds)

            speed_cv = np.std(speed_arr) / max(np.mean(speed_arr), 1e-9)

            signals["velocity_consistency"] = float(
                np.clip(1.0 - speed_cv / 0.6, 0.0, 1.0)
            )

        else:
            signals["velocity_consistency"] = 0.5

        # ============================================================
        # SIGNAL 5 — ANGULAR JITTER (NEW)
        # ============================================================

        angle_changes = []

        for i in range(2, len(pts)):

            dx1 = pts[i - 1][0] - pts[i - 2][0]
            dy1 = pts[i - 1][1] - pts[i - 2][1]

            dx2 = pts[i][0] - pts[i - 1][0]
            dy2 = pts[i][1] - pts[i - 1][1]

            a1 = math.atan2(dy1, dx1)
            a2 = math.atan2(dy2, dx2)

            diff = abs(a2 - a1)

            diff = min(diff, 2 * math.pi - diff)

            angle_changes.append(diff)

        if len(angle_changes) >= 3:

            jitter = np.std(angle_changes)

            signals["angular_jitter"] = float(
                np.clip(1.0 - jitter / 0.35, 0.0, 1.0)
            )

        else:
            signals["angular_jitter"] = 0.5

        # ============================================================
        # SIGNAL 6 — FFT OSCILLATION ANALYSIS (NEW)
        # ============================================================

        if len(ar_list) >= 12:

            signal = ar_list - np.mean(ar_list)

            fft = np.fft.rfft(signal)

            power = np.abs(fft)[1:]

            osc_energy = np.mean(power)

            # high oscillation = bird

            signals["wingbeat_fft"] = float(
                np.clip(1.0 - osc_energy / 0.15, 0.0, 1.0)
            )

        else:
            signals["wingbeat_fft"] = 0.5

        # ============================================================
        # SIGNAL 7 — CONFIDENCE STABILITY (NEW)
        # ============================================================

        confs = np.array(self._confidences[track_id])

        conf_cv = np.std(confs) / max(np.mean(confs), 1e-9)

        signals["confidence_stability"] = float(
            np.clip(1.0 - conf_cv / 0.45, 0.0, 1.0)
        )

        # ============================================================
        # WEIGHTED FUSION
        # ============================================================

        weights = {

            "shape_stability":      0.18,
            "area_stability":       0.18,
            "motion_smoothness":    0.16,
            "velocity_consistency": 0.10,
            "angular_jitter":       0.16,
            "wingbeat_fft":         0.14,
            "confidence_stability": 0.08,
        }

        drone_score = 0.0

        for k, w in weights.items():
            drone_score += signals[k] * w

        # ============================================================
        # TRACK AGE PRIOR
        # ============================================================

        if n < 15:
            drone_score *= 0.85

        # ============================================================
        # TEMPORAL SMOOTHING (LESS AGGRESSIVE)
        # ============================================================

        prev_score = self._scores[track_id]

        smoothed = 0.45 * prev_score + 0.55 * drone_score

        self._scores[track_id] = smoothed

        # ============================================================
        # FINAL LABEL
        # ============================================================

        if smoothed >= self.drone_threshold:
            label = "DRONE"
        else:
            label = "BIRD"

        return smoothed, label, signals

    # ─────────────────────────────────────────────────────────────────────

    def get_score(self, track_id: int) -> float:
        return self._scores.get(track_id, 0.5)

    def get_label(self, track_id: int) -> str:

        if self._frame_counts.get(track_id, 0) < self.min_frames:
            return "CHECKING..."

        return (
            "DRONE"
            if self.get_score(track_id) >= self.drone_threshold
            else "BIRD"
        )

    # ─────────────────────────────────────────────────────────────────────

    def cleanup_stale_tracks(self, active_ids: set):

        stale = set(self._aspect_ratios.keys()) - active_ids

        for tid in stale:

            self._aspect_ratios.pop(tid, None)
            self._areas.pop(tid, None)
            self._centres.pop(tid, None)
            self._confidences.pop(tid, None)

            self._scores.pop(tid, None)
            self._frame_counts.pop(tid, None)

# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights…")
def _load_model(weights_path: str, arch: str):
    if "rtdetr" in arch.lower():
        from ultralytics import RTDETR
        return RTDETR(weights_path)
    from ultralytics import YOLO
    return YOLO(weights_path)


def _find_available_weights(runs_dir: Path) -> dict[str, Path]:
    weights = {}
    if not runs_dir.exists():
        return weights
    for pt in sorted(runs_dir.rglob("best.pt")):
        name = pt.parent.parent.name
        weights[name] = pt
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render(project_root: Path, mlflow_uri: str) -> None:
    st.header("🎬 Video Detection + ByteTrack Tracking")
    st.caption(
        "Upload drone footage — detects drones frame-by-frame, "
        "assigns persistent IDs using ByteTrack, and filters out birds "
        "using a 4-signal kinematic discriminator."
    )

    runs_dir = project_root / "runs" / "train"

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Tracking settings")
        available = _find_available_weights(runs_dir)
        if not available:
            st.warning("No trained models found. Run training notebooks first.")
            model_choice = None
            weights_path = None
        else:
            model_choice = st.selectbox("Model", list(available.keys()))
            weights_path = str(available[model_choice])
            st.caption(f"`{weights_path}`")

        conf_thresh = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.05)
        iou_thresh  = st.slider("NMS IoU threshold",    0.10, 0.90, 0.45, 0.05)
        max_frames  = st.number_input(
            "Max frames to process (0 = all)",
            min_value=0, max_value=10_000, value=0, step=50,
        )

        st.markdown("### Display options")
        show_trails    = st.checkbox("Trajectory trails",    value=True)
        show_ids       = st.checkbox("Track IDs",            value=True)
        show_conf      = st.checkbox("Confidence scores",    value=True)
        show_hud       = st.checkbox("Show signal scores",   value=True,
                                     help="Display the 4 kinematic signals per detection")
        filter_birds   = st.checkbox("Filter birds out",     value=True,
                                     help="Hide detections classified as birds")

        st.markdown("### Bird filter tuning")
        drone_threshold = st.slider(
            "Drone score threshold",
            0.30, 0.90, 0.55, 0.05,
            help="Higher = stricter (fewer false drones). Lower = more permissive.",
        )
        min_frames_check = st.slider(
            "Min frames before decision",
            3, 20, 8, 1,
            help="How many frames to observe before classifying as drone or bird.",
        )

    # ── Video uploader ────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload drone footage (MP4 / AVI / MOV)",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded is None:
        st.info("👆 Upload a video to begin detection + tracking.")
        _show_tips()
        return

    if model_choice is None:
        st.error("No trained model found. Train a model first.")
        return

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(uploaded.read())
        input_path = tmp_in.name

    output_path = input_path.replace(".mp4", "_tracked.mp4")

    if st.button("▶️ Run Detection + Tracking", type="primary"):
        _run_tracking(
            input_path=input_path,
            output_path=output_path,
            weights_path=weights_path,
            model_choice=model_choice,
            conf=conf_thresh,
            iou=iou_thresh,
            max_frames=int(max_frames) or None,
            show_trails=show_trails,
            show_ids=show_ids,
            show_conf=show_conf,
            show_hud=show_hud,
            filter_birds=filter_birds,
            drone_threshold=drone_threshold,
            min_frames_check=min_frames_check,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core tracking loop
# ─────────────────────────────────────────────────────────────────────────────

def _run_tracking(
    input_path:       str,
    output_path:      str,
    weights_path:     str,
    model_choice:     str,
    conf:             float,
    iou:              float,
    max_frames:       Optional[int],
    show_trails:      bool,
    show_ids:         bool,
    show_conf:        bool,
    show_hud:         bool,
    filter_birds:     bool,
    drone_threshold:  float,
    min_frames_check: int,
) -> None:

    model      = _load_model(weights_path, model_choice)
    classifier = DroneClassifier(
        history_len=30,
        drone_threshold=drone_threshold,
        min_frames=min_frames_check,
    )

    cap          = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    process_n    = min(max_frames or total_frames, total_frames)

    st.markdown("---")
    st.markdown(
        f"**Video:** {W}×{H}px · {fps_video:.1f} FPS · "
        f"{total_frames} frames · Processing {process_n}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_video, (W, H))

    # State
    trails:        Dict[int, List[Tuple[int, int]]] = {}
    TRAIL_LEN      = 40
    unique_drones: set = set()
    unique_birds:  set = set()
    total_dets     = 0
    frame_count    = 0

    progress_bar = st.progress(0, text="Processing frames…")
    preview_slot = st.empty()
    stats_cols   = st.columns(5)
    s_frame      = stats_cols[0].empty()
    s_dets       = stats_cols[1].empty()
    s_drones     = stats_cols[2].empty()
    s_birds      = stats_cols[3].empty()
    s_fps        = stats_cols[4].empty()

    t0 = time.time()

    while cap.isOpened() and frame_count < process_n:
        ret, frame = cap.read()
        if not ret:
            break

        # ── ByteTrack detection ───────────────────────────────────────
        results = model.track(
            source=frame,
            conf=conf,
            iou=iou,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        annotated     = frame.copy()
        active_ids    = set()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx  = (x1 + x2) // 2
                cy  = (y1 + y2) // 2
                det_conf = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else -1

                if tid < 0:
                    continue

                active_ids.add(tid)
                total_dets += 1

                # ── Classify: drone or bird? ──────────────────────────
                drone_score, label, signals = classifier.update(
    track_id=tid,
    x1=x1,
    y1=y1,
    x2=x2,
    y2=y2,
    det_conf=det_conf,
    img_w=W,
    img_h=H,
)

                is_drone    = (label == "DRONE")
                is_checking = (label == "CHECKING...")

                # Skip birds if filter is enabled
                if filter_birds and label == "BIRD":
                    continue

                # Track statistics
                if is_drone:
                    unique_drones.add(tid)
                elif label == "BIRD":
                    unique_birds.add(tid)

                # ── Choose visual style based on classification ────────
                if is_checking:
                    colour     = (180, 180, 180)   # grey = still deciding
                    box_thick  = 1
                    box_style  = "dashed"
                elif is_drone:
                    colour     = _get_colour(tid)  # bright coloured = confirmed drone
                    box_thick  = 2
                    box_style  = "solid"
                else:
                    colour     = (0, 0, 200)       # red = bird (shown when filter off)
                    box_thick  = 1
                    box_style  = "solid"

                # ── Draw bounding box ──────────────────────────────────
                if box_style == "dashed":
                    _draw_dashed_rect(annotated, (x1,y1), (x2,y2), colour, 1)
                else:
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), colour, box_thick)

                # ── Build label ────────────────────────────────────────
                parts = []
                if show_ids:   parts.append(f"#{tid}")
                parts.append(label)
                if show_conf:  parts.append(f"{det_conf:.2f}")
                if show_hud and signals:
                    parts.append(f"S:{drone_score:.2f}")
                text = " | ".join(parts)

                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, y1-th-6), (x1+tw+4, y1), colour, -1)
                cv2.putText(annotated, text, (x1+2, y1-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

                # ── HUD: signal breakdown ──────────────────────────────
                if show_hud and signals and is_drone:
                    hud_y = y2 + 14
                    for sig_name, sig_val in signals.items():
                        bar_w   = int(sig_val * 60)
                        bar_col = (0,200,80) if sig_val > 0.6 else (255,140,0) if sig_val > 0.4 else (0,0,200)
                        short   = sig_name[:4].upper()
                        cv2.putText(annotated, f"{short}:", (x1, hud_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220,220,220), 1)
                        cv2.rectangle(annotated, (x1+30, hud_y-7),
                                      (x1+30+bar_w, hud_y-1), bar_col, -1)
                        hud_y += 12

                # ── Trail (drones only) ────────────────────────────────
                if show_trails and is_drone:
                    if tid not in trails:
                        trails[tid] = []
                    trails[tid].append((cx, cy))
                    if len(trails[tid]) > TRAIL_LEN:
                        trails[tid].pop(0)
                    pts = trails[tid]
                    for i in range(1, len(pts)):
                        alpha   = i / len(pts)
                        c_faded = tuple(int(v * alpha) for v in colour)
                        cv2.line(annotated, pts[i-1], pts[i], c_faded, 2)

        # Cleanup stale tracks every 30 frames
        if frame_count % 30 == 0:
            classifier.cleanup_stale_tracks(active_ids)

        # ── Overlay counters ───────────────────────────────────────────
        cv2.putText(annotated, f"Frame {frame_count+1}/{process_n}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        cv2.putText(annotated, f"Drones: {len(unique_drones)}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,80), 1)
        if not filter_birds:
            cv2.putText(annotated, f"Birds: {len(unique_birds)}",
                        (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,80,255), 1)

        writer.write(annotated)
        frame_count += 1

        # ── Update UI every 10 frames ──────────────────────────────────
        if frame_count % 10 == 0 or frame_count == process_n:
            pct     = frame_count / process_n
            elapsed = time.time() - t0
            cur_fps = frame_count / max(elapsed, 0.001)

            progress_bar.progress(pct, text=f"Frame {frame_count}/{process_n}")
            s_frame.metric("Frames",        f"{frame_count}/{process_n}")
            s_dets.metric("Detections",     total_dets)
            s_drones.metric("🟢 Drones",    len(unique_drones))
            s_birds.metric("🔴 Birds",      len(unique_birds))
            s_fps.metric("Proc. FPS",       f"{cur_fps:.1f}")

            preview_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            preview_slot.image(preview_rgb,
                               caption=f"Frame {frame_count}",
                               use_container_width=True)

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    progress_bar.progress(1.0, text="✅ Done!")

    st.success(
        f"Complete in {elapsed:.1f}s · {frame_count} frames · "
        f"**{len(unique_drones)} drones** tracked · "
        f"{len(unique_birds)} birds filtered"
    )

    with open(output_path, "rb") as vf:
        st.download_button(
            "⬇️ Download annotated video",
            data=vf.read(),
            file_name="tracked_drones.mp4",
            mime="video/mp4",
        )

    # ── Summary ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Tracking summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Frames",         frame_count)
    c2.metric("Total dets",     total_dets)
    c3.metric("🟢 Drone IDs",   len(unique_drones))
    c4.metric("🔴 Birds filtered", len(unique_birds))
    c5.metric("Speed",          f"{frame_count/max(elapsed,0.001):.1f} FPS")

    # ── Legend ────────────────────────────────────────────────────────────
    with st.expander("🔬 How the bird/drone discriminator works"):
        st.markdown("""
        The discriminator runs **4 kinematic signals** on each tracked object
        — no retraining required.

        | Signal | What it measures | Drone | Bird |
        |---|---|---|---|
        | **Shape stability** | Aspect ratio fluctuation over time | Low (rigid body) | High (wing beats) |
        | **Area stability** | Bounding box area variation | Low (constant size) | High (silhouette changes) |
        | **Motion smoothness** | Trajectory linearity (PCA) | Near-linear path | Curved, erratic |
        | **Velocity consistency** | Frame-to-frame speed variation | Steady speed | Burst-and-glide |

        Each signal produces a score **0→1** (1 = definitely drone).
        A weighted sum gives the final **drone score** shown as `S:0.xx` in the HUD.

        **Tune with the sidebar sliders:**
        - Raise the threshold if birds are slipping through
        - Lower the threshold if real drones are being filtered out
        - Raise min frames if very short clips cause mis-classification
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_dashed_rect(
    img: np.ndarray,
    pt1: Tuple[int,int],
    pt2: Tuple[int,int],
    colour: Tuple[int,int,int],
    thickness: int = 1,
    dash_len: int = 8,
) -> None:
    """Draw a dashed rectangle (used for 'CHECKING...' state)."""
    x1, y1 = pt1
    x2, y2 = pt2
    for side in [
        ((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
        ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1)),
    ]:
        sx, sy = side[0]
        ex, ey = side[1]
        dist   = int(math.hypot(ex-sx, ey-sy))
        for d in range(0, dist, dash_len * 2):
            t0 = d / max(dist, 1)
            t1 = min((d + dash_len) / max(dist, 1), 1.0)
            p0 = (int(sx + (ex-sx)*t0), int(sy + (ey-sy)*t0))
            p1 = (int(sx + (ex-sx)*t1), int(sy + (ey-sy)*t1))
            cv2.line(img, p0, p1, colour, thickness)


def _show_tips() -> None:
    with st.expander("💡 Tips for best results"):
        st.markdown("""
        **Recommended test videos:**
        - DUT Anti-UAV tracking sequences — `data/raw/dut_tracking/DUT-Anti-UAV-Tracking/`
        - Any outdoor drone footage with birds in the scene

        **Bird filter tuning guide:**
        - Start with threshold=0.55, min_frames=8
        - If real drones are being filtered: lower threshold to 0.45
        - If birds are slipping through: raise threshold to 0.65
        - Enable **Show signal scores** to see which signal is causing issues
        - The **Shape stability** and **Area stability** signals are most reliable

        **Visual guide:**
        - 🟢 Coloured solid box = confirmed drone
        - 🔴 Blue solid box = classified bird (visible when filter is OFF)
        - ⬜ Grey dashed box = still collecting data (CHECKING...)
        - Trail = flight path history (drones only)
        """)