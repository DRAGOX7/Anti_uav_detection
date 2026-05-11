"""
app/pages/image_detection.py
=============================
Streamlit page: Upload an image, pick a model, run inference, visualise.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import mlflow
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Model loader (cached so weights aren't reloaded on every interaction)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model weights…")
def _load_model(weights_path: str, arch: str):
    """Load and cache a model from disk or MLflow registry."""
    if arch.startswith("rtdetr"):
        from ultralytics import RTDETR  # noqa: PLC0415

        return RTDETR(weights_path)
    else:
        from ultralytics import YOLO  # noqa: PLC0415

        return YOLO(weights_path)


def _find_available_weights(weights_dir: Path) -> dict[str, Path]:
    """
    Scan app/model_weights/ for .pt files.
    Returns { display_name: path } mapping.
    """
    weights = {}
    if not weights_dir.exists():
        return weights
    for pt in sorted(weights_dir.rglob("*.pt")):
        rel_parent = pt.parent.relative_to(weights_dir)
        if rel_parent == Path("."):
            name = pt.stem
        else:
            name = f"{rel_parent}/{pt.stem}"
        weights[name] = pt
    return weights


def _run_inference(model, img_array: np.ndarray, conf: float, iou: float):
    """Run YOLO/RTDETR inference and return results object."""
    return model.predict(
        source=img_array,
        conf=conf,
        iou=iou,
        verbose=False,
        save=False,
    )


def _draw_detections(img_bgr: np.ndarray, results, colour=(0, 200, 80)) -> np.ndarray:
    """Draw bounding boxes and labels on image. Returns annotated copy."""
    out = img_bgr.copy()
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            int(box.cls[0])
            label = f"drone {conf:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(
                out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )
    return out


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------


def render(project_root: Path, mlflow_uri: str) -> None:
    st.header("🖼️ Image Detection")
    st.caption("Upload a drone image and run detection with your trained model.")

    weights_dir = project_root / "app" / "model_weights"

    # ── Sidebar controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Detection settings")

        available = _find_available_weights(weights_dir)
        if not available:
            st.warning(
                "No trained models found in `app/model_weights/`.\n\n"
                "Download or deploy the production model to this folder."
            )
            model_choice = None
            weights_path = None
        else:
            model_choice = st.selectbox("Model / HP combo", list(available.keys()))
            weights_path = str(available[model_choice])
            st.caption(f"Weights: `{weights_path}`")

        conf_thresh = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
        iou_thresh = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
        show_raw = st.checkbox("Show original image alongside", value=True)

    # ── File uploader ────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a drone image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Any RGB image. Tip: try with images containing tiny drones!",
    )

    if uploaded is None:
        st.info("👆 Upload an image to begin detection.")
        _show_demo_notice()
        return

    if model_choice is None:
        st.error("No trained models available in `app/model_weights/`.")
        return

    # ── Load & display image ─────────────────────────────────────────────
    pil_img = Image.open(uploaded).convert("RGB")
    img_np = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ── Run detection ────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        model = _load_model(weights_path, model_choice)
        results = _run_inference(model, img_bgr, conf_thresh, iou_thresh)

    # ── Extract detection info ────────────────────────────────────────────
    all_boxes = []
    for r in results:
        if r.boxes:
            for box in r.boxes:
                all_boxes.append(
                    {
                        "x1": int(box.xyxy[0][0]),
                        "y1": int(box.xyxy[0][1]),
                        "x2": int(box.xyxy[0][2]),
                        "y2": int(box.xyxy[0][3]),
                        "conf": float(box.conf[0]),
                    }
                )

    # ── Metrics row ──────────────────────────────────────────────────────
    n_det = len(all_boxes)
    max_conf = max((b["conf"] for b in all_boxes), default=0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drones detected", n_det)
    c2.metric("Max confidence", f"{max_conf:.3f}" if max_conf else "—")
    c3.metric("Image size", f"{pil_img.width}×{pil_img.height}")
    c4.metric("Model", model_choice.replace("_", " "))

    st.markdown("---")

    # ── Side-by-side images ───────────────────────────────────────────────
    annotated_bgr = _draw_detections(img_bgr, results)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if show_raw:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(pil_img, use_column_width=True)
        with col2:
            st.subheader(f"Detected ({n_det} drone{'s' if n_det != 1 else ''})")
            st.image(annotated_rgb, use_column_width=True)
    else:
        st.subheader(f"Detection result — {n_det} drone{'s' if n_det != 1 else ''} found")
        st.image(annotated_rgb, use_column_width=True)

    # ── Bounding box table ────────────────────────────────────────────────
    if all_boxes:
        st.markdown("---")
        st.subheader("Detection details")
        import pandas as pd  # noqa: PLC0415

        df = pd.DataFrame(all_boxes)
        df["width"] = df["x2"] - df["x1"]
        df["height"] = df["y2"] - df["y1"]
        df["area_%"] = (df["width"] * df["height"] / (pil_img.width * pil_img.height) * 100).round(
            3
        )
        df["conf"] = df["conf"].round(4)
        st.dataframe(
            df[["x1", "y1", "x2", "y2", "width", "height", "area_%", "conf"]],
            use_container_width=True,
        )

    # ── Download annotated image ──────────────────────────────────────────
    st.markdown("---")
    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="PNG")
    st.download_button(
        "⬇️ Download annotated image",
        data=buf.getvalue(),
        file_name=f"detected_{uploaded.name}",
        mime="image/png",
    )

    # ── Log inference to MLflow ───────────────────────────────────────────
    _log_inference_to_mlflow(mlflow_uri, model_choice, n_det, max_conf, pil_img)


def _show_demo_notice() -> None:
    with st.expander("💡 Sample images to try"):
        st.markdown("""
        Once the dataset is downloaded, you can find sample test images at:
        ```
        data/raw/dut_anti_uav/DUT-Anti-UAV-Detection/test/images/
        ```
        These include a wide variety of drone types, backgrounds,
        lighting conditions, and drone sizes — including very tiny drones
        that occupy less than 0.5% of the image.
        """)


def _log_inference_to_mlflow(
    uri: str,
    model_name: str,
    n_detections: int,
    max_conf: float,
    img: Image.Image,
) -> None:
    """Log inference event to MLflow (optional, non-blocking)."""
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("inference-logs")
        with mlflow.start_run(run_name=f"inference-{model_name}"):
            mlflow.log_params(
                {
                    "model": model_name,
                    "img_width": img.width,
                    "img_height": img.height,
                }
            )
            mlflow.log_metrics(
                {
                    "n_detections": n_detections,
                    "max_confidence": max_conf,
                }
            )
    except Exception:
        pass  # Logging failure should never block the UI
