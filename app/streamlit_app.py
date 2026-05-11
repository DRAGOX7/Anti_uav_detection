"""
app/streamlit_app.py
=====================
Main Streamlit entry point for the AerialGuard: Anti-UAV System.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import base64
import streamlit as st
def get_image_base64(path):
    """Helper function to convert local images to base64 for HTML injection."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# --- NOW you can call it ---
rocket_base64 = get_image_base64("assets/rocket.png")

# 1. Page configuration (Must be the FIRST Streamlit command)
st.set_page_config(
    page_title="AerialGuard | Drone Detection",
    page_icon="assets/drone.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Add your custom sidebar logo (Immediately after page config)
st.logo("assets/drone.png")

# 2. Project root and Path setup
# Ensures modules in src.models, app.pages, etc., are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 3. MLflow URI Configuration
MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    (PROJECT_ROOT / "mlflow" / "mlruns").as_uri(), 
)


# 4. Professional "Defense" Themed CSS
st.markdown(
    """
    <style>
    :root { --accent: #1b4332; --accent-light: #2d6a4f; }

    .main-header {
        background: linear-gradient(135deg, #121212 0%, #1E1E1E 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 8px solid #FFB300; /* Matches your amber accent */
        box-shadow: 0 4px 10px rgba(0,0,0,0.5); /* Adds a slight shadow to make it pop */
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; letter-spacing: -0.5px; }
    .main-header p  { margin-top: 0.5rem; opacity: 0.8; font-size: 1.1rem; }

    .metric-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-5px); border-color: #FFB300; }
    .metric-card .value { font-size: 1.8rem; font-weight: 800; color: #FFB300; }
    .metric-card .label { font-size: 0.85rem; font-weight: 500; color: #FFB300; margin-top: 5px; text-transform: uppercase; }

    .sidebar-logo { 
        font-size: 1.4rem; 
        font-weight: 700; 
        color: #FFB300; 
        text-align: center; 
        padding: 1rem 0;
        border-bottom: 1px solid #eee;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Home Page Implementation
# ---------------------------------------------------------------------------
def _render_home() -> None:
    """Renders the dashboard landing page with key project metrics."""
    # 1. Read the local image and convert it to a base64 string
    def get_image_base64(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    # Make sure this path matches exactly where your image is
    img_base64 = get_image_base64("assets/dronecolor.png")

    # 2. Inject the base64 string directly into the HTML using an f-string
    st.markdown(
        f"""
        <div class="main-header">
            <h1>
                <img src="data:image/png;base64,{img_base64}" width="40" style="vertical-align: middle; margin-right: 12px; margin-top: -5px;"> 
                AerialGuard: Anti-UAV Defense System
            </h1>
            <p>Intelligence, Surveillance, and Reconnaissance (ISR) for Hostile Drone Detection</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top-level metrics based on final test results
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("0.977", "mAP@50", "YOLOv11-S"),
        ("32.5", "FPS", "Real-time (RTX 4050)"),
        ("1280px", "Resolution", "HP3 Configuration"),
        ("ByteTrack", "Tracker", "ID Persistence"),
    ]
    
    for col, (val, lbl, sub) in zip([col1, col2, col3, col4], stats):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="value">{val}</div>'
                f'<div class="label">{lbl} <br><span style="font-weight:400; opacity:0.7">{sub}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("📋 System Capability Overview")
        st.markdown("""
        AerialGuard is a multi-stage Computer Vision system designed to provide 
        automated detection and tracking of Unmanned Aerial Vehicles (UAVs).
        
        **Core Methodologies**
        * **Detection Strategy**: High-resolution (1280px) inference using HP3 configurations 
            to solve the "small target" problem common in high-altitude drone imagery.
        * **Architecture Evolution**: Benchmarking cutting-edge one-stage 
            detectors (**YOLOv11-S**) against classic two-stage architectures (**Faster R-CNN**).
        * **Tracking Engine**: Integration of **ByteTrack** for consistent object identification 
            across video frames without the overhead of a Re-ID network.
        
        **Dataset Statistics**
        * **Primary Source**: DUT Anti-UAV (10,000+ annotated frames).
        * **Validation**: 70/15/15 split across diverse environmental backgrounds.
        """)
        
        with st.expander("🛠️ View System Directory Structure"):
            st.code("""
anti-uav-detection/
├── src/
│   ├── training/      # MLflow-integrated training logic
│   ├── models/        # YOLOv11, YOLOv8, Faster R-CNN
│   └── evaluation/    # Notebook 08 final benchmarks
├── app/
│   ├── streamlit_app.py
│   └── pages/         # Modular dashboard views
└── mlflow/            # Metric & Artifact tracking
            """, language="text")

    with c2:
        st.markdown(
    f"""
    <div style="display: flex; align-items: center; margin-top: 20px; margin-bottom: 10px;">
        <img src="data:image/png;base64,{rocket_base64}" width="28" style="margin-right: 12px;">
        <h3 style="margin: 0; color: white; font-weight: 600;">Operational Quickstart</h3>
    </div>
    """,
    unsafe_allow_html=True
)
        with st.container(border=True):
            st.write("**1. Visual Inspection**")
            st.caption("Upload static imagery to evaluate detection confidence.")
            
            st.write("**2. Live-Action Tracking**")
            st.caption("Process MP4 footage to visualize tracking IDs and persistent boxes.")
            
            st.write("**3. Analytical Comparison**")
            st.caption("Review side-by-side metrics from MLflow to justify model selection.")
            
        st.success("System Status: **READY**")
        st.markdown("**Academic Verification**")
        st.caption("Jordan University of Science & Technology")
        st.caption("Course: AI447 Computer Vision")

    st.markdown("---")
    st.caption("© 2026 AerialGuard Anti-UAV Project. Developed for academic evaluation.")

# ---------------------------------------------------------------------------
# Sidebar Navigation and Page Routing
# ---------------------------------------------------------------------------
with st.sidebar:
    # 1. Add your new amber drone image
    # Note: We use columns to perfectly center the image in the sidebar
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/dronecolor.png", use_container_width=True)
    
    # 2. Add the title text right beneath it (without the emoji)
    st.markdown(
        '<div class="sidebar-logo" style="margin-top: -15px;">AERIALGUARD</div>', 
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        options=[
            "🏠  Project Overview",
            "🖼️  Image Detection",
            "🎬  Video + Tracking",
            "📊  Architecture Benchmarks",
            "🔬  MLflow Experiment Logs",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.info(f"**Tracking Active**\n\nURI: `{Path(MLFLOW_URI).name}`")
    st.caption("AI447 Computer Vision — Spring 2025-2026")

# Execute routing based on sidebar selection
try:
    if page == "🏠  Project Overview":
        _render_home()
    elif page == "🖼️  Image Detection":
        from app.pages import image_detection
        image_detection.render(PROJECT_ROOT, MLFLOW_URI)
    elif page == "🎬  Video + Tracking":
        from app.pages import video_tracking
        video_tracking.render(PROJECT_ROOT, MLFLOW_URI)
    elif page == "📊  Architecture Benchmarks":
        from app.pages import model_comparison
        model_comparison.render(MLFLOW_URI)
    elif page == "🔬  MLflow Experiment Logs":
        from app.pages import mlflow_dashboard
        mlflow_dashboard.render(MLFLOW_URI)
except ImportError as e:
    st.error(f"Module Loading Error: {e}")
    st.warning("Ensure all page files exist in the `app/pages/` directory.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")