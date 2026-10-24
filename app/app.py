import os
import sys

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import json
import torch

from src.config import cfg
from src.infer import TamperInferenceEngine
from src.data_generator import SyntheticTamperingGenerator

# ---------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------
st.set_page_config(
    page_title="DeepVision Forensics Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for sticky header, cards, and theme toggle
st.markdown(
    """
    <style>
    /* Sticky top header */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        background-color: var(--background-primary);
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .branding {font-size: 1.5rem; font-weight: 700; color: #0F172A;}
    .status-badge {padding: 0.2rem 0.6rem; border-radius: 0.5rem; font-weight: 600;}
    .badge-auth {background-color: #F0FDF4; color: #166534; border: 1px solid #22C55E;}
    .badge-tampered {background-color: #FEF2F2; color: #991B1B; border: 1px solid #EF4444;}
    .metric-chip {background: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 0.5rem; padding: 0.2rem 0.5rem; margin-left: 0.5rem;}
    .card {background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 0.75rem; padding: 1rem; margin-bottom: 1rem;}
    .evidence-box {background-color: #F8FAFC; border-left: 4px solid #3B82F6; border-radius: 6px; padding: 12px; margin-top: 8px; margin-bottom: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# Sidebar – Model Settings & Theme Toggle
# ---------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/fingerprint-scan.png", width=64)
st.sidebar.title("DeepVision Forensics")
st.sidebar.markdown("**Dual‑Branch Forensic Deep Learning Framework**")

checkpoint_path = st.sidebar.text_input("Model Checkpoint Path", value=cfg.best_model_path)
ela_quality = st.sidebar.slider("ELA JPEG Resave Quality", min_value=70, max_value=98, value=cfg.ela_quality, step=1)
ela_scale = st.sidebar.slider("ELA Contrast Amplification", min_value=5.0, max_value=30.0, value=cfg.ela_scale, step=1.0)
conf_threshold = st.sidebar.slider("Tampering Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
mask_threshold = st.sidebar.slider("Heatmap Binarization Cutoff", min_value=0.1, max_value=0.8, value=0.35, step=0.05)

# Theme toggle (light/dark) – simple CSS switch
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
if theme == "Dark":
    st.markdown("""<style>html {filter: invert(1) hue-rotate(180deg);} img {filter: invert(1) hue-rotate(180deg);} </style>""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Inference Engine (cached)
# ---------------------------------------------------
@st.cache_resource
def load_engine(path: str):
    return TamperInferenceEngine(checkpoint_path=path)

engine = load_engine(checkpoint_path)

# ---------------------------------------------------
# Sticky Header (branding only – status updates appear later)
# ---------------------------------------------------
st.markdown(
    """
    <div class='sticky-header'>
        <div class='branding'>DeepVision Forensics</div>
        <div id='status-indicator' class='status-badge badge-auth'>Model Ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# Main Workflow – Upload & Run Analysis
# ---------------------------------------------------
col_upload, col_preview = st.columns([1, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload image for forensic examination",
        type=["jpg", "jpeg", "png", "bmp", "tif", "webp"]
    )
    # Load sample picker if samples exist
    img_folder = os.path.join(cfg.generated_data_dir, "images")
    sample_files = []
    if os.path.isdir(img_folder):
        sample_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    selected_sample = None
    if sample_files:
        selected_sample = st.selectbox("Select sample image", ["--None--"] + sample_files, index=0)
    sample_btn = st.button("Load Selected Sample", use_container_width=True)
    # Store uploaded image bytes
    if uploaded_file is not None:
        st.session_state['selected_image_bytes'] = uploaded_file.read()
    # Store chosen sample path
    if sample_btn and selected_sample and selected_sample != "--None--":
        st.session_state['sample_path'] = os.path.join(img_folder, selected_sample)

with col_preview:
    if uploaded_file is not None:
        preview_img = Image.open(uploaded_file).convert("RGB")
        st.image(preview_img, caption="Uploaded Image Preview", use_container_width=True)
    else:
        st.info("No image selected yet.")

# Decide which image to analyze
image_to_analyze = None
# Priority: uploaded image > selected sample > generated sample
if uploaded_file is not None:
    image_to_analyze = Image.open(uploaded_file).convert("RGB")
elif st.session_state.get('sample_path'):
    try:
        image_to_analyze = Image.open(st.session_state['sample_path']).convert("RGB")
        st.sidebar.success(f"Loaded sample: {os.path.basename(st.session_state['sample_path'])}")
    except Exception as e:
        st.sidebar.error(f"Failed to load selected sample: {e}")
else:
    # Fallback to random sample or synthetic generation
    img_folder = os.path.join(cfg.generated_data_dir, "images")
    if os.path.isdir(img_folder):
        files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            import random
            choice = random.choice(files)
            image_to_analyze = Image.open(choice).convert("RGB")
            st.sidebar.success(f"Loaded random sample: {os.path.basename(choice)}")
    if image_to_analyze is None:
        gen = SyntheticTamperingGenerator()
        proc = gen.generate_procedural_scene()
        image_to_analyze, _ = gen.create_copy_move(proc)
        st.sidebar.info("Generated a synthetic sample image.")

if image_to_analyze is not None:
    st.subheader("Selected Image")
    st.image(image_to_analyze, use_container_width=True)
    run_btn = st.button("Run Analysis", type="primary")
else:
    run_btn = False

# ---------------------------------------------------
# Perform Analysis when user clicks Run
# ---------------------------------------------------
if run_btn:
    # Update status badge
    st.markdown("""<script>document.getElementById('status-indicator').innerText='Processing';document.getElementById('status-indicator').className='status-badge badge-tampered';</script>""", unsafe_allow_html=True)
    with st.spinner("Running forensic analysis…"):
        res = engine.predict(
            image_to_analyze,
            ela_quality=ela_quality,
            ela_scale=ela_scale,
            mask_threshold=mask_threshold,
        )
    # Determine verdict
    is_tampered = res["tamper_probability"] >= conf_threshold
    prob_pct = res["tamper_probability"] * 100.0

    # ---------------------------------------------------
    # Verdict Banner (post‑analysis)
    # ---------------------------------------------------
    badge_class = "badge-tampered" if is_tampered else "badge-auth"
    verdict_text = "TAMPERED" if is_tampered else "AUTHENTIC"
    verdict_label = "Manipulated Image Detected" if is_tampered else "Pristine Image"
    st.markdown(
        f"""
        <div class='sticky-header'>
            <div class='status-badge {badge_class}'>{verdict_text}</div>
            <div style='font-weight:600;'>{verdict_label}</div>
            <div>
                <span class='metric-chip'>Probability: {prob_pct:.1f}%</span>
                <span class='metric-chip'>Confidence: {res['type_confidence']*100:.1f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------
    # Result Tabs
    # ---------------------------------------------------
    result_tabs = st.tabs(["Tamper Detection & Visuals", "Forensic Evidence"])
    # ---- Tab 1: Visual Evidence Cards ----
    with result_tabs[0]:
        st.subheader("Visual Evidence")
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown("**Original Image**")
            st.image(res["original_image"], use_container_width=True)
        with v2:
            st.markdown("**Error Level Analysis (ELA)**")
            st.image(res["ela_image"], use_container_width=True)
        with v3:
            st.markdown("**Heatmap Localization**")
            st.image(res["tamper_mask_colored"], use_container_width=True)

    # ---- Tab 2: Forensic Evidence Details ----
    with result_tabs[1]:
        st.subheader("Quantitative Forensic Diagnostics")
        dossier = res["forensic_dossier"]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("ELA DCT Grid Error", f"{dossier['ela_metrics']['dct_grid_inconsistency']:.2f}")
        with col2:
            st.metric("PRNU Noise Dev", f"{dossier['sensor_noise_metrics']['noise_residual_std']:.2f}")
        with col3:
            st.metric("Copy‑Move Matches", f"{dossier['copy_move_metrics']['copy_move_duplicate_pairs']}")
        with col4:
            st.metric("Boundary Gradient", f"{dossier['boundary_metrics']['boundary_gradient_discontinuity']:.2f}x")
        with col5:
            st.metric("FFT Spectral Peak", f"{dossier['spectral_metrics']['fft_peak_to_average_ratio']:.2f}")

        st.markdown("---")
        st.subheader("Forensic Findings & Attribution Rationale")
        for point in dossier["forensic_evidence_summary"]:
            st.markdown(f"<div class='evidence-box'>📌 {point}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Attribution Class Probabilities")
        probs = res["type_probabilities"]
        df_probs = pd.DataFrame({"Method": list(probs.keys()), "Confidence (%)": [v*100 for v in probs.values()]})
        st.bar_chart(df_probs.set_index("Method"))
else:
    st.info("Upload an image or load a sample, then click **Run Analysis**.")
