import os
import sys
import time
import json
import uuid
import webbrowser
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import cfg
from src.infer import TamperInferenceEngine
from src.data_generator import SyntheticTamperingGenerator

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "web", "templates"),
    static_folder=os.path.join(BASE_DIR, "web", "static")
)
CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "web", "static", "uploads")
FIGURES_FOLDER = os.path.join(BASE_DIR, "results", "paper_figures")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)

# Initialize Inference Engine
print("Initializing ForensIQ Deep Forensic Model...")
engine = TamperInferenceEngine()
print("Model loaded successfully!")


@app.route("/")
def index():
    """Renders the main forensic web dashboard."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """
    Analyzes an uploaded image for tampering and returns:
      - Binary detection status (Authentic vs Tampered)
      - Attributed tampering method & probability breakdown
      - URLs for original, ELA, raw mask, binary mask, and localization overlays
      - Bounding box coordinates
      - Quantitative physical forensic evidence diagnostics
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    unique_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    orig_filename = f"upload_{unique_id}{ext}"
    orig_path = os.path.join(UPLOAD_FOLDER, orig_filename)
    file.save(orig_path)

    # Sliders / parameters
    ela_quality = int(request.form.get("ela_quality", cfg.ela_quality))
    ela_scale = float(request.form.get("ela_scale", cfg.ela_scale))
    mask_threshold = float(request.form.get("mask_threshold", 0.35))

    # Run inference
    res = engine.predict(
        orig_path,
        ela_quality=ela_quality,
        ela_scale=ela_scale,
        mask_threshold=mask_threshold
    )

    # Save visual outputs to static/uploads
    ela_filename = f"ela_{unique_id}.jpg"
    overlay_filename = f"overlay_{unique_id}.jpg"
    mask_filename = f"mask_{unique_id}.png"
    contour_filename = f"contour_{unique_id}.jpg"

    res["ela_image"].save(os.path.join(UPLOAD_FOLDER, ela_filename))
    res["localization_overlay"].save(os.path.join(UPLOAD_FOLDER, overlay_filename))
    res["tamper_mask_binary"].save(os.path.join(UPLOAD_FOLDER, mask_filename))
    res["contour_overlay"].save(os.path.join(UPLOAD_FOLDER, contour_filename))

    return jsonify({
        "success": True,
        "is_tampered": res["is_tampered"],
        "verdict": res["verdict"],
        "tamper_probability": float(res["tamper_probability"]),
        "predicted_type": res["predicted_type"],
        "type_confidence": float(res["type_confidence"]),
        "type_probabilities": res["type_probabilities"],
        "bounding_boxes": res["bounding_boxes"],
        "visuals": {
            "original": f"/static/uploads/{orig_filename}",
            "ela": f"/static/uploads/{ela_filename}",
            "overlay": f"/static/uploads/{overlay_filename}",
            "mask": f"/static/uploads/{mask_filename}",
            "contours": f"/static/uploads/{contour_filename}"
        },
        "forensic_dossier": res["forensic_dossier"]
    })


@app.route("/api/sample", methods=["GET"])
def analyze_sample():
    """Loads a pre-generated sample from the dataset and runs analysis."""
    img_folder = os.path.join(cfg.generated_data_dir, "images")
    if not os.path.exists(img_folder):
        return jsonify({"error": "Generated samples not found. Run dataset generation first."}), 404

    files = [f for f in os.listdir(img_folder) if f.endswith(".jpg")]
    if not files:
        return jsonify({"error": "No sample images found"}), 404

    import random
    choice = random.choice(files)
    sample_path = os.path.join(img_folder, choice)

    unique_id = str(uuid.uuid4())[:8]
    orig_filename = f"sample_{unique_id}.jpg"
    orig_path = os.path.join(UPLOAD_FOLDER, orig_filename)

    img = Image.open(sample_path).convert("RGB")
    img.save(orig_path)

    res = engine.predict(orig_path)

    ela_filename = f"ela_{unique_id}.jpg"
    overlay_filename = f"overlay_{unique_id}.jpg"
    mask_filename = f"mask_{unique_id}.png"
    contour_filename = f"contour_{unique_id}.jpg"

    res["ela_image"].save(os.path.join(UPLOAD_FOLDER, ela_filename))
    res["localization_overlay"].save(os.path.join(UPLOAD_FOLDER, overlay_filename))
    res["tamper_mask_binary"].save(os.path.join(UPLOAD_FOLDER, mask_filename))
    res["contour_overlay"].save(os.path.join(UPLOAD_FOLDER, contour_filename))

    return jsonify({
        "success": True,
        "sample_name": choice,
        "is_tampered": res["is_tampered"],
        "verdict": res["verdict"],
        "tamper_probability": float(res["tamper_probability"]),
        "predicted_type": res["predicted_type"],
        "type_confidence": float(res["type_confidence"]),
        "type_probabilities": res["type_probabilities"],
        "bounding_boxes": res["bounding_boxes"],
        "visuals": {
            "original": f"/static/uploads/{orig_filename}",
            "ela": f"/static/uploads/{ela_filename}",
            "overlay": f"/static/uploads/{overlay_filename}",
            "mask": f"/static/uploads/{mask_filename}",
            "contours": f"/static/uploads/{contour_filename}"
        },
        "forensic_dossier": res["forensic_dossier"]
    })


@app.route("/api/simulate", methods=["POST"])
def simulate_attack():
    """
    Simulates a live tampering attack (Copy-Move, Splicing, Inpainting, AI-Gen)
    on a procedural scene and runs the detection pipeline.
    """
    data = request.get_json(force=True, silent=True) or request.form or {}
    attack_type = data.get("attack_type", "Copy-Move")
    gen = SyntheticTamperingGenerator()
    base_img = gen.generate_procedural_scene()

    if attack_type == "Copy-Move":
        attacked, mask = gen.create_copy_move(base_img)
        true_name = "Copy-Move"
    elif attack_type == "Splicing":
        donor = gen.generate_procedural_scene()
        attacked, mask = gen.create_splicing(base_img, donor)
        true_name = "Splicing"
    elif attack_type == "Inpainting":
        attacked, mask = gen.create_inpainting(base_img)
        true_name = "Inpainting"
    else:
        attacked, mask = gen.create_ai_generated(base_img)
        true_name = "AI-Generated"

    unique_id = str(uuid.uuid4())[:8]
    base_fname = f"sim_base_{unique_id}.jpg"
    attacked_fname = f"sim_att_{unique_id}.jpg"
    gt_mask_fname = f"sim_gt_{unique_id}.png"

    Image.fromarray(base_img).save(os.path.join(UPLOAD_FOLDER, base_fname))
    Image.fromarray(attacked).save(os.path.join(UPLOAD_FOLDER, attacked_fname))
    Image.fromarray(mask).save(os.path.join(UPLOAD_FOLDER, gt_mask_fname))

    # Run inference on attacked image
    res = engine.predict(attacked)

    pred_mask_fname = f"sim_pred_mask_{unique_id}.png"
    contour_fname = f"sim_contour_{unique_id}.jpg"
    res["tamper_mask_binary"].save(os.path.join(UPLOAD_FOLDER, pred_mask_fname))
    res["contour_overlay"].save(os.path.join(UPLOAD_FOLDER, contour_fname))

    return jsonify({
        "success": True,
        "attack_type": true_name,
        "predicted_type": res["predicted_type"],
        "tamper_probability": float(res["tamper_probability"]),
        "type_confidence": float(res["type_confidence"]),
        "match": bool(res["predicted_type"] == true_name),
        "visuals": {
            "base": f"/static/uploads/{base_fname}",
            "attacked": f"/static/uploads/{attacked_fname}",
            "gt_mask": f"/static/uploads/{gt_mask_fname}",
            "predicted_mask": f"/static/uploads/{pred_mask_fname}",
            "contours": f"/static/uploads/{contour_fname}"
        }
    })


@app.route("/api/paper_figures", methods=["GET"])
def get_paper_figures():
    """Returns metadata and download URLs for research paper figures."""
    figures = [
        {
            "id": "fig1",
            "title": "Figure 1: Dual-Branch Neural Architecture Schematic",
            "description": "Publication-grade diagram illustrating tensor pathways, dual-stream ResNet backbones, cross-stream gated attention, and multi-task heads.",
            "url": "/static/figures/fig1_architecture.png",
            "filename": "fig1_architecture.png"
        },
        {
            "id": "fig2",
            "title": "Figure 2: Qualitative Multi-Tampering Attribution & Localization Grid",
            "description": "5-row x 5-column benchmark grid comparing Input, Ground Truth, ELA Residual, Predicted Heatmap Mask, and Contours.",
            "url": "/static/figures/fig2_qualitative_results_grid.png",
            "filename": "fig2_qualitative_results_grid.png"
        },
        {
            "id": "fig3",
            "title": "Figure 3: Multi-Class ROC Curves by Category",
            "description": "Receiver Operating Characteristic curves with AUC scores for each tampering category and macro-average.",
            "url": "/static/figures/fig3_roc_curves.png",
            "filename": "fig3_roc_curves.png"
        },
        {
            "id": "fig4",
            "title": "Figure 4: Training Loss and Metric Convergence",
            "description": "Convergence curves showing Multi-Task Loss, Binary Detection Accuracy, and Attribution Macro-F1 across epochs.",
            "url": "/static/figures/fig4_training_curves.png",
            "filename": "fig4_training_curves.png"
        },
        {
            "id": "fig5",
            "title": "Figure 5: Physical Forensic Evidence Radar Signatures",
            "description": "Radar plot comparing DCT inconsistency, PRNU noise dev, keypoint duplicate matches, boundary gradients, and FFT spectral peaks.",
            "url": "/static/figures/fig5_forensic_signatures.png",
            "filename": "fig5_forensic_signatures.png"
        }
    ]
    return jsonify(figures)


@app.route("/static/figures/<path:filename>")
def serve_figure(filename):
    """Serves high-resolution research paper figures."""
    return send_from_directory(FIGURES_FOLDER, filename)


def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    port = 5000
    print(f"\n=======================================================")
    print(f"  ForensIQ: Local Forensic Web Application")
    print(f"  Website URL: http://127.0.0.1:{port}")
    print(f"  Launching web browser...")
    print(f"=======================================================\n")
    
    # Auto-open browser in a non-blocking background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(host="127.0.0.1", port=port, debug=False)
