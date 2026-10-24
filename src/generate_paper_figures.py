"""
Generate all paper-quality figures for the DeepVision Forensics paper.

Figures produced:
  1. Training curves (loss + accuracy + F1 over epochs)
  2. ROC curve for binary tamper detection (on CASIA2)
  3. Confusion matrix for multi-class type attribution (on CASIA2)
  4. Cross-dataset comparison bar chart (CASIA2 vs REAL-TS)
  5. Qualitative results grid (sample images + ELA + predictions)
  6. ELA forensic signature visualization (pristine vs tampered)
  7. Per-class precision / recall / F1 bar chart (on CASIA2)

Usage:
    python src/generate_paper_figures.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from PIL import Image
import torch
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score, roc_auc_score,
    precision_recall_curve
)
from torch.utils.data import DataLoader

from src.config import cfg
from src.dataset import ForensicDataset
from src.models.dual_branch_net import DualBranchTamperDetector
from src.forensic_features import compute_ela

# ── Style setup ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
OUT_DIR = os.path.join(cfg.results_dir, "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: load model
# ═══════════════════════════════════════════════════════════════════════════════
def load_model(model_path=None, device=None):
    model_path = model_path or cfg.best_model_path
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    backbone = cfg.backbone
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        backbone = checkpoint["config"].get("backbone", cfg.backbone)
    model = DualBranchTamperDetector(num_classes=cfg.num_classes, backbone=backbone, pretrained=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: run inference on a dataset, return raw predictions
# ═══════════════════════════════════════════════════════════════════════════════
def run_inference(model, device, dataset_root):
    meta_path = os.path.join(dataset_root, "metadata.csv")
    df = pd.read_csv(meta_path)
    ds = ForensicDataset(metadata_df=df, is_training=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    all_bin_probs, all_bin_targets = [], []
    all_type_probs, all_type_targets = [], []
    all_paths = []

    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            ela = batch["ela"].to(device)
            outputs = model(rgb, ela, return_mask=False)

            bin_probs = torch.sigmoid(outputs["binary_logits"]).cpu().numpy().flatten()
            type_softmax = torch.softmax(outputs["type_logits"], dim=1).cpu().numpy()

            all_bin_probs.extend(bin_probs)
            all_bin_targets.extend(batch["binary_label"].numpy().flatten())
            all_type_probs.extend(type_softmax)
            all_type_targets.extend(batch["type_label"].numpy())
            all_paths.extend(batch["image_path"])

    return {
        "bin_probs": np.array(all_bin_probs),
        "bin_targets": np.array(all_bin_targets),
        "type_probs": np.array(all_type_probs),
        "type_targets": np.array(all_type_targets),
        "paths": all_paths,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1: Training Curves
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_training_curves():
    print("  Generating Fig 1: Training curves ...")
    history_path = os.path.join(cfg.results_dir, "training_history.json")
    with open(history_path) as f:
        h = json.load(f)

    epochs = list(range(1, len(h["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Loss
    ax = axes[0, 0]
    ax.plot(epochs, h["train_loss"], color=COLORS[0], linewidth=2, label="Train Loss")
    ax.plot(epochs, h["val_loss"], color=COLORS[1], linewidth=2, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Binary Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [v * 100 for v in h["val_binary_acc"]], color=COLORS[2], linewidth=2, label="Val Binary Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Binary Detection Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([85, 100])

    # (c) Binary F1
    ax = axes[1, 0]
    ax.plot(epochs, h["val_binary_f1"], color=COLORS[3], linewidth=2, label="Val Binary F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("(c) Binary Detection F1 Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.85, 1.0])

    # (d) Type Attribution Accuracy + F1
    ax = axes[1, 1]
    ax.plot(epochs, [v * 100 for v in h["val_type_acc"]], color=COLORS[4], linewidth=2, label="Val Type Acc")
    ax.plot(epochs, h["val_type_f1"], color=COLORS[0], linewidth=2, linestyle="--", label="Val Type F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("(d) Type Attribution Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training & Validation Metrics Over Epochs", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig1_training_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2: ROC Curve (Binary Detection) on CASIA2
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_roc_curve(preds):
    print("  Generating Fig 2: ROC curve ...")
    fpr, tpr, _ = roc_curve(preds["bin_targets"], preds["bin_probs"])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLORS[0], linewidth=2.5, label=f"DeepVision (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS[0])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – Binary Tampering Detection (CASIA2)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    # Inset: zoomed top-left
    axins = ax.inset_axes([0.4, 0.15, 0.45, 0.45])
    axins.plot(fpr, tpr, color=COLORS[0], linewidth=2)
    axins.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    axins.set_xlim([0, 0.15])
    axins.set_ylim([0.85, 1.0])
    axins.grid(True, alpha=0.3)
    axins.set_title("Zoomed", fontsize=9)
    ax.indicate_inset_zoom(axins, edgecolor="gray")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig2_roc_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3: Precision-Recall Curve on CASIA2
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_precision_recall(preds):
    print("  Generating Fig 3: Precision-Recall curve ...")
    precision, recall, _ = precision_recall_curve(preds["bin_targets"], preds["bin_probs"])
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=COLORS[2], linewidth=2.5, label=f"DeepVision (AP = {pr_auc:.4f})")
    ax.fill_between(recall, precision, alpha=0.15, color=COLORS[2])
    ax.axhline(y=preds["bin_targets"].mean(), color="gray", linestyle="--", linewidth=1, label="No-skill Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve – Binary Tampering Detection (CASIA2)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig3_precision_recall.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4: Confusion Matrix on CASIA2
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_confusion_matrix(preds):
    print("  Generating Fig 4: Confusion matrix ...")
    type_preds = np.argmax(preds["type_probs"], axis=1)
    type_targets = preds["type_targets"].astype(int)

    # Only use labels that are actually present
    present_labels = sorted(set(type_targets) | set(type_preds))
    present_names = [cfg.class_names[i] for i in present_labels]

    cm = confusion_matrix(type_targets, type_preds, labels=present_labels)
    # Normalize by row
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=present_names, yticklabels=present_names,
                cbar_kws={"shrink": 0.8})
    axes[0].set_xlabel("Predicted Type")
    axes[0].set_ylabel("Ground Truth Type")
    axes[0].set_title("(a) Confusion Matrix – Counts")

    # (b) Normalized (percentages)
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", ax=axes[1],
                xticklabels=present_names, yticklabels=present_names,
                cbar_kws={"shrink": 0.8})
    axes[1].set_xlabel("Predicted Type")
    axes[1].set_ylabel("Ground Truth Type")
    axes[1].set_title("(b) Confusion Matrix – Normalized")

    plt.suptitle("Tampering Type Attribution – CASIA2 Cross-Dataset Evaluation",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig4_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5: Cross-Dataset Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_cross_dataset_comparison(casia_preds, realts_preds):
    print("  Generating Fig 5: Cross-dataset comparison ...")

    def compute_metrics(preds):
        bin_preds = (preds["bin_probs"] > 0.5).astype(int)
        acc = accuracy_score(preds["bin_targets"], bin_preds)
        p, r, f1, _ = precision_recall_fscore_support(preds["bin_targets"], bin_preds, average="binary", zero_division=0)
        try:
            ra = roc_auc_score(preds["bin_targets"], preds["bin_probs"])
        except:
            ra = 0.0
        return {"Accuracy": acc, "Precision": p, "Recall": r, "F1-Score": f1, "ROC-AUC": ra}

    casia_m = compute_metrics(casia_preds)
    realts_m = compute_metrics(realts_preds)

    metrics = list(casia_m.keys())
    casia_vals = [casia_m[m] * 100 for m in metrics]
    realts_vals = [realts_m[m] * 100 for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, casia_vals, width, label="CASIA2 (12,614 imgs)", color=COLORS[0], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, realts_vals, width, label="REAL-TS (440 imgs)", color=COLORS[1], edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score (%)")
    ax.set_title("Cross-Dataset Generalization – Binary Tampering Detection", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 115])
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig5_cross_dataset_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6: ELA Forensic Signature Visualization
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_ela_visualization(casia_preds, model, device):
    print("  Generating Fig 6: ELA forensic signatures ...")

    # Pick one pristine and one tampered image
    pristine_idx = np.where(casia_preds["bin_targets"] == 0)[0]
    tampered_idx = np.where(casia_preds["bin_targets"] == 1)[0]

    # Select a few diverse samples
    np.random.seed(42)
    sel_pristine = np.random.choice(pristine_idx, size=min(3, len(pristine_idx)), replace=False)
    sel_tampered = np.random.choice(tampered_idx, size=min(3, len(tampered_idx)), replace=False)
    selections = list(sel_pristine) + list(sel_tampered)
    labels = ["Pristine"] * len(sel_pristine) + ["Tampered"] * len(sel_tampered)

    n = len(selections)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (idx, label) in enumerate(zip(selections, labels)):
        img_path = casia_preds["paths"][idx]
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except:
            pil_img = Image.new("RGB", (256, 256), (0, 0, 0))

        ela_img = compute_ela(pil_img, quality=cfg.ela_quality, scale=cfg.ela_scale)
        ela_enhanced = compute_ela(pil_img, quality=cfg.ela_quality, scale=30.0)

        # Column 0: Original
        axes[i, 0].imshow(pil_img.resize((256, 256)))
        axes[i, 0].set_title(f"Original ({label})", fontsize=10)
        axes[i, 0].axis("off")

        # Column 1: ELA (standard)
        axes[i, 1].imshow(ela_img.resize((256, 256)))
        axes[i, 1].set_title("ELA (scale=15)", fontsize=10)
        axes[i, 1].axis("off")

        # Column 2: ELA (enhanced)
        axes[i, 2].imshow(ela_enhanced.resize((256, 256)))
        axes[i, 2].set_title("ELA Enhanced (scale=30)", fontsize=10)
        axes[i, 2].axis("off")

        # Color-code row background
        color = "#e8f5e9" if label == "Pristine" else "#ffebee"
        for j in range(3):
            axes[i, j].patch.set_facecolor(color)

    plt.suptitle("Error Level Analysis (ELA) – Pristine vs Tampered Images",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig6_ela_visualization.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7: Qualitative Results Grid
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_qualitative_results(casia_preds, model, device):
    print("  Generating Fig 7: Qualitative results grid ...")

    # Select 4 correct and 2 incorrect predictions
    bin_preds_arr = (casia_preds["bin_probs"] > 0.5).astype(int)
    correct_mask = bin_preds_arr == casia_preds["bin_targets"].astype(int)
    incorrect_mask = ~correct_mask

    np.random.seed(123)
    correct_idx = np.where(correct_mask)[0]
    incorrect_idx = np.where(incorrect_mask)[0]

    # Pick 2 correct pristine, 2 correct tampered, 2 incorrect
    correct_pristine = correct_idx[casia_preds["bin_targets"][correct_idx] == 0]
    correct_tampered = correct_idx[casia_preds["bin_targets"][correct_idx] == 1]

    sel = []
    sel.extend(np.random.choice(correct_pristine, size=min(2, len(correct_pristine)), replace=False))
    sel.extend(np.random.choice(correct_tampered, size=min(2, len(correct_tampered)), replace=False))
    if len(incorrect_idx) > 0:
        sel.extend(np.random.choice(incorrect_idx, size=min(2, len(incorrect_idx)), replace=False))

    n = len(sel)
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 2.8))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, idx in enumerate(sel):
        img_path = casia_preds["paths"][idx]
        gt = int(casia_preds["bin_targets"][idx])
        prob = casia_preds["bin_probs"][idx]
        pred = int(prob > 0.5)
        correct = pred == gt

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except:
            pil_img = Image.new("RGB", (256, 256), (0, 0, 0))

        ela_img = compute_ela(pil_img, quality=cfg.ela_quality, scale=cfg.ela_scale)

        gt_str = "Pristine" if gt == 0 else "Tampered"
        pred_str = "Pristine" if pred == 0 else "Tampered"
        status = "✓" if correct else "✗"
        color = "#2e7d32" if correct else "#c62828"

        axes[i, 0].imshow(pil_img.resize((256, 256)))
        axes[i, 0].set_title(f"GT: {gt_str}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(ela_img.resize((256, 256)))
        axes[i, 1].set_title("ELA Map", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].bar(["Pristine", "Tampered"], [1 - prob, prob],
                       color=[COLORS[0], COLORS[1]], edgecolor="white")
        axes[i, 2].set_ylim([0, 1])
        axes[i, 2].set_title(f"{status} Pred: {pred_str} ({prob:.2f})", fontsize=9, color=color)
        axes[i, 2].set_ylabel("Confidence", fontsize=8)

    plt.suptitle("Qualitative Prediction Results – CASIA2 Cross-Dataset Evaluation",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig7_qualitative_results.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8: Per-Class Metrics Bar Chart (CASIA2)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_per_class_metrics(preds):
    print("  Generating Fig 8: Per-class metrics ...")
    type_preds = np.argmax(preds["type_probs"], axis=1)
    type_targets = preds["type_targets"].astype(int)

    present_labels = sorted(set(type_targets) | set(type_preds))
    present_names = [cfg.class_names[i] for i in present_labels]

    report = classification_report(type_targets, type_preds, labels=present_labels,
                                   target_names=present_names, output_dict=True, zero_division=0)

    precs = [report[n]["precision"] for n in present_names]
    recs = [report[n]["recall"] for n in present_names]
    f1s = [report[n]["f1-score"] for n in present_names]

    x = np.arange(len(present_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precs, width, label="Precision", color=COLORS[0], edgecolor="white")
    ax.bar(x, recs, width, label="Recall", color=COLORS[2], edgecolor="white")
    ax.bar(x + width, f1s, width, label="F1-Score", color=COLORS[3], edgecolor="white")

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Type Attribution Metrics – CASIA2", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(present_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.15])

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fig8_per_class_metrics.png")
    plt.savefig(path)
    plt.close()
    print(f"    Saved -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  DeepVision Forensics – Paper Figure Generator")
    print("=" * 60)

    # 1. Training curves (no model needed)
    fig1_training_curves()

    # 2-8. Load model and run inference
    print("\n  Loading model ...")
    model, device = load_model()

    casia_root = os.path.join(cfg.base_dir, "CASIA2")
    realts_root = os.path.join(cfg.base_dir, "REAL-TS")

    print(f"  Running inference on CASIA2 ({casia_root}) ...")
    casia_preds = run_inference(model, device, casia_root)
    print(f"    → {len(casia_preds['bin_targets'])} images processed")

    print(f"  Running inference on REAL-TS ({realts_root}) ...")
    realts_preds = run_inference(model, device, realts_root)
    print(f"    → {len(realts_preds['bin_targets'])} images processed")

    # Generate all figures
    print("\n  Generating figures ...\n")
    fig2_roc_curve(casia_preds)
    fig3_precision_recall(casia_preds)
    fig4_confusion_matrix(casia_preds)
    fig5_cross_dataset_comparison(casia_preds, realts_preds)
    fig6_ela_visualization(casia_preds, model, device)
    fig7_qualitative_results(casia_preds, model, device)
    fig8_per_class_metrics(casia_preds)

    # Save a summary JSON
    summary = {
        "casia2": {
            "n_images": len(casia_preds["bin_targets"]),
            "binary_accuracy": float(accuracy_score(casia_preds["bin_targets"], (casia_preds["bin_probs"] > 0.5).astype(int))),
            "roc_auc": float(roc_auc_score(casia_preds["bin_targets"], casia_preds["bin_probs"])),
        },
        "realts": {
            "n_images": len(realts_preds["bin_targets"]),
            "binary_accuracy": float(accuracy_score(realts_preds["bin_targets"], (realts_preds["bin_probs"] > 0.5).astype(int))),
            "roc_auc": float(roc_auc_score(realts_preds["bin_targets"], realts_preds["bin_probs"])),
        },
        "figures_generated": [
            "fig1_training_curves.png",
            "fig2_roc_curve.png",
            "fig3_precision_recall.png",
            "fig4_confusion_matrix.png",
            "fig5_cross_dataset_comparison.png",
            "fig6_ela_visualization.png",
            "fig7_qualitative_results.png",
            "fig8_per_class_metrics.png",
        ]
    }
    summary_path = os.path.join(OUT_DIR, "figures_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  All 8 figures saved to: {OUT_DIR}")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
