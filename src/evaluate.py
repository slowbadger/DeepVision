import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)
import torch
from typing import Dict, Any, Optional

from src.config import cfg
from src.dataset import ForensicDataset, get_dataloaders
from src.models.dual_branch_net import DualBranchTamperDetector

def evaluate_model(
    model_path: Optional[str] = None,
    test_loader = None,
    device: Optional[str] = None,
    dataset_root: Optional[str] = None,
    simulate_jpeg: bool = False,
    pre_jpeg_quality: int = 95,
    tta: bool = False,
    threshold: float = 0.5,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the tampering detection model:
      - Binary accuracy & F1
      - Multi-class attribution report
      - Confusion matrix generation
      - ROC-AUC computation
      - Threshold calibration (optimal operating point)
      - Optional in-memory JPEG compression simulation
      - Optional Test-Time Augmentation (TTA)
    """
    model_path = model_path or cfg.best_model_path
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = output_dir or cfg.results_dir
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train the model first.")

    # Load model and auto-detect backbone
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    backbone = cfg.backbone
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        backbone = checkpoint["config"].get("backbone", cfg.backbone)

    model = DualBranchTamperDetector(num_classes=cfg.num_classes, backbone=backbone, pretrained=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if test_loader is None:
        # Determine which metadata CSV to load
        if dataset_root:
            meta_path = os.path.join(dataset_root, "metadata.csv")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"metadata.csv not found in {dataset_root}")
        else:
            meta_path = os.path.join(cfg.generated_data_dir, "metadata.csv")
        df = pd.read_csv(meta_path)

        if dataset_root:
            # External / unseen dataset → use ALL images as the test set (no split)
            from torch.utils.data import DataLoader
            test_ds = ForensicDataset(
                metadata_df=df,
                is_training=False,
                simulate_jpeg=simulate_jpeg,
                pre_jpeg_quality=pre_jpeg_quality
            )
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
            status_msg = f"External dataset: using ALL {len(df)} images for evaluation."
            if simulate_jpeg:
                status_msg += f" (JPEG simulation active @ Q={pre_jpeg_quality})"
            if tta:
                status_msg += " (TTA active)"
            print(status_msg)
        else:
            # Internal dataset → use the standard train/val/test split
            _, _, test_loader = get_dataloaders(
                df, batch_size=cfg.batch_size,
                simulate_jpeg=simulate_jpeg,
                pre_jpeg_quality=pre_jpeg_quality
            )

    all_bin_preds = []
    all_bin_probs = []
    all_bin_targets = []
    all_type_preds = []
    all_type_probs = []
    all_type_targets = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            rgb = batch["rgb"].to(device)
            ela = batch["ela"].to(device)
            bin_targets = batch["binary_label"].to(device).unsqueeze(1)
            type_targets = batch["type_label"].to(device)

            if tta:
                out1 = model(rgb, ela, return_mask=False)
                out2 = model(torch.flip(rgb, dims=[-1]), torch.flip(ela, dims=[-1]), return_mask=False)
                bin_logits = (out1["binary_logits"] + out2["binary_logits"]) / 2.0
                type_logits = (out1["type_logits"] + out2["type_logits"]) / 2.0
            else:
                outputs = model(rgb, ela, return_mask=False)
                bin_logits = outputs["binary_logits"]
                type_logits = outputs["type_logits"]

            bin_probs = torch.sigmoid(bin_logits).cpu().numpy().flatten()
            bin_preds = (bin_probs >= threshold).astype(int)

            type_softmax = torch.softmax(type_logits, dim=1).cpu().numpy()
            type_preds = np.argmax(type_softmax, axis=1)

            all_bin_probs.extend(bin_probs)
            all_bin_preds.extend(bin_preds)
            all_bin_targets.extend(bin_targets.cpu().numpy().flatten())

            all_type_probs.extend(type_softmax)
            all_type_preds.extend(type_preds)
            all_type_targets.extend(type_targets.cpu().numpy())

    all_bin_targets = np.array(all_bin_targets)
    all_bin_preds = np.array(all_bin_preds)
    all_bin_probs = np.array(all_bin_probs)

    all_type_targets = np.array(all_type_targets)
    all_type_preds = np.array(all_type_preds)

    # 1. Binary Metrics at specified threshold
    bin_acc = accuracy_score(all_bin_targets, all_bin_preds)
    bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(all_bin_targets, all_bin_preds, average='binary', zero_division=0)
    try:
        bin_auc = roc_auc_score(all_bin_targets, all_bin_probs)
    except Exception:
        bin_auc = 0.0

    # 1b. Optimal threshold calibration (best F1 operating point)
    best_th = 0.5
    best_th_f1 = -1.0
    best_th_acc = 0.0
    for th in np.linspace(0.1, 0.95, 86):
        th_p = (all_bin_probs >= th).astype(int)
        _, _, cur_f1, _ = precision_recall_fscore_support(all_bin_targets, th_p, average='binary', zero_division=0)
        if cur_f1 > best_th_f1:
            best_th_f1 = cur_f1
            best_th = float(th)
            best_th_acc = float(accuracy_score(all_bin_targets, th_p))

    # 2. Multi-Class Type Attribution Metrics
    type_acc = accuracy_score(all_type_targets, all_type_preds)
    type_p, type_r, type_f1, _ = precision_recall_fscore_support(all_type_targets, all_type_preds, average='macro', zero_division=0)

    class_names = cfg.class_names
    target_names = [class_names[i] for i in sorted(list(set(all_type_targets).union(set(all_type_preds))))]
    report_dict = classification_report(all_type_targets, all_type_preds, target_names=target_names, output_dict=True, zero_division=0)

    # 3. Confusion Matrix Plot
    cm = confusion_matrix(all_type_targets, all_type_preds, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Tampering Type Confusion Matrix')
    plt.xlabel('Predicted Type')
    plt.ylabel('Ground Truth Type')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()

    results = {
        "binary_detection": {
            "threshold": float(threshold),
            "accuracy": float(bin_acc),
            "precision": float(bin_p),
            "recall": float(bin_r),
            "f1_score": float(bin_f1),
            "roc_auc": float(bin_auc),
            "optimal_threshold": {
                "threshold": best_th,
                "f1_score": float(best_th_f1),
                "accuracy": float(best_th_acc)
            }
        },
        "type_attribution": {
            "accuracy": float(type_acc),
            "macro_precision": float(type_p),
            "macro_recall": float(type_r),
            "macro_f1": float(type_f1),
            "report": report_dict
        },
        "confusion_matrix_path": cm_path
    }

    eval_json_path = os.path.join(save_dir, "evaluation_metrics.json")
    with open(eval_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Binary Detection (Threshold = {threshold:.2f}):")
    print(f"  Accuracy:  {bin_acc*100:.2f}%")
    print(f"  Precision: {bin_p*100:.2f}%")
    print(f"  Recall:    {bin_r*100:.2f}%")
    print(f"  F1-Score:  {bin_f1:.4f}")
    print(f"  ROC-AUC:   {bin_auc:.4f}")
    print(f"\nCalibrated Operating Point:")
    print(f"  Optimal Threshold: {best_th:.2f}")
    print(f"  Calibrated F1:     {best_th_f1:.4f} (Acc: {best_th_acc*100:.2f}%)")
    print(f"\nTampering Type Attribution:")
    print(f"  Accuracy:  {type_acc*100:.2f}%")
    print(f"  Macro F1:  {type_f1:.4f}")
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved metrics JSON:     {eval_json_path}")
    print("====================================================\n")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DeepVision model on a dataset.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint (defaults to cfg.best_model_path)")
    parser.add_argument("--dataset_root", type=str, default=None, help="Root folder containing metadata.csv for the test dataset")
    parser.add_argument("--simulate_jpeg", action="store_true", help="Simulate JPEG compression in memory (recommended for lossless TIFF/PNG like REAL-TS)")
    parser.add_argument("--pre_jpeg_quality", type=int, default=95, help="Quality for in-memory JPEG simulation (default: 95)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for binary tampering classification (default: 0.5)")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (flips)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save confusion matrix and JSON metrics")
    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        dataset_root=args.dataset_root,
        simulate_jpeg=args.simulate_jpeg,
        pre_jpeg_quality=args.pre_jpeg_quality,
        threshold=args.threshold,
        tta=args.tta,
        output_dir=args.output_dir
    )

