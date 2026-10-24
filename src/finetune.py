"""
Fine-tuning script for DeepVision Dual-Branch Tamper Detector.

Adapts an existing trained checkpoint to a target dataset (e.g. CASIA2 or custom)
using a low learning rate, optional backbone freezing, and optional compression augmentation.

Usage Examples:
    # 1. Fine-tune on CASIA2 for 5 epochs with low LR:
    python src/finetune.py --dataset_root "C:/Users/agvar/OneDrive/Desktop/DL Project/CASIA2" --epochs 5 --lr 2e-5

    # 2. Fine-tune with backbone frozen (head-only adaptation):
    python src/finetune.py --dataset_root "C:/Users/agvar/OneDrive/Desktop/DL Project/CASIA2" --freeze_backbone --epochs 5

    # 3. Fine-tune with random JPEG simulation for format robustness:
    python src/finetune.py --dataset_root "C:/Users/agvar/OneDrive/Desktop/DL Project/CASIA2" --simulate_jpeg --epochs 5
"""

import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Any, Optional
from tqdm import tqdm

from src.config import cfg
from src.dataset import ForensicDataset, get_dataloaders
from src.models.dual_branch_net import DualBranchTamperDetector


class DiceLoss(nn.Module):
    """Dice loss for mask segmentation / localization."""
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


def train_one_epoch(model, dataloader, optimizer, bce_loss, ce_loss, dice_loss, device):
    model.train()
    running_loss = 0.0
    all_bin_preds, all_bin_targets = [], []

    for batch in tqdm(dataloader, desc="Fine-Tuning", leave=False):
        rgb = batch["rgb"].to(device)
        ela = batch["ela"].to(device)
        bin_targets = batch["binary_label"].to(device).unsqueeze(1)
        type_targets = batch["type_label"].to(device)
        mask_targets = batch["mask"].to(device)

        optimizer.zero_grad()

        outputs = model(rgb, ela, return_mask=True)
        bin_logits = outputs["binary_logits"]
        type_logits = outputs["type_logits"]
        mask_logits = outputs["mask_logits"]

        loss_bin = bce_loss(bin_logits, bin_targets)
        loss_type = ce_loss(type_logits, type_targets)
        loss_mask = dice_loss(mask_logits, mask_targets)

        # Multi-task objective: prioritize binary detection during fine-tuning
        total_loss = 1.5 * loss_bin + 0.8 * loss_type + 0.5 * loss_mask
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        bin_preds = (torch.sigmoid(bin_logits) > 0.5).float().cpu().numpy().flatten()
        all_bin_preds.extend(bin_preds)
        all_bin_targets.extend(bin_targets.cpu().numpy().flatten())

    train_acc = accuracy_score(all_bin_targets, all_bin_preds)
    return running_loss / max(len(dataloader), 1), train_acc


def validate(model, dataloader, bce_loss, ce_loss, dice_loss, device):
    model.eval()
    running_loss = 0.0
    all_bin_preds, all_bin_targets = [], []
    all_bin_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            rgb = batch["rgb"].to(device)
            ela = batch["ela"].to(device)
            bin_targets = batch["binary_label"].to(device).unsqueeze(1)
            type_targets = batch["type_label"].to(device)
            mask_targets = batch["mask"].to(device)

            outputs = model(rgb, ela, return_mask=True)
            bin_logits = outputs["binary_logits"]
            type_logits = outputs["type_logits"]
            mask_logits = outputs["mask_logits"]

            loss_bin = bce_loss(bin_logits, bin_targets)
            loss_type = ce_loss(type_logits, type_targets)
            loss_mask = dice_loss(mask_logits, mask_targets)

            total_loss = 1.5 * loss_bin + 0.8 * loss_type + 0.5 * loss_mask
            running_loss += total_loss.item()

            probs = torch.sigmoid(bin_logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            all_bin_probs.extend(probs)
            all_bin_preds.extend(preds)
            all_bin_targets.extend(bin_targets.cpu().numpy().flatten())

    val_acc = accuracy_score(all_bin_targets, all_bin_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_bin_targets, all_bin_preds, average='binary', zero_division=0)
    return running_loss / max(len(dataloader), 1), val_acc, p, r, f1


def finetune(
    dataset_root: str,
    checkpoint_in: Optional[str] = None,
    checkpoint_out: Optional[str] = None,
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 16,
    val_split: float = 0.2,
    freeze_backbone: bool = False,
    simulate_jpeg: bool = False,
    device_str: Optional[str] = None
):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_in = checkpoint_in or cfg.best_model_path
    checkpoint_out = checkpoint_out or os.path.join(cfg.checkpoint_dir, "finetuned_tamper_detector.pth")
    os.makedirs(os.path.dirname(checkpoint_out), exist_ok=True)

    print(f"\n=======================================================")
    print(f" DeepVision Forensics - Fine-Tuning Pipeline")
    print(f" Source Checkpoint: {checkpoint_in}")
    print(f" Target Checkpoint: {checkpoint_out}")
    print(f" Dataset Root:      {dataset_root}")
    print(f" Device:            {device}")
    print(f" Epochs:            {epochs} | LR: {lr} | Batch Size: {batch_size}")
    print(f" Freeze Backbone:   {freeze_backbone} | JPEG Simulation: {simulate_jpeg}")
    print(f"=======================================================\n")

    if not os.path.exists(checkpoint_in):
        raise FileNotFoundError(f"Source checkpoint not found at: {checkpoint_in}")

    # Load dataset
    meta_path = os.path.join(dataset_root, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.csv not found in {dataset_root}")
    df = pd.read_csv(meta_path)
    print(f"Loaded {len(df)} images from {meta_path}")

    train_loader, val_loader, _ = get_dataloaders(
        metadata_df=df,
        batch_size=batch_size,
        val_split=val_split,
        test_split=0.05,
        simulate_jpeg=simulate_jpeg,
        random_jpeg_quality=simulate_jpeg
    )

    # Load Model
    ckpt = torch.load(checkpoint_in, map_location=device, weights_only=False)
    backbone = cfg.backbone
    if isinstance(ckpt, dict) and "config" in ckpt:
        backbone = ckpt["config"].get("backbone", cfg.backbone)

    model = DualBranchTamperDetector(num_classes=cfg.num_classes, backbone=backbone, pretrained=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    if freeze_backbone:
        print("[!] Freezing RGB and ELA feature extractor backbones; only heads will train.")
        for param in model.rgb_branch.parameters():
            param.requires_grad = False
        for param in model.ela_branch.parameters():
            param.requires_grad = False

    model = model.to(device)

    # Losses & Optimizer
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_f1 = -1.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_f1": []
    }

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, bce_loss, ce_loss, dice_loss, device)
        val_loss, val_acc, val_p, val_r, val_f1 = validate(model, val_loader, bce_loss, ce_loss, dice_loss, device)
        elapsed = time.time() - t0

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1:.4f} "
              f"(Prec: {val_p*100:.1f}%, Rec: {val_r*100:.1f}%)")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "backbone": backbone,
                    "img_size": cfg.img_size,
                    "class_names": cfg.class_names,
                    "finetuned_on": dataset_root
                }
            }, checkpoint_out)
            print(f"  --> Saved new best checkpoint to {checkpoint_out} (Val F1: {best_f1:.4f})")

    history_path = os.path.join(cfg.results_dir, "finetune_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nFine-tuning finished!")
    print(f"Best model: {checkpoint_out}")
    print(f"History:    {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeepVision tamper detector on target dataset")
    parser.add_argument("--dataset_root", type=str, required=True, help="Folder containing metadata.csv for fine-tuning")
    parser.add_argument("--checkpoint_in", type=str, default=None, help="Base model checkpoint to start from")
    parser.add_argument("--checkpoint_out", type=str, default=None, help="Where to save the fine-tuned checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone feature extractors")
    parser.add_argument("--simulate_jpeg", action="store_true", help="Apply random JPEG compression simulation during training")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    finetune(
        dataset_root=args.dataset_root,
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
        freeze_backbone=args.freeze_backbone,
        simulate_jpeg=args.simulate_jpeg,
        device_str=args.device
    )
