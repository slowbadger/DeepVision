import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
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


class Trainer:
    """Multi-task trainer for image tampering detection & attribution."""
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        device: Optional[str] = None,
        lr: float = cfg.learning_rate,
        weight_decay: float = cfg.weight_decay,
        checkpoint_dir: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Trainer initialized using device: {self.device}")

        self.model = model if model else DualBranchTamperDetector(
            num_classes=cfg.num_classes,
            backbone=cfg.backbone,
            pretrained=cfg.pretrained
        )
        self.model = self.model.to(self.device)

        self.checkpoint_dir = checkpoint_dir or cfg.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.results_dir, exist_ok=True)

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)

        # Enable cuDNN benchmark for faster convolutions on fixed-size inputs
        # History to store per‑epoch metrics
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_binary_acc": [], "val_binary_f1": [],
            "val_type_acc": [], "val_type_f1": []
        }


    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        running_total_loss = 0.0
        running_bin_loss = 0.0
        running_type_loss = 0.0
        running_mask_loss = 0.0

        all_type_preds = []
        all_type_targets = []
        all_bin_preds = []
        all_bin_targets = []

        for batch in tqdm(train_loader, desc='Training', leave=False):
            rgb = batch["rgb"].to(self.device)
            ela = batch["ela"].to(self.device)
            bin_targets = batch["binary_label"].to(self.device).unsqueeze(1)
            type_targets = batch["type_label"].to(self.device)
            mask_targets = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(rgb, ela, return_mask=True)
            bin_logits = outputs["binary_logits"]
            type_logits = outputs["type_logits"]
            mask_logits = outputs["mask_logits"]

            # Multi-task loss
            loss_bin = self.bce_loss(bin_logits, bin_targets)
            loss_type = self.ce_loss(type_logits, type_targets)
            loss_mask = self.bce_loss(mask_logits, mask_targets) + self.dice_loss(mask_logits, mask_targets)

            total_loss = 1.0 * loss_bin + 1.0 * loss_type + 0.5 * loss_mask
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            running_total_loss += total_loss.item()
            running_bin_loss += loss_bin.item()
            running_type_loss += loss_type.item()
            running_mask_loss += loss_mask.item()

            # Record predictions
            bin_preds = (torch.sigmoid(bin_logits) > 0.5).float().cpu().numpy()
            type_preds = torch.argmax(type_logits, dim=1).cpu().numpy()

            all_bin_preds.extend(bin_preds.flatten())
            all_bin_targets.extend(bin_targets.cpu().numpy().flatten())
            all_type_preds.extend(type_preds)
            all_type_targets.extend(type_targets.cpu().numpy())

        num_batches = len(train_loader)
        train_bin_acc = accuracy_score(all_bin_targets, all_bin_preds)
        train_type_acc = accuracy_score(all_type_targets, all_type_preds)

        return {
            "loss": running_total_loss / num_batches,
            "bin_loss": running_bin_loss / num_batches,
            "type_loss": running_type_loss / num_batches,
            "mask_loss": running_mask_loss / num_batches,
            "bin_acc": train_bin_acc,
            "type_acc": train_type_acc
        }

    def evaluate(self, val_loader) -> Dict[str, Any]:
        self.model.eval()
        running_total_loss = 0.0
        all_type_preds = []
        all_type_targets = []
        all_bin_preds = []
        all_bin_targets = []

        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(self.device)
                ela = batch["ela"].to(self.device)
                bin_targets = batch["binary_label"].to(self.device).unsqueeze(1)
                type_targets = batch["type_label"].to(self.device)
                mask_targets = batch["mask"].to(self.device)

                outputs = self.model(rgb, ela, return_mask=True)
                bin_logits = outputs["binary_logits"]
                type_logits = outputs["type_logits"]
                mask_logits = outputs["mask_logits"]

                loss_bin = self.bce_loss(bin_logits, bin_targets)
                loss_type = self.ce_loss(type_logits, type_targets)
                loss_mask = self.bce_loss(mask_logits, mask_targets) + self.dice_loss(mask_logits, mask_targets)
                total_loss = 1.0 * loss_bin + 1.0 * loss_type + 0.5 * loss_mask

                running_total_loss += total_loss.item()

                bin_preds = (torch.sigmoid(bin_logits) > 0.5).float().cpu().numpy()
                type_preds = torch.argmax(type_logits, dim=1).cpu().numpy()

                all_bin_preds.extend(bin_preds.flatten())
                all_bin_targets.extend(bin_targets.cpu().numpy().flatten())
                all_type_preds.extend(type_preds)
                all_type_targets.extend(type_targets.cpu().numpy())

        num_batches = len(val_loader)
        val_loss = running_total_loss / num_batches

        bin_acc = accuracy_score(all_bin_targets, all_bin_preds)
        _, _, bin_f1, _ = precision_recall_fscore_support(all_bin_targets, all_bin_preds, average='binary', zero_division=0)

        type_acc = accuracy_score(all_type_targets, all_type_preds)
        _, _, type_f1, _ = precision_recall_fscore_support(all_type_targets, all_type_preds, average='macro', zero_division=0)

        return {
            "loss": val_loss,
            "bin_acc": bin_acc,
            "bin_f1": bin_f1,
            "type_acc": type_acc,
            "type_f1": type_f1,
            "all_bin_preds": all_bin_preds,
            "all_bin_targets": all_bin_targets,
            "all_type_preds": all_type_preds,
            "all_type_targets": all_type_targets
        }

    def fit(self, train_loader, val_loader, epochs: int = cfg.epochs, save_path: Optional[str] = None) -> str:
        best_save_path = save_path or cfg.best_model_path
        best_score = -1.0

        print(f"\n=======================================================")
        print(f" Starting Dual-Branch Tampering Detector Training")
        print(f" Epochs: {epochs} | Device: {self.device} | Backbone: {cfg.backbone}")
        print(f" Target Classes: {cfg.class_names}")
        print(f"=======================================================\n")

        # Progress bar for epochs
        epoch_bar = tqdm(total=epochs, desc='Epochs', unit='epoch')
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            # -------- Training --------
            train_metrics = self.train_epoch(train_loader)
            # -------- Validation --------
            val_metrics = self.evaluate(val_loader)
            elapsed = time.time() - t0

            # Update scheduler based on validation binary accuracy (more stable)
            self.scheduler.step(val_metrics["bin_acc"])

            # Record history for later plotting or inspection
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_binary_acc"].append(val_metrics["bin_acc"])
            self.history["val_binary_f1"].append(val_metrics["bin_f1"])
            self.history["val_type_acc"].append(val_metrics["type_acc"])
            self.history["val_type_f1"].append(val_metrics["type_f1"])

            print(f"Epoch {epoch:03d} | Time: {elapsed:.2f}s | Train Loss: {train_metrics['loss']:.4f} "
                  f"| Val Loss: {val_metrics['loss']:.4f} "
                  f"| Val Bin Acc: {val_metrics['bin_acc']:.4f} "
                  f"| Val Bin F1: {val_metrics['bin_f1']:.4f} "
                  f"| Val Type Acc: {val_metrics['type_acc']:.4f} "
                  f"| Val Type F1: {val_metrics['type_f1']:.4f}")

            # Save best model based on combined binary + type accuracy
            current_score = val_metrics["bin_acc"] + val_metrics["type_acc"]
            if current_score > best_score:
                best_score = current_score
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "config": {
                        "backbone": cfg.backbone,
                        "img_size": cfg.img_size,
                        "class_names": cfg.class_names,
                    }
                }, best_save_path)
                print(f"[✓] New best model saved (score={best_score:.4f}) to {best_save_path}")

            epoch_bar.update(1)
        epoch_bar.close()

        # Save training history
        history_path = os.path.join(cfg.results_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\nTraining complete! Best checkpoint saved to {best_save_path}")
        return best_save_path




