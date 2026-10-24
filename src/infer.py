import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Dict, Any, Union, Optional

from src.config import cfg
from src.forensic_features import compute_ela
from src.models.dual_branch_net import DualBranchTamperDetector
from src.forensic_analyzer import ForensicEvidenceAnalyzer


class TamperInferenceEngine:
    """
    Inference Engine for Image Tampering Detection & Attribution:
      - Binary detection (Authentic vs Tampered)
      - Fine‑grained tampering type attribution
      - Spatial localization heatmap mask & bounding contours
      - Forensic evidence & signal diagnostics
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        img_size: int = cfg.img_size,
        torchscript_path: Optional[str] = None,
    ):
        # Device setup
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.img_size = img_size

        # Transforms (shared for both TorchScript and checkpoint models)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.ela_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

        # Prefer TorchScript model if available
        default_ts_path = os.path.join('checkpoints', 'deepvision_forensics.pt')
        self.torchscript_path = torchscript_path or (default_ts_path if os.path.exists(default_ts_path) else None)
        if self.torchscript_path:
            self.model = torch.jit.load(self.torchscript_path, map_location=self.device)
            self.model.eval()
            self.use_torchscript = True
            print(f"Loaded TorchScript model from {self.torchscript_path}")
        else:
            # Fallback to checkpoint loading
            self.checkpoint_path = checkpoint_path or cfg.best_model_path
            self.use_torchscript = False
            backbone = cfg.backbone
            state_dict = None
            if os.path.exists(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and "config" in checkpoint:
                    backbone = checkpoint["config"].get("backbone", cfg.backbone)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            self.model = DualBranchTamperDetector(
                num_classes=cfg.num_classes,
                backbone=backbone,
                pretrained=False
            )
            if state_dict is not None:
                self.model.load_state_dict(state_dict)
                print(f"Loaded checkpoint ({backbone}) from: {self.checkpoint_path}")
            else:
                print(f"Warning: Checkpoint not found at {self.checkpoint_path}. Using initialized weights.")
            self.model = self.model.to(self.device)
            self.model.eval()

    def predict(
        self,
        image_input: Union[str, Image.Image, np.ndarray],
        ela_quality: int = cfg.ela_quality,
        ela_scale: float = cfg.ela_scale,
        mask_threshold: float = 0.35
    ) -> Dict[str, Any]:
        """Run comprehensive inference on an input image and return diagnostics.

        A safety check ensures the required transforms are present even if the
        object was instantiated in a context where ``__init__`` did not run
        (e.g., during hot‑reloading)."""
        # Ensure transforms exist – this guards against the AttributeError seen in logs.
        if not hasattr(self, "rgb_transform"):
            self.rgb_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if not hasattr(self, "ela_transform"):
            self.ela_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ])

        """Run comprehensive inference on an input image and return diagnostics."""
        # Load image
        if isinstance(image_input, str):
            orig_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            orig_pil = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            orig_pil = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input type.")

        orig_w, orig_h = orig_pil.size
        orig_np = np.array(orig_pil)

        # Compute forensic ELA
        ela_pil = compute_ela(orig_pil, quality=ela_quality, scale=ela_scale)

        # Preprocess tensors
        rgb_tensor = self.rgb_transform(orig_pil).unsqueeze(0).to(self.device)
        ela_tensor = self.ela_transform(ela_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.use_torchscript:
                outputs = self.model(rgb_tensor, ela_tensor)
            else:
                outputs = self.model(rgb_tensor, ela_tensor, return_mask=True)

            # Binary prediction
            bin_prob = torch.sigmoid(outputs["binary_logits"]).item()
            is_tampered = bool(bin_prob > 0.5)

            # Tampering type attribution
            type_probs = torch.softmax(outputs["type_logits"], dim=1).cpu().numpy()[0]
            pred_type_idx = int(np.argmax(type_probs))

            # Mask localization heatmap
            mask_logits = outputs.get("mask_logits")
            if mask_logits is not None:
                mask_prob = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
                mask_resized = cv2.resize(mask_prob, (orig_w, orig_h))
            else:
                mask_resized = np.zeros((orig_h, orig_w), dtype=np.float32)

        # Post‑processing for visualisation
        if not is_tampered and pred_type_idx != 0:
            if type_probs[0] > 0.35:
                pred_type_idx = 0
        pred_type_name = cfg.class_names[pred_type_idx]
        type_breakdown = {cfg.class_names[i]: float(type_probs[i]) for i in range(len(cfg.class_names))}

        # Binary mask & contours
        binary_mask_np = ((mask_resized > mask_threshold) * 255).astype(np.uint8) if is_tampered else np.zeros((orig_h, orig_w), dtype=np.uint8)
        binary_mask_pil = Image.fromarray(binary_mask_np)
        contours, _ = cv2.findContours(binary_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        contour_viz_np = orig_np.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append((int(x), int(y), int(w), int(h)))
                cv2.rectangle(contour_viz_np, (x, y), (x + w, y + h), (255, 50, 50), 2)
                cv2.drawContours(contour_viz_np, [cnt], -1, (0, 255, 255), 2)
                cv2.putText(contour_viz_np, f"{pred_type_name}", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)
        contour_viz_pil = Image.fromarray(contour_viz_np)

        # Heatmap visualisation
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        heatmap_colored_pil = Image.fromarray(heatmap_colored)
        alpha = 0.50 if is_tampered else 0.15
        overlay = np.uint8((1.0 - alpha) * orig_np + alpha * heatmap_colored)
        overlay_pil = Image.fromarray(overlay)

        # Forensic dossier
        forensic_dossier = ForensicEvidenceAnalyzer.generate_full_forensic_dossier(
            orig_np,
            pred_type=pred_type_name,
            type_confidence=float(type_probs[pred_type_idx]),
            mask=mask_resized
        )

        return {
            "is_tampered": is_tampered,
            "tamper_probability": float(bin_prob),
            "verdict": "TAMPERED" if is_tampered else "AUTHENTIC (PRISTINE)",
            "predicted_type": pred_type_name,
            "type_confidence": float(type_probs[pred_type_idx]),
            "type_probabilities": type_breakdown,
            "original_image": orig_pil,
            "ela_image": ela_pil,
            "tamper_mask_raw": mask_resized,
            "tamper_mask_binary": binary_mask_pil,
            "tamper_mask_colored": heatmap_colored_pil,
            "localization_overlay": overlay_pil,
            "contour_overlay": contour_viz_pil,
            "bounding_boxes": bounding_boxes,
            "forensic_dossier": forensic_dossier
        }

if __name__ == "__main__":
    engine = TamperInferenceEngine()
    print("Inference Engine ready.")
