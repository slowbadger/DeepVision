import argparse
import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from src.models.dual_branch_net import DualBranchTamperDetector
from src.config import cfg
from src.forensic_features import compute_ela

def preprocess_image(image_path: str):
    """Load an image, compute ELA, and apply the same transforms as the web engine.
    Returns tensors (rgb, ela) ready for the model.
    """
    img = Image.open(image_path).convert('RGB')
    # Compute ELA using the same utility
    ela = compute_ela(img, quality=cfg.ela_quality, scale=cfg.ela_scale)
    # Define transforms (same as in TamperInferenceEngine)
    rgb_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ela_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor()
    ])
    rgb_tensor = rgb_transform(img).unsqueeze(0)
    ela_tensor = ela_transform(ela).unsqueeze(0)
    return rgb_tensor, ela_tensor

def load_model():
    model_path = os.path.join('checkpoints', 'deepvision_forensics.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TorchScript model not found at {model_path}. Run export_model.py first.")
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='Run inference with DeepVision Forensics model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    args = parser.parse_args()

    rgb, ela = preprocess_image(args.image)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    rgb = rgb.to(device)
    ela = ela.to(device)
    model = load_model().to(device)

    with torch.no_grad():
        outputs = model(rgb, ela)
        # The scripted model returns a dict with the same keys as the original engine
        binary_logits = outputs["binary_logits"]
        type_logits = outputs["type_logits"]
        # binary_logits shape: (1,1) – sigmoid gives tampered probability
        binary_prob = torch.sigmoid(binary_logits).item()
        pred_type_idx = torch.argmax(type_logits, dim=1).item()
        class_name = cfg.class_names[pred_type_idx]
        print(f"Binary tampering probability (tampered): {binary_prob:.4f}")
        print(f"Predicted tampering type: {class_name}")

if __name__ == '__main__':
    main()
