import os
import torch
from src.models.dual_branch_net import DualBranchTamperDetector
from src.config import cfg

def main():
    # Path to the best checkpoint saved during training
    ckpt_path = os.path.join(cfg.checkpoint_dir, "best_tamper_detector.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    # Load checkpoint (saved dictionary with model state and config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Re‑create the model using the saved config (backbone, num_classes, etc.)
    model_cfg = ckpt.get("config", {})
    model = DualBranchTamperDetector(
        num_classes=cfg.num_classes,
        backbone=model_cfg.get("backbone", cfg.backbone),
        pretrained=model_cfg.get("pretrained", cfg.pretrained),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Export to TorchScript (trace) – we need dummy RGB and ELA tensors
    dummy_rgb = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    dummy_ela = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    scripted = torch.jit.trace(model, (dummy_rgb, dummy_ela), strict=False)

    out_path = os.path.join("checkpoints", "deepvision_forensics.pt")
    scripted.save(out_path)
    print(f"Exported TorchScript model saved to {out_path}")

if __name__ == "__main__":
    main()
