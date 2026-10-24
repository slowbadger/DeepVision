import argparse
import os
import pandas as pd
from src.config import cfg
from src.dataset import get_dataloaders
from src.train import Trainer
from src.evaluate import evaluate_model
from src.data_generator import SyntheticTamperingGenerator

def main():
    parser = argparse.ArgumentParser(description="Train Dual-Branch Tampering Detector")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--backbone", type=str, default="resnet34", choices=["resnet18", "resnet34"], help="Backbone CNN")
    parser.add_argument("--data-path", type=str, default=None, help="Path to custom dataset or metadata.csv")
    parser.add_argument("--auto-generate", action="store_true", help="Generate synthetic samples if missing")
    parser.add_argument("--samples-per-class", type=int, default=60, help="Samples per class if auto-generating")
    args = parser.parse_args()

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.lr
    cfg.backbone = args.backbone

    meta_path = args.data_path or os.path.join(cfg.generated_data_dir, "metadata.csv")

    if not os.path.exists(meta_path):
        print(f"Dataset metadata not found at {meta_path}.")
        print(f"Auto-generating synthetic dataset ({args.samples_per_class} per class)...")
        gen = SyntheticTamperingGenerator()
        df = gen.generate_dataset(num_samples_per_class=args.samples_per_class)
    else:
        print(f"Loading dataset from: {meta_path}")
        df = pd.read_csv(meta_path)

    print(f"Dataset summary: Total {len(df)} samples")
    print(df["label_name"].value_counts())

    train_loader, val_loader, test_loader = get_dataloaders(
        df,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        seed=cfg.seed
    )

    trainer = Trainer(lr=cfg.learning_rate)
    best_ckpt = trainer.fit(train_loader, val_loader, epochs=cfg.epochs)

    print("\nRunning post-training evaluation on held-out test split...")
    evaluate_model(model_path=best_ckpt, test_loader=test_loader)

if __name__ == "__main__":
    main()
