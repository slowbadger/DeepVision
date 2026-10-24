# prepare_casia_metadata.py
"""Generate a metadata CSV for the CASIA dataset placed in the project root.
The script scans the CASIA folder (any folder whose name contains "CASIA")
and writes ``data/generated/metadata.csv`` which the training pipeline reads.
"""
import os
import sys
import pandas as pd
from src.dataset import ForensicDataset
from src.config import cfg

def find_casia_root() -> str:
    # Look for a directory named like *CASIA* inside the project root
    root = cfg.base_dir
    candidates = []
    for entry in os.listdir(root):
        if os.path.isdir(os.path.join(root, entry)) and "casia" in entry.lower():
            candidates.append(entry)
    if not candidates:
        raise FileNotFoundError(
            "CASIA folder not found in the project root. "
            "Place the extracted CASIA dataset folder (e.g., 'CASIA2') at the root."
        )
    # If multiple, pick the first one (user can rename to avoid ambiguity)
    return os.path.join(root, candidates[0])

def main() -> None:
    casia_root = find_casia_root()
    print(f"[✓] Found CASIA directory: {casia_root}")

    # Use the dataset class to scan and build a DataFrame
    dataset = ForensicDataset(data_dir=casia_root)
    df = dataset.df
    if df.empty:
        raise RuntimeError("No images were discovered in the CASIA directory.")

    # Ensure output directory exists and write CSV
    out_dir = cfg.generated_data_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "metadata.csv")
    df.to_csv(out_path, index=False)

    print(f"[✓] Metadata CSV written to: {out_path}")
    print(f"    Total samples: {len(df)}")
    print("    Class distribution:")
    print(df["label_name"].value_counts())

if __name__ == "__main__":
    main()
