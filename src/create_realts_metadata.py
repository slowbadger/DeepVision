# Helper to generate metadata.csv for the REAL-TS dataset
"""
REAL-TS folder structure:
    REAL-TS/
    ├── Canon_60D/
    │   ├── pristine/            (authentic TIF images)
    │   ├── tampered-realistic/  (tampered TIF images)
    │   └── ground-truth/        (binary masks – skipped)
    ├── Nikon_D90/
    │   └── ...
    ├── Nikon_D7000/
    │   └── ...
    ├── Sony_A57/
    │   └── ...
    ├── camera_models/
    └── thumbnails/

Usage:
    python src/create_realts_metadata.py --root_dir "C:/Users/agvar/OneDrive/Desktop/DL Project/REAL-TS"

Output:
    metadata.csv with columns: image_path, binary_label, type_label
"""

import os
import argparse
import pandas as pd

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Camera-model folders to scan (skip thumbnails, camera_models, etc.)
CAMERA_FOLDERS = ["Canon_60D", "Nikon_D90", "Nikon_D7000", "Sony_A57"]


def create_realts_metadata(root_dir: str, output_dir: str = None):
    records = []

    for camera in CAMERA_FOLDERS:
        camera_path = os.path.join(root_dir, camera)
        if not os.path.isdir(camera_path):
            print(f"  [SKIP] Camera folder not found: {camera_path}")
            continue

        # --- Pristine images (binary_label=0, type_label=0) ---
        pristine_dir = os.path.join(camera_path, "pristine")
        if os.path.isdir(pristine_dir):
            for fname in os.listdir(pristine_dir):
                if fname.lower().endswith(VALID_EXTS):
                    records.append({
                        "image_path": os.path.abspath(os.path.join(pristine_dir, fname)),
                        "binary_label": 0,
                        "type_label": 0,
                    })

        # --- Tampered images (binary_label=1, type_label=1) ---
        tampered_dir = os.path.join(camera_path, "tampered-realistic")
        if os.path.isdir(tampered_dir):
            for fname in os.listdir(tampered_dir):
                if fname.lower().endswith(VALID_EXTS):
                    records.append({
                        "image_path": os.path.abspath(os.path.join(tampered_dir, fname)),
                        "binary_label": 1,
                        "type_label": 1,  # generic "tampered" class
                    })

        # Ground-truth masks are intentionally skipped (they are not input images)

        print(f"  [OK] {camera}: "
              f"{len([r for r in records if camera in r['image_path'] and r['binary_label'] == 0])} pristine, "
              f"{len([r for r in records if camera in r['image_path'] and r['binary_label'] == 1])} tampered")

    df = pd.DataFrame(records)
    if df.empty:
        print("ERROR: No images found. Check the folder structure.")
        return

    if output_dir is None:
        output_dir = root_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nMetadata CSV written to {csv_path}")
    print(f"  Total rows : {len(df)}")
    print(f"  Pristine   : {len(df[df['binary_label'] == 0])}")
    print(f"  Tampered   : {len(df[df['binary_label'] == 1])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create metadata.csv for the REAL-TS dataset")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of the REAL-TS dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write metadata.csv (defaults to root_dir)")
    args = parser.parse_args()
    create_realts_metadata(args.root_dir, args.output_dir)
