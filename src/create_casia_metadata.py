import os
import pandas as pd

def create_casia2_metadata(root_dir, output_dir=None):
    """Generate a metadata.csv for the CASIA2 dataset.

    The CASIA2 folder structure is expected to be:
        <root>/Au   – authentic images
        <root>/Tp   – tampered (copy‑move) images
    The function creates a DataFrame with columns:
        image_path, binary_label, type_label
    binary_label: 0 = authentic, 1 = tampered
    type_label:   0 = authentic, 1 = copy‑move (you can extend this if you split by attack type)
    """
    au_dir = os.path.join(root_dir, "Au")
    tp_dir = os.path.join(root_dir, "Tp")

    records = []
    # Authentic images
    for fname in os.listdir(au_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp')):
            records.append({
                "image_path": os.path.abspath(os.path.join(au_dir, fname)),
                "binary_label": 0,
                "type_label": 0,
            })
    # Tampered images (treated as copy‑move for now)
    for fname in os.listdir(tp_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp')):
            records.append({
                "image_path": os.path.abspath(os.path.join(tp_dir, fname)),
                "binary_label": 1,
                "type_label": 1,
            })

    df = pd.DataFrame(records)
    if output_dir is None:
        output_dir = root_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metadata CSV written to {csv_path} ({len(df)} rows)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create metadata.csv for CASIA2")
    parser.add_argument("--casia_root", type=str, required=True, help="Path to the CASIA2 root folder (contains Au/ and Tp/)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write metadata.csv (defaults to casia_root)")
    args = parser.parse_args()
    create_casia2_metadata(args.casia_root, args.output_dir)
