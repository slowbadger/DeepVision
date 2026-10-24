import argparse
from src.data_generator import SyntheticTamperingGenerator
from src.config import cfg

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Forensic Tampering Dataset")
    parser.add_argument("--samples", type=int, default=60, help="Number of samples per class (default: 60)")
    parser.add_argument("--output", type=str, default=cfg.generated_data_dir, help="Output directory")
    parser.add_argument("--img-size", type=int, default=cfg.img_size, help="Image resolution (default: 256)")
    args = parser.parse_args()

    print(f"Starting synthetic dataset generation: {args.samples} per class...")
    generator = SyntheticTamperingGenerator(output_dir=args.output, img_size=args.img_size)
    df = generator.generate_dataset(num_samples_per_class=args.samples)
    print(f"Generated {len(df)} samples across {cfg.num_classes} classes.")
    print(df["label_name"].value_counts())

if __name__ == "__main__":
    main()
