import io
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple, Dict, Any, Optional

from src.config import cfg
from src.forensic_features import compute_ela

class ForensicDataset(Dataset):
    """
    Forensic PyTorch Dataset supporting:
      1. Synthetically generated multi-tampering dataset
      2. CASIA v1.0 / CASIA v2.0 / CoMoFoD benchmarks
      3. Custom folder hierarchies
    Returns:
      - rgb_tensor: (3, H, W) normalized
      - ela_tensor: (3, H, W) normalized
      - binary_label: (1,) float (0: pristine, 1: tampered)
      - type_label: () int64 (0..4)
      - mask_tensor: (1, H, W) binary tampering mask
    """

    def __init__(
        self,
        metadata_df: Optional[pd.DataFrame] = None,
        data_dir: Optional[str] = None,
        img_size: int = 256,
        is_training: bool = True,
        ela_quality: int = 90,
        ela_scale: float = 15.0,
        simulate_jpeg: bool = False,
        pre_jpeg_quality: int = 95,
        random_jpeg_quality: bool = False
    ):
        self.img_size = img_size
        self.is_training = is_training
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale
        self.simulate_jpeg = simulate_jpeg
        self.pre_jpeg_quality = pre_jpeg_quality
        self.random_jpeg_quality = random_jpeg_quality

        if metadata_df is not None:
            self.df = metadata_df.reset_index(drop=True)
        elif data_dir is not None:
            self.df = self._scan_directory(data_dir)
        else:
            meta_path = os.path.join(cfg.generated_data_dir, "metadata.csv")
            if os.path.exists(meta_path):
                self.df = pd.read_csv(meta_path)
            else:
                self.df = pd.DataFrame(columns=["image_path", "mask_path", "label", "is_tampered"])

        # RGB Transforms (Standard ImageNet normalization)
        if self.is_training:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # ELA Transforms (Normalized to [0, 1])
        self.ela_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

    def _scan_directory(self, data_dir: str) -> pd.DataFrame:
        """
        Scans a directory for images. Supports:
        - CASIA v2.0 structures (Au / Tp folders, Sp_ / CM_ prefixes)
        - Subfolders named by class ('Pristine', 'Copy-Move', 'Splicing', 'Inpainting', 'AI-Generated')
        """
        rows = []
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp")
        all_files = []
        for ext in valid_exts:
            all_files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

        for file_path in all_files:
            lower = file_path.lower()
            if "_mask" in lower:
                continue

            # Check CASIA style
            if "casia" in lower or "au" in lower or "tp" in lower:
                fname = os.path.basename(file_path)
                if fname.startswith("Au_") or "\\au\\" in lower or "/au/" in lower:
                    rows.append({
                        "image_path": file_path, "mask_path": "", "label": 0, "label_name": "Pristine", "is_tampered": 0
                    })
                elif "tp_c" in lower or "cm_" in lower:
                    rows.append({
                        "image_path": file_path, "mask_path": "", "label": 1, "label_name": "Copy-Move", "is_tampered": 1
                    })
                elif "tp_s" in lower or "sp_" in lower:
                    rows.append({
                        "image_path": file_path, "mask_path": "", "label": 2, "label_name": "Splicing", "is_tampered": 1
                    })
                else:
                    rows.append({
                        "image_path": file_path, "mask_path": "", "label": 1, "label_name": "Copy-Move", "is_tampered": 1
                    })
            else:
                # Class by folder name
                assigned = False
                for c_idx, c_name in enumerate(cfg.class_names):
                    c_slug = c_name.lower().replace("-", "").replace("_", "")
                    if c_slug in lower:
                        rows.append({
                            "image_path": file_path,
                            "mask_path": "",
                            "label": c_idx,
                            "label_name": c_name,
                            "is_tampered": 0 if c_idx == 0 else 1
                        })
                        assigned = True
                        break
                if not assigned:
                    # Default: assume pristine if authentic/pristine in path, else tampered
                    if "pristine" in lower or "authentic" in lower:
                        rows.append({"image_path": file_path, "mask_path": "", "label": 0, "label_name": "Pristine", "is_tampered": 0})
                    else:
                        rows.append({"image_path": file_path, "mask_path": "", "label": 1, "label_name": "Copy-Move", "is_tampered": 1})

        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        # Load RGB image
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback black image if corrupted
            pil_img = Image.new("RGB", (self.img_size, self.img_size), color=(0, 0, 0))

        # Optional JPEG pre-compression simulation (for lossless formats like TIFF / PNG)
        if self.simulate_jpeg:
            try:
                q = int(np.random.randint(75, 98)) if self.random_jpeg_quality else int(self.pre_jpeg_quality)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=q)
                buf.seek(0)
                pil_img = Image.open(buf).convert("RGB")
            except Exception:
                pass

        # Compute ELA stream
        ela_img = compute_ela(pil_img, quality=self.ela_quality, scale=self.ela_scale)

        rgb_tensor = self.rgb_transform(pil_img)
        ela_tensor = self.ela_transform(ela_img)

        # Labels - support both old and new CSV formats
        if "label" in row:
            label_type = int(row["label"])
        elif "type_label" in row:
            label_type = int(row["type_label"])
        else:
            raise KeyError("Neither 'label' nor 'type_label' column found in metadata.")

        if "is_tampered" in row:
            is_tampered = float(row["is_tampered"])
        elif "binary_label" in row:
            is_tampered = float(row["binary_label"])
        else:
            raise KeyError("Neither 'is_tampered' nor 'binary_label' column found in metadata.")

        # Load Mask if exists
        mask_path = row.get("mask_path", "")
        if mask_path and os.path.exists(str(mask_path)):
            mask_img = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size))
            mask_tensor = transforms.ToTensor()(mask_img)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        return {
            "rgb": rgb_tensor,
            "ela": ela_tensor,
            "type_label": torch.tensor(label_type, dtype=torch.long),
            "binary_label": torch.tensor(is_tampered, dtype=torch.float32),
            "mask": mask_tensor,
            "image_path": img_path
        }


def get_dataloaders(
    metadata_df: pd.DataFrame,
    batch_size: int = 16,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    simulate_jpeg: bool = False,
    pre_jpeg_quality: int = 95,
    random_jpeg_quality: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Splits metadata into train, val, test loaders."""
    np.random.seed(seed)
    n = len(metadata_df)
    indices = np.random.permutation(n)

    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_df = metadata_df.iloc[train_idx].reset_index(drop=True)
    val_df = metadata_df.iloc[val_idx].reset_index(drop=True)
    test_df = metadata_df.iloc[test_idx].reset_index(drop=True)

    train_ds = ForensicDataset(
        train_df, is_training=True,
        simulate_jpeg=simulate_jpeg,
        pre_jpeg_quality=pre_jpeg_quality,
        random_jpeg_quality=random_jpeg_quality
    )
    val_ds = ForensicDataset(
        val_df, is_training=False,
        simulate_jpeg=simulate_jpeg,
        pre_jpeg_quality=pre_jpeg_quality,
        random_jpeg_quality=False
    )
    test_ds = ForensicDataset(
        test_df, is_training=False,
        simulate_jpeg=simulate_jpeg,
        pre_jpeg_quality=pre_jpeg_quality,
        random_jpeg_quality=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
