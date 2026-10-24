import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Config:
    # Random seed
    seed: int = 42

    # Image properties
    img_size: int = 256
    channels: int = 3

    # Classes
    num_classes: int = 5
    class_names: List[str] = None
    binary_classes: List[str] = None

    def __post_init__(self):
        if self.class_names is None:
            self.class_names = [
                "Pristine",
                "Copy-Move",
                "Splicing",
                "Inpainting",
                "AI-Generated"
            ]
        if self.binary_classes is None:
            self.binary_classes = ["Pristine", "Tampered"]

    # ELA (Error Level Analysis) Settings
    ela_quality: int = 90
    ela_scale: float = 15.0

    # Model parameters
    backbone: str = "resnet34"  # torchvision resnet18 / resnet34 / efficientnet_b0
    pretrained: bool = True
    embedding_dim: int = 512
    dropout: float = 0.3

    # Training parameters
    batch_size: int = 16
    epochs: int = 25
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    val_split: float = 0.2
    test_split: float = 0.1

    # Paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = os.path.join(base_dir, "data")
    raw_data_dir: str = os.path.join(data_dir, "raw")
    generated_data_dir: str = os.path.join(data_dir, "generated")
    checkpoint_dir: str = os.path.join(base_dir, "checkpoints")
    results_dir: str = os.path.join(base_dir, "results")
    best_model_path: str = os.path.join(checkpoint_dir, "best_tamper_detector.pth")

# Global default config
cfg = Config()
