import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_ela(image: Image.Image, quality: int = 90, scale: float = 15.0) -> Image.Image:
    """
    Computes the Error Level Analysis (ELA) of an image.
    
    ELA works by resaving an image at a known JPEG quality level and computing
    the absolute difference between original and resaved. In pristine images,
    error levels across blocks are relatively uniform. In tampered regions
    (splicing, copy-move, inpainting), error levels diverge noticeably.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resave to in-memory JPEG
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    # Compute absolute difference
    ela_image = ImageChops.difference(image, resaved)

    # Scale the differences for forensic amplification
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale_factor = scale if scale > 0 else (255.0 / max_diff)
    enhancer = ImageEnhance.Brightness(ela_image)
    ela_image = enhancer.enhance(scale_factor)

    return ela_image


def compute_ela_array(img_array: np.ndarray, quality: int = 90, scale: float = 15.0) -> np.ndarray:
    """
    Computes ELA directly on a numpy array (H, W, 3) in uint8 [0, 255].
    Returns (H, W, 3) uint8.
    """
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    ela_img = compute_ela(pil_img, quality=quality, scale=scale)
    return np.array(ela_img)


class SRMConv2d(nn.Module):
    """
    Spatial Rich Model (SRM) High-Pass Convolutional Layer.
    Extracts high-frequency noise residuals (sensor noise PRNU, edge anomalies).
    Contains 3 standardized 5x5 filters.
    """
    def __init__(self, freeze: bool = True):
        super(SRMConv2d, self).__init__()
        
        # 3 SRM kernels of size 5x5
        srm_kernels = np.zeros((3, 1, 5, 5), dtype=np.float32)

        # 1st-order horizontal/vertical residual
        srm_kernels[0, 0] = np.array([
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0],
            [0,  1, -2,  1,  0],
            [0,  0,  0,  0,  0],
            [0,  0,  0,  0,  0]
        ], dtype=np.float32) / 2.0

        # 2nd-order residual
        srm_kernels[1, 0] = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  2, -4,  2,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=np.float32) / 4.0

        # Square 3rd-order residual
        srm_kernels[2, 0] = np.array([
            [-1,  2,  -2,  2, -1],
            [ 2, -6,   8, -6,  2],
            [-2,  8, -12,  8, -2],
            [ 2, -6,   8, -6,  2],
            [-1,  2,  -2,  2, -1]
        ], dtype=np.float32) / 12.0

        self.conv = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv.weight = nn.Parameter(torch.from_numpy(srm_kernels), requires_grad=not freeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, 1, H, W) or (B, 3, H, W)
        Returns high-pass noise residuals: (B, 3, H, W)
        """
        if x.shape[1] == 3:
            # Convert RGB to grayscale luminance Y = 0.299 R + 0.587 G + 0.114 B
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x
        return self.conv(gray)
