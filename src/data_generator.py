import os
import random
import math
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from typing import Tuple, List, Optional
from src.config import cfg

class SyntheticTamperingGenerator:
    """
    Automated Synthetic Image Tampering Generator.
    Produces high-fidelity:
      - Pristine images (Authentic)
      - Copy-Move forgery (intra-image duplication with geometric transformation)
      - Splicing forgery (inter-image donor patch insertion with boundary blending)
      - Inpainting / Object Removal (patch removal and Telea/Navier-Stokes synthesis)
      - AI-Generated / Deepfake synthesis (generative spectral artifacts & facial/patch synthesis)
    """

    def __init__(self, output_dir: Optional[str] = None, img_size: int = 256):
        self.output_dir = output_dir or cfg.generated_data_dir
        self.img_size = img_size
        self.images_dir = os.path.join(self.output_dir, "images")
        self.masks_dir = os.path.join(self.output_dir, "masks")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)

    def generate_procedural_scene(self) -> np.ndarray:
        """
        Generates diverse procedural base scenes (landscapes, gradients, textures, shapes)
        when external raw photos are not provided.
        """
        h, w = self.img_size, self.img_size
        scene_type = random.choice(["sky_landscape", "geometric_scene", "noisy_texture", "checker_pattern"])

        if scene_type == "sky_landscape":
            # Sky gradient
            top_color = np.array([random.randint(100, 220), random.randint(120, 200), random.randint(180, 255)])
            mid_color = np.array([random.randint(180, 240), random.randint(180, 230), random.randint(190, 240)])
            bot_color = np.array([random.randint(40, 100), random.randint(80, 160), random.randint(30, 90)])
            
            img = np.zeros((h, w, 3), dtype=np.float32)
            for y in range(h // 2):
                alpha = y / (h // 2)
                img[y, :] = (1 - alpha) * top_color + alpha * mid_color
            for y in range(h // 2, h):
                alpha = (y - h // 2) / (h // 2)
                img[y, :] = (1 - alpha) * mid_color + alpha * bot_color

            # Add procedural terrain / hills
            num_hills = random.randint(2, 4)
            for i in range(num_hills):
                base_y = random.randint(int(h * 0.45), int(h * 0.75))
                freq = random.uniform(0.01, 0.04)
                color = [random.randint(30, 90), random.randint(50, 140), random.randint(20, 70)]
                pts = [[0, h], [w, h]]
                for x in range(w, -1, -5):
                    curve_y = int(base_y + 20 * math.sin(x * freq + i))
                    pts.append([x, curve_y])
                cv2.fillPoly(img, [np.array(pts)], color)

            # Add sun or clouds
            sun_x = random.randint(40, w - 40)
            sun_y = random.randint(30, int(h * 0.35))
            cv2.circle(img, (sun_x, sun_y), random.randint(15, 30), (255, 245, 180), -1)

        elif scene_type == "geometric_scene":
            bg_color = [random.randint(200, 245), random.randint(200, 245), random.randint(200, 245)]
            img = np.ones((h, w, 3), dtype=np.float32) * bg_color
            for _ in range(random.randint(5, 10)):
                color = [random.randint(20, 220), random.randint(20, 220), random.randint(20, 220)]
                shape = random.choice(["circle", "rect", "triangle"])
                if shape == "circle":
                    center = (random.randint(20, w - 20), random.randint(20, h - 20))
                    radius = random.randint(15, 45)
                    cv2.circle(img, center, radius, color, -1)
                elif shape == "rect":
                    pt1 = (random.randint(10, w - 60), random.randint(10, h - 60))
                    pt2 = (pt1[0] + random.randint(30, 70), pt1[1] + random.randint(30, 70))
                    cv2.rectangle(img, pt1, pt2, color, -1)
                else:
                    pts = np.array([
                        [random.randint(10, w - 10), random.randint(10, h - 10)],
                        [random.randint(10, w - 10), random.randint(10, h - 10)],
                        [random.randint(10, w - 10), random.randint(10, h - 10)]
                    ], np.int32)
                    cv2.fillPoly(img, [pts], color)

        elif scene_type == "noisy_texture":
            noise = np.random.normal(128, 40, (h // 4, w // 4, 3))
            noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
            img = np.clip(noise, 0, 255)
        else:
            # Checker / pattern
            grid_size = random.choice([16, 32, 64])
            c1 = np.array([random.randint(30, 100), random.randint(30, 100), random.randint(30, 100)])
            c2 = np.array([random.randint(160, 230), random.randint(160, 230), random.randint(160, 230)])
            img = np.zeros((h, w, 3), dtype=np.float32)
            for i in range(0, h, grid_size):
                for j in range(0, w, grid_size):
                    if ((i // grid_size) + (j // grid_size)) % 2 == 0:
                        img[i:i+grid_size, j:j+grid_size] = c1
                    else:
                        img[i:i+grid_size, j:j+grid_size] = c2

        # Add camera sensor noise simulation (PRNU)
        sensor_noise = np.random.normal(0, random.uniform(1.5, 4.0), (h, w, 3))
        img = np.clip(img + sensor_noise, 0, 255).astype(np.uint8)

        # JPEG compression round-trip to simulate realistic camera encoding
        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(88, 96)])
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _generate_random_mask(self, h: int, w: int) -> np.ndarray:
        """Generates an irregular polygon or blob mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        mask_type = random.choice(["polygon", "ellipse", "smooth_blob"])
        
        if mask_type == "ellipse":
            cx = random.randint(int(w * 0.25), int(w * 0.75))
            cy = random.randint(int(h * 0.25), int(h * 0.75))
            axes = (random.randint(18, 55), random.randint(18, 55))
            angle = random.randint(0, 360)
            cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
        elif mask_type == "polygon":
            cx = random.randint(int(w * 0.3), int(w * 0.7))
            cy = random.randint(int(h * 0.3), int(h * 0.7))
            num_pts = random.randint(5, 9)
            angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_pts)])
            radius_mean = random.randint(25, 60)
            pts = []
            for a in angles:
                r = radius_mean * random.uniform(0.7, 1.3)
                px = int(np.clip(cx + r * math.cos(a), 0, w - 1))
                py = int(np.clip(cy + r * math.sin(a), 0, h - 1))
                pts.append([px, py])
            cv2.fillPoly(mask, [np.array(pts)], 255)
        else:
            # Smooth blob
            cx = random.randint(int(w * 0.3), int(w * 0.7))
            cy = random.randint(int(h * 0.3), int(h * 0.7))
            radius = random.randint(25, 55)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
        return mask

    def create_copy_move(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates copy-move forgery: clones an irregular patch from one part of the image
        and pastes it elsewhere, with geometric transformation and alpha boundary blending.
        """
        h, w = img.shape[:2]
        tampered = img.copy()
        tamper_mask = np.zeros((h, w), dtype=np.uint8)

        # Region size
        patch_w = random.randint(int(w * 0.18), int(w * 0.35))
        patch_h = random.randint(int(h * 0.18), int(h * 0.35))

        # Source coordinates
        src_x = random.randint(5, max(6, w - patch_w - 5))
        src_y = random.randint(5, max(6, h - patch_h - 5))
        patch = img[src_y:src_y+patch_h, src_x:src_x+patch_w].copy()

        # Local patch mask (irregular shape)
        local_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
        cv2.ellipse(local_mask, (patch_w // 2, patch_h // 2), 
                    (patch_w // 2 - 2, patch_h // 2 - 2), 
                    random.randint(0, 180), 0, 360, 255, -1)

        # Optional transformation: rotation or scaling
        if random.random() > 0.5:
            angle = random.uniform(-25, 25)
            M = cv2.getRotationMatrix2D((patch_w / 2, patch_h / 2), angle, random.uniform(0.9, 1.1))
            patch = cv2.warpAffine(patch, M, (patch_w, patch_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            local_mask = cv2.warpAffine(local_mask, M, (patch_w, patch_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        # Target non-overlapping location
        max_attempts = 10
        dst_x, dst_y = src_x, src_y
        for _ in range(max_attempts):
            cand_x = random.randint(5, max(6, w - patch_w - 5))
            cand_y = random.randint(5, max(6, h - patch_h - 5))
            if abs(cand_x - src_x) > patch_w // 2 or abs(cand_y - src_y) > patch_h // 2:
                dst_x, dst_y = cand_x, cand_y
                break

        # Boundary feathering
        feather = cv2.GaussianBlur(local_mask, (7, 7), 2.0).astype(np.float32) / 255.0
        feather = np.expand_dims(feather, axis=-1)

        target_roi = tampered[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w].astype(np.float32)
        blended = (1.0 - feather) * target_roi + feather * patch.astype(np.float32)
        tampered[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = np.clip(blended, 0, 255).astype(np.uint8)
        tamper_mask[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = (local_mask > 0).astype(np.uint8) * 255

        return tampered, tamper_mask

    def create_splicing(self, base_img: np.ndarray, donor_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates splicing forgery: pastes an irregular object from donor_img into base_img.
        Simulates illumination/color adjustment differences and distinct noise patterns.
        """
        h, w = base_img.shape[:2]
        tampered = base_img.copy()
        tamper_mask = np.zeros((h, w), dtype=np.uint8)

        patch_w = random.randint(int(w * 0.2), int(w * 0.4))
        patch_h = random.randint(int(h * 0.2), int(h * 0.4))

        donor_x = random.randint(0, max(1, donor_img.shape[1] - patch_w))
        donor_y = random.randint(0, max(1, donor_img.shape[0] - patch_h))
        donor_patch = donor_img[donor_y:donor_y+patch_h, donor_x:donor_x+patch_w].copy()

        # Irregular extraction mask
        local_mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
        num_pts = random.randint(5, 8)
        cx, cy = patch_w // 2, patch_h // 2
        pts = []
        for a in sorted([random.uniform(0, 2 * math.pi) for _ in range(num_pts)]):
            r = min(patch_w, patch_h) * 0.4 * random.uniform(0.7, 1.2)
            pts.append([int(np.clip(cx + r * math.cos(a), 0, patch_w - 1)),
                        int(np.clip(cy + r * math.sin(a), 0, patch_h - 1))])
        cv2.fillPoly(local_mask, [np.array(pts)], 255)

        # Target placement
        dst_x = random.randint(5, max(6, w - patch_w - 5))
        dst_y = random.randint(5, max(6, h - patch_h - 5))

        # Color harmonization difference (typical in splicing)
        alpha_color = random.uniform(0.85, 1.15)
        donor_patch = np.clip(donor_patch.astype(np.float32) * alpha_color, 0, 255).astype(np.uint8)

        feather = cv2.GaussianBlur(local_mask, (7, 7), 2.0).astype(np.float32) / 255.0
        feather = np.expand_dims(feather, axis=-1)

        target_roi = tampered[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w].astype(np.float32)
        blended = (1.0 - feather) * target_roi + feather * donor_patch.astype(np.float32)
        tampered[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = np.clip(blended, 0, 255).astype(np.uint8)
        tamper_mask[dst_y:dst_y+patch_h, dst_x:dst_x+patch_w] = (local_mask > 0).astype(np.uint8) * 255

        return tampered, tamper_mask

    def create_inpainting(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates inpainting / object removal forgery using Telea / Navier-Stokes algorithm.
        This removes natural sensor PRNU and causes localized smoothing/diffusion artifacts.
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Region to remove
        cx = random.randint(int(w * 0.25), int(w * 0.75))
        cy = random.randint(int(h * 0.25), int(h * 0.75))
        radius = random.randint(int(w * 0.08), int(w * 0.18))
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        # Inpaint with Navier-Stokes or Telea
        method = random.choice([cv2.INPAINT_TELEA, cv2.INPAINT_NS])
        # OpenCV inpaint requires BGR
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        inpainted_bgr = cv2.inpaint(bgr, mask, inpaintRadius=5, flags=method)
        tampered = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

        return tampered, mask

    def create_ai_generated(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates AI-generated / deepfake synthesis artifacts:
        - Checkerboard / upsampling spectral anomalies in high frequencies
        - Localized diffusion synthesis with subtle boundary blending
        """
        h, w = img.shape[:2]
        tampered = img.copy()
        tamper_mask = np.zeros((h, w), dtype=np.uint8)

        # Pick synthetic target region
        cx = random.randint(int(w * 0.3), int(w * 0.7))
        cy = random.randint(int(h * 0.3), int(h * 0.7))
        rx = random.randint(int(w * 0.15), int(w * 0.28))
        ry = random.randint(int(h * 0.15), int(h * 0.28))
        cv2.ellipse(tamper_mask, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, 255, -1)

        # Generate generative artifact patch
        y1, y2 = max(0, cy - ry), min(h, cy + ry)
        x1, x2 = max(0, cx - rx), min(w, cx + rx)
        patch = tampered[y1:y2, x1:x2].astype(np.float32)

        # AI synthesis artifacts: subtle bilateral smoothing + high-frequency GAN checkerboard pattern
        patch_bgr = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2BGR)
        smoothed = cv2.bilateralFilter(patch_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        smoothed_rgb = cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB).astype(np.float32)

        # High-frequency spectral artifact (periodic grid pattern from transposed convolutions)
        ph, pw = patch.shape[:2]
        grid = np.zeros((ph, pw, 1), dtype=np.float32)
        grid[::2, ::2, 0] = random.uniform(3.0, 7.0)
        grid[1::2, 1::2, 0] = -random.uniform(3.0, 7.0)

        ai_patch = np.clip(smoothed_rgb + grid, 0, 255)

        # Blend into image
        sub_mask = tamper_mask[y1:y2, x1:x2].astype(np.float32) / 255.0
        sub_mask = cv2.GaussianBlur(sub_mask, (7, 7), 1.5)
        sub_mask = np.expand_dims(sub_mask, axis=-1)

        tampered[y1:y2, x1:x2] = np.clip((1.0 - sub_mask) * patch + sub_mask * ai_patch, 0, 255).astype(np.uint8)
        return tampered, tamper_mask

    def generate_dataset(self, num_samples_per_class: int = 50) -> pd.DataFrame:
        """
        Generates a balanced dataset covering:
        0: Pristine
        1: Copy-Move
        2: Splicing
        3: Inpainting
        4: AI-Generated
        """
        metadata = []
        classes = cfg.class_names
        print(f"Generating synthetic forensic dataset: {num_samples_per_class} per class ({num_samples_per_class * len(classes)} total)...")

        for class_idx, class_name in enumerate(classes):
            print(f"  Generating Class [{class_idx}] {class_name}...")
            for i in range(num_samples_per_class):
                base_img = self.generate_procedural_scene()

                if class_idx == 0:
                    # Pristine
                    tampered_img = base_img.copy()
                    mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                    is_tampered = 0
                elif class_idx == 1:
                    # Copy-Move
                    tampered_img, mask = self.create_copy_move(base_img)
                    is_tampered = 1
                elif class_idx == 2:
                    # Splicing
                    donor_img = self.generate_procedural_scene()
                    tampered_img, mask = self.create_splicing(base_img, donor_img)
                    is_tampered = 1
                elif class_idx == 3:
                    # Inpainting
                    tampered_img, mask = self.create_inpainting(base_img)
                    is_tampered = 1
                elif class_idx == 4:
                    # AI-Generated
                    tampered_img, mask = self.create_ai_generated(base_img)
                    is_tampered = 1

                # Realistic JPEG save step (sensor compression)
                final_bgr = cv2.cvtColor(tampered_img, cv2.COLOR_RGB2BGR)
                img_name = f"{class_name.lower().replace('-', '_')}_{i:04d}.jpg"
                mask_name = f"{class_name.lower().replace('-', '_')}_{i:04d}_mask.png"

                img_path = os.path.join(self.images_dir, img_name)
                mask_path = os.path.join(self.masks_dir, mask_name)

                # Save compressed image & mask
                quality = random.randint(85, 95)
                cv2.imwrite(img_path, final_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
                cv2.imwrite(mask_path, mask)

                metadata.append({
                    "filename": img_name,
                    "mask_filename": mask_name,
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "label": class_idx,
                    "label_name": class_name,
                    "is_tampered": is_tampered
                })

        df = pd.DataFrame(metadata)
        meta_csv_path = os.path.join(self.output_dir, "metadata.csv")
        df.to_csv(meta_csv_path, index=False)
        print(f"Dataset generation complete! Saved {len(df)} samples to {self.output_dir}")
        print(f"Metadata index saved to {meta_csv_path}")
        return df

if __name__ == "__main__":
    generator = SyntheticTamperingGenerator()
    df = generator.generate_dataset(num_samples_per_class=40)
    print("Preview of generated metadata:")
    print(df.head())
