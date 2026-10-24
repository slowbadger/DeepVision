# DeepVision Forensics: Image Tampering Detection & Attribution

## Overview
DeepVision Forensics is a professional, local web application for forensic analysis of digital images. It detects whether an image has been **tampered** and, if so, identifies the **type of tampering** (Copy‑Move, Splicing, Inpainting/Object Removal, AI‑Generated/Deepfake, or authentic). The system also produces explainable heat‑maps and error‑level analysis (ELA) overlays to localize suspicious regions.

The core model is a **dual‑branch neural network** that processes both the raw RGB image and its ELA representation, fusing features via cross‑stream attention. The best checkpoint is stored at:

```
checkpoints/best_tamper_detector.pth
```

## Project Structure
```
DL Project/
├── checkpoints/                # Model checkpoint
├── data/                       # Dataset (raw & generated)
├── src/                        # Core code (data loaders, model, inference)
├── web/                        # Flask front‑end (templates, static assets)
├── server.py                   # Flask API server
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Quickstart Guide

### 1. Install dependencies
```bash
# Activate your virtual environment
.\\venv\\Scripts\\activate
# Install required packages
pip install -r requirements.txt
```

### 2. Export the trained model to TorchScript
```bash
python export_model.py
```

### 3. Run inference on a single image
```bash
python run_inference.py --image <path/to/image.jpg> --device cuda
```

*Use `--device cpu` if a GPU is not available.*

## API Routes
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/sample` | Returns a random sample image (JSON with `image_path`). |
| `POST` | `/api/analyze` | Accepts a JSON payload `{ "image": "<base64‑encoded>" }`. Returns a JSON verdict with confidence scores, tampering type, and heat‑map URLs. |
| `POST` | `/api/simulate` | Accepts a JSON payload `{ "type": "copy_move"|"splicing"|"inpainting"|"ai_gen", "params": {...} }`. Returns a simulated tampered image and its ground‑truth mask. |

> **Note:** The legacy `/api/paper_figures` endpoint has been deprecated and is no longer used.

## Model Details
- **Architecture:** Dual‑branch network (ResNet34 backbones) processing RGB and ELA streams, fused via cross‑stream attention, with three heads for binary authenticity, tampering‑type classification (5‑class), and pixel‑wise heat‑map decoding.
- **Training:** Multi‑task loss (BCE for authenticity, CE for type, Dice for mask). See `src/train.py` for training scripts.
- **Inference:** Run `python run_inference.py --image <path>` for a CLI quick test; the Flask API uses the same inference routine.

## License & Citation
This project is provided for research and educational purposes. Please cite the associated paper if you use the code or pretrained model.

---

For any questions or contributions, open an issue or submit a pull request.
