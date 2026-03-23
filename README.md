# 🎬 Video Scene Understanding Pipeline

> End-to-end pipeline that analyzes videos by detecting shot boundaries, classifying scene types, and generating natural language captions for each scene.

**Video in → Structured scene-by-scene summary out**

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Evaluation](#evaluation)
- [Technologies](#technologies)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project implements a **multi-stage video scene understanding pipeline** that combines three AI models to automatically analyze video content:

| Stage | Task | Model/Tool | Purpose |
|-------|------|------------|---------|
| 1 | Shot Boundary Detection | PySceneDetect (ContentDetector) | Detect scene transitions in video |
| 2 | Keyframe Extraction | OpenCV | Extract representative frame per shot |
| 3 | Scene Classification | ViT-Base (fine-tuned) | Classify scene type (6 categories) |
| 4 | Scene Captioning | BLIP-Base (pretrained) | Generate natural language descriptions |

The pipeline produces a **structured JSON summary** and a **visual HTML report** with timeline, thumbnails, labels, and captions for each detected scene.

---

## Pipeline Architecture
```
┌─────────────────┐
│   Input Video    │
└────────┬────────┘
         ▼
┌─────────────────────────────────────┐
│  Stage 1: Shot Boundary Detection   │
│  PySceneDetect · ContentDetector    │
│  Threshold: 27.0 · 576.8 FPS       │
└────────┬────────────────────────────┘
         ▼
┌─────────────────────────────────────┐
│  Stage 2: Keyframe Extraction       │
│  OpenCV · Middle/First/Best frame   │
└────────┬────────────────────────────┘
         ▼
┌─────────────────────────────────────┐
│  Stage 3: Scene Classification      │
│  ViT-Base · Fine-tuned · 95.00%    │
│  6 classes · Two-phase training     │
└────────┬────────────────────────────┘
         ▼
┌─────────────────────────────────────┐
│  Stage 4: Scene Captioning          │
│  BLIP-Base · Beam Search            │
│  80% keyword match · Zero-shot      │
└────────┬────────────────────────────┘
         ▼
┌─────────────────────────────────────┐
│  Output: JSON + HTML Report         │
│  Timeline · Thumbnails · Captions   │
└─────────────────────────────────────┘
```

---

## Key Results

### Scene Classification (ViT-Base)

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **95.00%** |
| Macro F1-Score | 0.9507 |
| Weighted F1-Score | 0.9499 |
| Macro Precision | 0.9506 |
| Macro Recall | 0.9510 |

#### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Buildings | 94.7% | 93.8% | 94.3% |
| Forest | 98.7% | 99.6% | 99.2% |
| Glacier | 93.6% | 90.4% | 92.0% |
| Mountain | 91.5% | 92.6% | 92.0% |
| Sea | 96.9% | 98.6% | 97.8% |
| Street | 94.9% | 95.6% | 95.2% |

### Scene Captioning (BLIP-Base)

| Class | Keyword Match Rate |
|-------|-------------------|
| Forest | 100% |
| Street | 95% |
| Sea | 85% |
| Mountain | 75% |
| Buildings | 65% |
| Glacier | 60% |
| **Overall** | **80%** |

### Pipeline Speed (Tesla T4 GPU)

| Stage | Time | Share |
|-------|------|-------|
| Shot Detection | 0.936s | 27.5% |
| Keyframe Extraction | 0.103s | 3.0% |
| Scene Classification | 0.143s | 4.2% |
| Scene Captioning | 2.219s | 65.2% |
| **Total** | **3.40s** | 100% |

Shot detection speed: **576.8 FPS**

---

## Project Structure
```
video-scene-understanding/
├── src/
│   ├── config.py           # All hyperparameters and paths
│   ├── prepare_data.py     # Dataset download and splitting
│   ├── dataset.py          # PyTorch Dataset and DataLoaders
│   ├── train.py            # Two-phase ViT training pipeline
│   ├── classifier.py       # ViT inference wrapper
│   ├── shot_detector.py    # Shot boundary detection + keyframe extraction
│   ├── captioner.py        # BLIP captioning module
│   ├── pipeline.py         # End-to-end pipeline orchestrator
│   └── evaluate.py         # Evaluation metrics and visualizations
├── models/
│   ├── vit_scene_classifier.pth   # Fine-tuned ViT weights
│   └── training_history.json      # Training logs
├── outputs/
│   ├── plots/              # All visualizations
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── per_class_metrics.png
│   │   ├── confidence_analysis.png
│   │   ├── error_analysis.png
│   │   ├── misclassified_samples.png
│   │   ├── speed_benchmark.png
│   │   ├── blip_captions_real.png
│   │   └── caption_quality.png
│   └── results/            # Pipeline outputs
│       └── evaluation_results.json
├── videos/                 # Test videos
└── README.md
```

---

## Installation

This project is designed to run on **Google Colab** with GPU runtime.

### 1. Clone the repository
```bash
git clone https://github.com/swanframe/video-scene-understanding.git
cd video-scene-understanding
```

### 2. Install dependencies
```bash
pip install scenedetect[opencv] transformers datasets accelerate evaluate timm scikit-learn matplotlib seaborn tqdm
```

### 3. Prepare dataset (requires Kaggle API)
```python
python src/prepare_data.py
```

---

## Usage

### Run the full pipeline on a video
```python
from src.pipeline import VideoScenePipeline

# Initialize (loads all models)
pipeline = VideoScenePipeline()

# Process a video
results = pipeline.process_video("path/to/your/video.mp4")

# Results include:
# - Shot boundaries with timestamps
# - Scene classification labels and confidence scores
# - Natural language captions for each scene
# - JSON results file + visual HTML report
```

### Use individual modules
```python
# Shot Detection
from src.shot_detector import detect_shots, extract_keyframes
scene_list = detect_shots("video.mp4", threshold=27.0)
keyframes = extract_keyframes("video.mp4", scene_list, method='middle')

# Scene Classification
from src.classifier import SceneClassifier
classifier = SceneClassifier()
result = classifier.classify_image("image.jpg")
print(f"{result['label']} ({result['confidence']:.1%})")

# Scene Captioning
from src.captioner import SceneCaptioner
captioner = SceneCaptioner()
captions = captioner.caption_image("image.jpg")
print(captions[0])
```

### Train the scene classifier
```python
from src.train import run_training
model, history, test_loader = run_training()
```

---

## Modules

### `shot_detector.py`
- **ContentDetector** with configurable threshold (default: 27.0)
- Three keyframe extraction methods: `middle`, `first`, `best` (sharpest via Laplacian variance)
- JSON metadata output with timestamps and frame indices
- Threshold sensitivity analysis

### `train.py`
- **Two-phase training strategy**:
  - Phase 1: Frozen ViT encoder, train classification head only (0.01% params)
  - Phase 2: Full fine-tuning with lower learning rate
- AdamW optimizer + ReduceLROnPlateau + Early Stopping
- Class-weighted CrossEntropyLoss for balanced training

### `classifier.py`
- Inference wrapper for the fine-tuned ViT model
- Top-k predictions with confidence scores
- Single image and batch classification support

### `captioner.py`
- BLIP-Base with beam search decoding (num_beams=4)
- Unconditional and conditional (prompted) captioning
- Multi-beam diverse caption generation

### `pipeline.py`
- Orchestrates all modules into a single `process_video()` call
- Generates structured JSON results
- Produces visual HTML reports with timeline, scene cards, and confidence bars

### `evaluate.py`
- Per-class precision, recall, F1 analysis
- Confidence distribution and accuracy-coverage curves
- Error analysis with misclassification pair ranking
- Pipeline speed benchmarking

---

## Evaluation

### Training Curves
Two-phase training showing frozen encoder (Phase 1) → full fine-tuning (Phase 2):
- Phase 1 best validation accuracy: **94.44%**
- Phase 2 best validation accuracy: **95.01%**

### Error Analysis
Top misclassification pairs reflect visual similarity:
- **glacier ↔ mountain** (72 errors) — shared snowy/rocky features
- **buildings ↔ street** (46 errors) — urban scenes with overlapping elements

### Confidence Analysis
- Model is well-calibrated with most predictions at high confidence
- At 0.95 confidence threshold: ~99% accuracy with 93% coverage

---

## Technologies

| Component | Technology | Version |
|-----------|-----------|---------|
| Deep Learning Framework | PyTorch | 2.10.0 |
| Vision Transformer | HuggingFace Transformers (ViT-Base) | 5.0.0 |
| Image Captioning | BLIP (Salesforce) | via Transformers |
| Shot Detection | PySceneDetect | 0.6.7 |
| Computer Vision | OpenCV | 4.13.0 |
| Image Processing | Timm | 1.0.25 |
| Dataset | Intel Image Classification (Kaggle) | 17,034 images |
| Training | Two-Phase (Frozen + Fine-Tune) | AdamW + Scheduler |
| Environment | Google Colab | T4 GPU |

---

## Acknowledgments

- **Intel Image Classification** dataset via [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **ViT-Base** pretrained model from [Google](https://huggingface.co/google/vit-base-patch16-224)
- **BLIP** captioning model from [Salesforce](https://huggingface.co/Salesforce/blip-image-captioning-base)
- **PySceneDetect** library by [Brandon Castellano](https://github.com/Breakthrough/PySceneDetect)
