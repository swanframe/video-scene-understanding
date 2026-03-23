"""
Video Scene Understanding Pipeline — Configuration
===================================================
All hyperparameters, paths, and settings in one place.
"""

import os
import torch

# ─── Project Paths ───────────────────────────────────────────
PROJECT_ROOT   = "/content/video-scene-understanding"
SRC_DIR        = os.path.join(PROJECT_ROOT, "src")
DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR    = os.path.join(PROJECT_ROOT, "outputs")
KEYFRAMES_DIR  = os.path.join(OUTPUTS_DIR, "keyframes")
RESULTS_DIR    = os.path.join(OUTPUTS_DIR, "results")
PLOTS_DIR      = os.path.join(OUTPUTS_DIR, "plots")
VIDEOS_DIR     = os.path.join(PROJECT_ROOT, "videos")

# ─── Device ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Shot Boundary Detection (PySceneDetect) ─────────────────
SBD_THRESHOLD       = 27.0      # ContentDetector threshold
SBD_MIN_SCENE_LEN   = 15        # Minimum scene length in frames

# ─── Scene Classification — Dataset ──────────────────────────
SCENE_CLASSES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]
NUM_CLASSES = len(SCENE_CLASSES)

TRAIN_SPLIT = 0.8
VAL_SPLIT   = 0.1
TEST_SPLIT  = 0.1

# ─── Scene Classification — ViT Training ─────────────────────
VIT_MODEL_NAME     = "google/vit-base-patch16-224"
IMAGE_SIZE         = 224
BATCH_SIZE         = 32
NUM_WORKERS        = 2

# Phase 1: Frozen encoder
PHASE1_EPOCHS      = 10
PHASE1_LR          = 1e-3

# Phase 2: Full fine-tuning
PHASE2_EPOCHS      = 15
PHASE2_LR          = 1e-5

WEIGHT_DECAY       = 1e-4
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR   = 0.5
EARLY_STOP_PATIENCE = 5

# ─── Scene Captioning (BLIP) ─────────────────────────────────
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CAPTION_MAX_LEN = 50

# ─── Evaluation ──────────────────────────────────────────────
CONF_THRESHOLD = 0.5

# ─── Random Seed ─────────────────────────────────────────────
SEED = 42


def show_config():
    """Print key configuration values."""
    print("=" * 55)
    print("VIDEO SCENE UNDERSTANDING — CONFIGURATION")
    print("=" * 55)
    print(f"  Device          : {DEVICE}")
    print(f"  Project root    : {PROJECT_ROOT}")
    print(f"  Scene classes   : {NUM_CLASSES} → {SCENE_CLASSES}")
    print(f"  ViT model       : {VIT_MODEL_NAME}")
    print(f"  BLIP model      : {BLIP_MODEL_NAME}")
    print(f"  Image size      : {IMAGE_SIZE}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Phase 1 (frozen): {PHASE1_EPOCHS} epochs, lr={PHASE1_LR}")
    print(f"  Phase 2 (full)  : {PHASE2_EPOCHS} epochs, lr={PHASE2_LR}")
    print(f"  SBD threshold   : {SBD_THRESHOLD}")
    print(f"  Seed            : {SEED}")
    print("=" * 55)
