"""
Dataset Preparation Script
==========================
Re-downloads and splits the Intel Image Classification dataset.
Run this at the start of each new Colab session.
"""

import os
import shutil
import random
import subprocess
import sys

# Add project root to path
sys.path.insert(0, '/content/video-scene-understanding')
from src.config import DATA_RAW, DATA_PROCESSED, SCENE_CLASSES, SEED

random.seed(SEED)


def download_dataset():
    """Download Intel Image Classification from Kaggle."""
    if os.path.exists(os.path.join(DATA_RAW, 'seg_train')):
        print("📂 Dataset already exists, skipping download.")
        return

    print("⬇️  Downloading Intel Image Classification dataset...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "puneet6060/intel-image-classification",
        "-p", DATA_RAW, "--unzip"
    ], check=True)
    print("✅ Download complete!")


def split_dataset(val_ratio=0.1):
    """Split seg_train → train + val, copy seg_test → test."""

    # Detect source paths (handle nested folders)
    seg_train = os.path.join(DATA_RAW, 'seg_train', 'seg_train')
    if not os.path.exists(seg_train):
        seg_train = os.path.join(DATA_RAW, 'seg_train')

    seg_test = os.path.join(DATA_RAW, 'seg_test', 'seg_test')
    if not os.path.exists(seg_test):
        seg_test = os.path.join(DATA_RAW, 'seg_test')

    classes = sorted([d for d in os.listdir(seg_train)
                      if os.path.isdir(os.path.join(seg_train, d))])

    # Create output folders
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(DATA_PROCESSED, split, cls), exist_ok=True)

    # Split train → train + val
    stats = {'train': 0, 'val': 0, 'test': 0}

    for cls in classes:
        cls_path = os.path.join(seg_train, cls)
        images = sorted([f for f in os.listdir(cls_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        random.shuffle(images)

        val_count  = int(len(images) * val_ratio)
        val_imgs   = images[:val_count]
        train_imgs = images[val_count:]

        for img in train_imgs:
            shutil.copy2(os.path.join(cls_path, img),
                         os.path.join(DATA_PROCESSED, 'train', cls, img))
        stats['train'] += len(train_imgs)

        for img in val_imgs:
            shutil.copy2(os.path.join(cls_path, img),
                         os.path.join(DATA_PROCESSED, 'val', cls, img))
        stats['val'] += len(val_imgs)

    # Copy test
    for cls in classes:
        cls_path = os.path.join(seg_test, cls)
        if os.path.exists(cls_path):
            images = [f for f in os.listdir(cls_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in images:
                shutil.copy2(os.path.join(cls_path, img),
                             os.path.join(DATA_PROCESSED, 'test', cls, img))
            stats['test'] += len(images)

    total = sum(stats.values())
    print(f"\n✅ Dataset ready: {total} images")
    for split, count in stats.items():
        print(f"   {split:6s} → {count:5d} ({count/total*100:.1f}%)")


def prepare_sample_videos():
    """Download sample test videos for pipeline demo."""
    videos_dir = '/content/video-scene-understanding/videos'
    os.makedirs(videos_dir, exist_ok=True)

    # Download a Creative Commons sample video
    sample_url = "https://www.pexels.com/download/video/3571264/"
    output_path = os.path.join(videos_dir, "sample_nature.mp4")

    if not os.path.exists(output_path):
        print("⬇️  Downloading sample video...")
        try:
            subprocess.run([
                "wget", "-q", "-O", output_path, sample_url
            ], check=True, timeout=60)
            print(f"✅ Sample video saved: {output_path}")
        except Exception as e:
            print(f"⚠️  Could not download sample video: {e}")
            print("   You can manually place .mp4 files in /content/video-scene-understanding/videos/")
    else:
        print("📂 Sample video already exists.")


if __name__ == "__main__":
    download_dataset()
    split_dataset()
    prepare_sample_videos()
