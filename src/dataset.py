"""
Scene Classification Dataset
=============================
Custom PyTorch Dataset for loading scene images with
ViT-compatible preprocessing.
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor
from collections import Counter

import sys
sys.path.insert(0, '/content/video-scene-understanding')
from src.config import (
    DATA_PROCESSED, VIT_MODEL_NAME, IMAGE_SIZE,
    BATCH_SIZE, NUM_WORKERS, SCENE_CLASSES, SEED
)


class SceneDataset(Dataset):
    """
    Dataset for scene classification.
    Loads images from class-organized folders and applies ViT preprocessing.
    """

    def __init__(self, split='train', processor=None):
        """
        Args:
            split: One of 'train', 'val', 'test'.
            processor: ViTImageProcessor instance (shared across splits).
        """
        self.split = split
        self.split_dir = os.path.join(DATA_PROCESSED, split)
        self.classes = SCENE_CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.processor = processor or ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)

        # Collect all image paths and labels
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image and convert to RGB
        image = Image.open(img_path).convert('RGB')

        # Apply ViT preprocessing (resize, normalize, convert to tensor)
        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dim

        return pixel_values, label

    def get_class_distribution(self):
        """Return class distribution as a Counter."""
        labels = [label for _, label in self.samples]
        dist = Counter(labels)
        return {self.classes[k]: v for k, v in sorted(dist.items())}

    def get_class_weights(self):
        """Compute inverse-frequency class weights for imbalanced data."""
        import torch
        labels = [label for _, label in self.samples]
        counts = Counter(labels)
        total = len(labels)
        weights = []
        for i in range(len(self.classes)):
            w = total / (len(self.classes) * counts.get(i, 1))
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(processor=None):
    """
    Create train, val, test DataLoaders.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, processor)
    """
    if processor is None:
        processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)

    train_dataset = SceneDataset(split='train', processor=processor)
    val_dataset   = SceneDataset(split='val',   processor=processor)
    test_dataset  = SceneDataset(split='test',  processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"✅ DataLoaders created:")
    print(f"   Train : {len(train_dataset):,} images → {len(train_loader)} batches")
    print(f"   Val   : {len(val_dataset):,} images → {len(val_loader)} batches")
    print(f"   Test  : {len(test_dataset):,} images → {len(test_loader)} batches")

    # Print class distribution
    dist = train_dataset.get_class_distribution()
    print(f"\n📊 Train class distribution:")
    for cls, count in dist.items():
        print(f"   {cls:12s} → {count:5d}")

    return train_loader, val_loader, test_loader, processor


if __name__ == "__main__":
    train_loader, val_loader, test_loader, processor = create_dataloaders()

    # Quick sanity check
    batch = next(iter(train_loader))
    pixels, labels = batch
    print(f"\n🔍 Batch shape : {pixels.shape}")
    print(f"   Labels      : {labels[:8].tolist()}")
