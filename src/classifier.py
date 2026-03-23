"""
Scene Classifier — Inference Module
====================================
Loads the fine-tuned ViT model and classifies keyframe images.
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm

sys.path.insert(0, '/content/video-scene-understanding')
from src.config import (
    VIT_MODEL_NAME, NUM_CLASSES, SCENE_CLASSES,
    DEVICE, MODELS_DIR, IMAGE_SIZE
)


class SceneClassifier:
    """
    Classify scene images using the fine-tuned ViT model.
    """

    def __init__(self, model_path=None, device=None):
        self.device = device or DEVICE
        model_path = model_path or os.path.join(MODELS_DIR, 'vit_scene_classifier.pth')

        print(f"🔄 Loading ViT scene classifier...")

        # Load processor
        self.processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)

        # Load model architecture
        self.model = ViTForImageClassification.from_pretrained(
            VIT_MODEL_NAME,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )

        # Load fine-tuned weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.class_names = checkpoint.get('class_names', SCENE_CLASSES)
        print(f"✅ Classifier loaded — {len(self.class_names)} classes on {self.device}")

    @torch.no_grad()
    def classify_image(self, image, top_k=3):
        """
        Classify a single image.

        Args:
            image: PIL Image or file path.
            top_k: Number of top predictions to return.

        Returns:
            Dict with 'label', 'confidence', and 'top_k' predictions.
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        inputs = self.processor(images=image, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self.device)

        outputs = self.model(pixel_values=pixel_values)
        probs = F.softmax(outputs.logits, dim=-1).squeeze()

        # Top-k predictions
        top_probs, top_indices = probs.topk(min(top_k, len(self.class_names)))

        top_predictions = []
        for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
            top_predictions.append({
                'label': self.class_names[idx.item()],
                'confidence': round(prob.item(), 4),
            })

        return {
            'label': top_predictions[0]['label'],
            'confidence': top_predictions[0]['confidence'],
            'top_k': top_predictions,
        }

    @torch.no_grad()
    def classify_batch(self, image_paths, top_k=3):
        """
        Classify a batch of images.

        Args:
            image_paths: List of image file paths.
            top_k: Number of top predictions per image.

        Returns:
            List of classification result dicts.
        """
        results = []
        for img_path in tqdm(image_paths, desc="  Classifying"):
            result = self.classify_image(img_path, top_k=top_k)
            results.append(result)
        return results

    def classify_keyframes(self, metadata):
        """
        Classify all keyframes from shot detection metadata.

        Args:
            metadata: List of keyframe metadata dicts.

        Returns:
            Updated metadata with 'scene_label' and 'scene_confidence' fields.
        """
        print(f"\n🏷️  Classifying {len(metadata)} keyframes...")

        for meta in tqdm(metadata, desc="  Classifying scenes"):
            result = self.classify_image(meta['keyframe_path'])
            meta['scene_label'] = result['label']
            meta['scene_confidence'] = result['confidence']
            meta['scene_top_k'] = result['top_k']

        print(f"✅ All {len(metadata)} keyframes classified!")
        return metadata


if __name__ == "__main__":
    classifier = SceneClassifier()

    test_img = "/content/video-scene-understanding/outputs/keyframes/synthetic_test_shot000.jpg"
    if os.path.exists(test_img):
        result = classifier.classify_image(test_img)
        print(f"\nLabel: {result['label']} ({result['confidence']:.2%})")
        print("Top-3:")
        for pred in result['top_k']:
            print(f"  {pred['label']:12s} → {pred['confidence']:.2%}")
