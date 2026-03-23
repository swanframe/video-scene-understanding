"""
Scene Captioning with BLIP
==========================
Uses Salesforce BLIP (Bootstrapping Language-Image Pre-training)
to generate natural language captions for keyframe images.
"""

import os
import sys
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

sys.path.insert(0, '/content/video-scene-understanding')
from src.config import BLIP_MODEL_NAME, CAPTION_MAX_LEN, DEVICE


class SceneCaptioner:
    """
    Generate captions for images using BLIP.
    Supports unconditional and conditional (prompted) captioning.
    """

    def __init__(self, model_name=None, device=None):
        self.device = device or DEVICE
        model_name = model_name or BLIP_MODEL_NAME

        print(f"🔄 Loading BLIP model: {model_name}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ BLIP loaded — {total_params:,} parameters on {self.device}")

    @torch.no_grad()
    def caption_image(self, image, prompt=None, max_length=None,
                      num_beams=4, num_captions=1):
        """
        Generate caption(s) for a single image.

        Args:
            image: PIL Image or file path.
            prompt: Optional text prompt for conditional captioning
                    (e.g., "a photograph of").
            max_length: Max tokens in generated caption.
            num_beams: Beam search width.
            num_captions: Number of captions to generate.

        Returns:
            List of caption strings.
        """
        max_length = max_length or CAPTION_MAX_LEN

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # Prepare inputs
        if prompt:
            inputs = self.processor(image, text=prompt, return_tensors='pt')
        else:
            inputs = self.processor(image, return_tensors='pt')

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_captions,
            early_stopping=True,
        )

        captions = self.processor.batch_decode(output_ids, skip_special_tokens=True)

        # Clean up prompted captions
        if prompt:
            captions = [cap.strip() for cap in captions]

        return captions

    @torch.no_grad()
    def caption_batch(self, image_paths, prompt=None, max_length=None, num_beams=4):
        """
        Generate captions for a batch of images.

        Args:
            image_paths: List of image file paths.
            prompt: Optional prompt for all images.
            max_length: Max tokens per caption.
            num_beams: Beam search width.

        Returns:
            List of caption strings (one per image).
        """
        max_length = max_length or CAPTION_MAX_LEN
        captions = []

        for img_path in tqdm(image_paths, desc="  Captioning"):
            caps = self.caption_image(
                img_path, prompt=prompt,
                max_length=max_length, num_beams=num_beams,
            )
            captions.append(caps[0])

        return captions

    def caption_keyframes(self, metadata, prompts=None):
        """
        Caption all keyframes from shot detection metadata.

        Args:
            metadata: List of keyframe metadata dicts (from shot_detector).
            prompts: Optional list of prompts to try for each image.

        Returns:
            Updated metadata with 'caption' and 'captions_prompted' fields.
        """
        default_prompts = [
            None,                       # Unconditional
            "a photograph of",          # Generic photo prompt
            "this scene shows",         # Scene description prompt
        ]
        prompts = prompts or default_prompts

        print(f"\n📝 Captioning {len(metadata)} keyframes...")
        print(f"   Prompts: {len(prompts)} variants per image\n")

        for meta in tqdm(metadata, desc="  Processing keyframes"):
            img_path = meta['keyframe_path']

            # Unconditional caption (primary)
            primary = self.caption_image(img_path, prompt=None)[0]
            meta['caption'] = primary

            # Multiple prompted captions
            meta['captions_prompted'] = {}
            for prompt in prompts:
                prompt_key = prompt if prompt else 'unconditional'
                caps = self.caption_image(img_path, prompt=prompt)
                meta['captions_prompted'][prompt_key] = caps[0]

        print(f"\n✅ All {len(metadata)} keyframes captioned!")
        return metadata

    def caption_with_multi_beam(self, image, num_captions=5, num_beams=8):
        """
        Generate multiple diverse captions using beam search.

        Args:
            image: PIL Image or file path.
            num_captions: Number of captions to return.
            num_beams: Total beam width (must be >= num_captions).

        Returns:
            List of caption strings.
        """
        return self.caption_image(
            image, prompt=None,
            num_beams=max(num_beams, num_captions),
            num_captions=num_captions,
        )


def save_captioned_metadata(metadata, output_path):
    """Save captioned metadata to JSON."""
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Captioned metadata saved → {output_path}")


if __name__ == "__main__":
    captioner = SceneCaptioner()

    # Test with a sample image
    test_img = "/content/video-scene-understanding/outputs/keyframes/synthetic_test_shot000.jpg"
    if os.path.exists(test_img):
        captions = captioner.caption_image(test_img)
        print(f"\nCaption: {captions[0]}")

        multi = captioner.caption_with_multi_beam(test_img, num_captions=3)
        print("\nMulti-beam captions:")
        for i, c in enumerate(multi):
            print(f"  {i+1}. {c}")
