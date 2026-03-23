"""
ViT Scene Classifier — Two-Phase Training
==========================================
Phase 1: Train only the classification head (encoder frozen)
Phase 2: Fine-tune the entire model with lower learning rate
"""

import os
import sys
import time
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from transformers import ViTForImageClassification

sys.path.insert(0, '/content/video-scene-understanding')
from src.config import (
    VIT_MODEL_NAME, NUM_CLASSES, SCENE_CLASSES, DEVICE, SEED,
    PHASE1_EPOCHS, PHASE1_LR, PHASE2_EPOCHS, PHASE2_LR,
    WEIGHT_DECAY, SCHEDULER_PATIENCE, SCHEDULER_FACTOR,
    EARLY_STOP_PATIENCE, MODELS_DIR
)
from src.dataset import create_dataloaders


def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_model():
    """
    Load ViT-Base pretrained on ImageNet and replace the classification head.
    """
    model = ViTForImageClassification.from_pretrained(
        VIT_MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {i: cls for i, cls in enumerate(SCENE_CLASSES)}
    model.config.label2id = {cls: i for i, cls in enumerate(SCENE_CLASSES)}
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {VIT_MODEL_NAME}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Device: {DEVICE}")

    return model


def freeze_encoder(model):
    """Freeze all ViT encoder layers, keep classifier trainable."""
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n❄️  Encoder FROZEN")
    print(f"   Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")


def unfreeze_encoder(model):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔓 Encoder UNFROZEN — Full fine-tuning")
    print(f"   Trainable: {trainable:,} parameters (100%)")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Training", leave=False)
    for pixels, labels in pbar:
        pixels = pixels.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixels)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on val/test set. Returns (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for pixels, labels in loader:
        pixels = pixels.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=pixels)
        loss = criterion(outputs.logits, labels)

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_phase(model, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs, phase_name, device=DEVICE):
    """
    Run a full training phase with early stopping.

    Returns:
        history dict and best model state_dict.
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [],
    }

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        elapsed = time.time() - epoch_start

        print(f"  Epoch {epoch+1:2d}/{num_epochs} │ "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% │ "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% │ "
              f"LR: {current_lr:.1e} │ {elapsed:.1f}s")

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✅ New best val accuracy: {best_val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n  ⏹️  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    print(f"\n  🏆 Best val accuracy: {best_val_acc*100:.2f}%")
    return history, best_model_state


def run_training():
    """Execute the full two-phase training pipeline."""
    set_seed()

    # --- Data ---
    train_loader, val_loader, test_loader, processor = create_dataloaders()

    # --- Model ---
    model = create_model()

    # --- Class weights for imbalanced data ---
    from src.dataset import SceneDataset
    train_dataset = SceneDataset(split='train', processor=processor)
    class_weights = train_dataset.get_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\n⚖️  Class weights: {class_weights.cpu().numpy().round(3)}")

    # ═══════════════════════════════════════════
    # PHASE 1: Frozen Encoder — Train Head Only
    # ═══════════════════════════════════════════
    freeze_encoder(model)

    optimizer_p1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p1, mode='min',
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
    )

    history_p1, best_state_p1 = train_phase(
        model, train_loader, val_loader, criterion,
        optimizer_p1, scheduler_p1,
        num_epochs=PHASE1_EPOCHS,
        phase_name="PHASE 1: Frozen Encoder — Train Classification Head",
    )

    # Load best Phase 1 weights
    model.load_state_dict(best_state_p1)

    # ═══════════════════════════════════════════
    # PHASE 2: Full Fine-Tuning
    # ═══════════════════════════════════════════
    unfreeze_encoder(model)

    optimizer_p2 = optim.AdamW(
        model.parameters(),
        lr=PHASE2_LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler_p2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_p2, mode='min',
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
    )

    history_p2, best_state_p2 = train_phase(
        model, train_loader, val_loader, criterion,
        optimizer_p2, scheduler_p2,
        num_epochs=PHASE2_EPOCHS,
        phase_name="PHASE 2: Full Fine-Tuning",
    )

    # Load best Phase 2 weights
    model.load_state_dict(best_state_p2)

    # ═══════════════════════════════════════════
    # Save Model & History
    # ═══════════════════════════════════════════
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, 'vit_scene_classifier.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': SCENE_CLASSES,
        'model_name': VIT_MODEL_NAME,
        'num_classes': NUM_CLASSES,
    }, model_path)
    print(f"\n💾 Model saved → {model_path}")

    # Combine histories
    full_history = {
        'phase1': history_p1,
        'phase2': history_p2,
    }
    history_path = os.path.join(MODELS_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(full_history, f, indent=2)
    print(f"📄 History saved → {history_path}")

    return model, full_history, test_loader


if __name__ == "__main__":
    model, history, test_loader = run_training()
