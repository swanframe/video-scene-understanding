"""
Evaluation Module
=================
Comprehensive evaluation for all pipeline components:
- ViT scene classification (per-class metrics, error analysis)
- BLIP caption quality analysis
- Pipeline speed benchmarks
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, '/content/video-scene-understanding')
from src.config import (
    DEVICE, SCENE_CLASSES, NUM_CLASSES, VIT_MODEL_NAME,
    MODELS_DIR, PLOTS_DIR, RESULTS_DIR, DATA_PROCESSED
)


def evaluate_classifier(model, test_loader, device=DEVICE):
    """
    Run full evaluation on the test set.

    Returns:
        Dict with all metrics, predictions, and labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for pixels, labels in tqdm(test_loader, desc="  Evaluating"):
            pixels = pixels.to(device)
            outputs = model(pixel_values=pixels)
            probs = torch.softmax(outputs.logits, dim=-1)
            _, predicted = outputs.logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        'macro_precision': precision_score(all_labels, all_preds, average='macro'),
        'macro_recall': recall_score(all_labels, all_preds, average='macro'),
    }

    # Per-class metrics
    per_class = {}
    report = classification_report(all_labels, all_preds,
                                    target_names=SCENE_CLASSES,
                                    output_dict=True)
    for cls in SCENE_CLASSES:
        per_class[cls] = {
            'precision': report[cls]['precision'],
            'recall': report[cls]['recall'],
            'f1': report[cls]['f1-score'],
            'support': report[cls]['support'],
        }
    metrics['per_class'] = per_class

    return {
        'metrics': metrics,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }


def plot_per_class_metrics(eval_results, save_dir=PLOTS_DIR):
    """Plot per-class precision, recall, F1 as grouped bar chart."""
    per_class = eval_results['metrics']['per_class']
    classes = list(per_class.keys())

    precision = [per_class[c]['precision'] * 100 for c in classes]
    recall = [per_class[c]['recall'] * 100 for c in classes]
    f1 = [per_class[c]['f1'] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    ax.set_ylabel('Score (%)')
    ax.set_title('Per-Class Classification Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(80, 102)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_metrics.png'), dpi=150)
    plt.show()
    print("✅ Per-class metrics plot saved!")


def plot_confidence_distribution(eval_results, save_dir=PLOTS_DIR):
    """Plot confidence distribution for correct vs incorrect predictions."""
    preds = eval_results['predictions']
    labels = eval_results['labels']
    probs = eval_results['probabilities']

    # Get max confidence per sample
    max_confs = probs.max(axis=1)
    correct_mask = preds == labels

    correct_confs = max_confs[correct_mask]
    incorrect_confs = max_confs[~correct_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    bins = np.linspace(0, 1, 30)
    ax1.hist(correct_confs, bins=bins, alpha=0.7, label=f'Correct (n={len(correct_confs)})',
             color='#2ecc71', edgecolor='white')
    ax1.hist(incorrect_confs, bins=bins, alpha=0.7, label=f'Incorrect (n={len(incorrect_confs)})',
             color='#e74c3c', edgecolor='white')
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('Confidence Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy at different confidence thresholds
    thresholds = np.arange(0.3, 1.0, 0.05)
    accs = []
    coverages = []
    for t in thresholds:
        mask = max_confs >= t
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).mean() * 100
            coverage = mask.mean() * 100
        else:
            acc = 0
            coverage = 0
        accs.append(acc)
        coverages.append(coverage)

    ax2_twin = ax2.twinx()
    line1, = ax2.plot(thresholds, accs, 'b-o', markersize=4, label='Accuracy')
    line2, = ax2_twin.plot(thresholds, coverages, 'r--s', markersize=4, label='Coverage')
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Accuracy (%)', color='blue')
    ax2_twin.set_ylabel('Coverage (%)', color='red')
    ax2.set_title('Accuracy vs Coverage at Different Thresholds')
    ax2.grid(alpha=0.3)
    ax2.legend(handles=[line1, line2], loc='center left')

    plt.suptitle('Confidence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'), dpi=150)
    plt.show()
    print("✅ Confidence analysis plot saved!")


def plot_error_analysis(eval_results, save_dir=PLOTS_DIR):
    """Analyze and visualize misclassified samples."""
    preds = eval_results['predictions']
    labels = eval_results['labels']
    probs = eval_results['probabilities']

    incorrect_mask = preds != labels
    incorrect_idx = np.where(incorrect_mask)[0]

    # Count error types (true_class → predicted_class)
    error_pairs = Counter()
    for idx in incorrect_idx:
        true_cls = SCENE_CLASSES[labels[idx]]
        pred_cls = SCENE_CLASSES[preds[idx]]
        error_pairs[(true_cls, pred_cls)] += 1

    # Top error pairs
    top_errors = error_pairs.most_common(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Error pair bar chart
    if top_errors:
        pair_labels = [f"{t}→{p}" for (t, p), _ in top_errors]
        pair_counts = [c for _, c in top_errors]
        colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(pair_labels)))
        ax1.barh(pair_labels[::-1], pair_counts[::-1], color=colors[::-1])
        ax1.set_xlabel('Number of Errors')
        ax1.set_title('Top Misclassification Pairs')
        for i, (label, count) in enumerate(zip(pair_labels[::-1], pair_counts[::-1])):
            ax1.text(count + 0.3, i, str(count), va='center', fontsize=10)

    # Per-class error rate
    error_rates = []
    for i, cls in enumerate(SCENE_CLASSES):
        cls_mask = labels == i
        cls_errors = (preds[cls_mask] != labels[cls_mask]).sum()
        cls_total = cls_mask.sum()
        error_rates.append(cls_errors / cls_total * 100 if cls_total > 0 else 0)

    colors2 = ['#e74c3c' if r > 10 else '#f39c12' if r > 5 else '#2ecc71' for r in error_rates]
    ax2.bar(SCENE_CLASSES, error_rates, color=colors2)
    ax2.set_ylabel('Error Rate (%)')
    ax2.set_title('Per-Class Error Rate')
    ax2.grid(axis='y', alpha=0.3)
    for i, r in enumerate(error_rates):
        ax2.text(i, r + 0.3, f'{r:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=150)
    plt.show()
    print("✅ Error analysis plot saved!")

    return {
        'total_errors': int(incorrect_mask.sum()),
        'total_samples': len(labels),
        'error_rate': float(incorrect_mask.mean() * 100),
        'top_error_pairs': [(f"{t}→{p}", c) for (t, p), c in top_errors],
    }


def benchmark_pipeline_speed(pipeline, video_path, num_runs=3):
    """
    Benchmark pipeline processing speed.

    Returns:
        Dict with timing for each stage.
    """
    from src.shot_detector import detect_shots, extract_keyframes
    from src.config import KEYFRAMES_DIR
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"\n⏱️  Benchmarking pipeline ({num_runs} runs)...")
    print(f"   Video: {os.path.basename(video_path)} — {total_frames} frames, {duration:.1f}s")

    timings = {
        'shot_detection': [],
        'keyframe_extraction': [],
        'classification': [],
        'captioning': [],
        'total': [],
    }

    for run in range(num_runs):
        print(f"\n  Run {run + 1}/{num_runs}...")

        # Shot detection
        t0 = time.time()
        scene_list = detect_shots(video_path)
        timings['shot_detection'].append(time.time() - t0)

        # Keyframe extraction
        t0 = time.time()
        tmp_dir = f'/tmp/bench_keyframes_{run}'
        os.makedirs(tmp_dir, exist_ok=True)
        metadata = extract_keyframes(video_path, scene_list,
                                      output_dir=tmp_dir, method='middle')
        timings['keyframe_extraction'].append(time.time() - t0)

        # Classification
        t0 = time.time()
        metadata = pipeline.classifier.classify_keyframes(metadata)
        timings['classification'].append(time.time() - t0)

        # Captioning
        t0 = time.time()
        for meta in metadata:
            pipeline.captioner.caption_image(meta['keyframe_path'])
        timings['captioning'].append(time.time() - t0)

        total = sum(timings[k][-1] for k in timings if k != 'total')
        timings['total'].append(total)

    # Compute averages
    results = {}
    print(f"\n{'='*60}")
    print(f"  SPEED BENCHMARK RESULTS (avg of {num_runs} runs)")
    print(f"{'='*60}")

    for stage, times in timings.items():
        avg = np.mean(times)
        std = np.std(times)
        results[stage] = {'mean': round(avg, 3), 'std': round(std, 3)}
        print(f"  {stage:25s} → {avg:.3f}s ± {std:.3f}s")

    results['video_info'] = {
        'frames': total_frames,
        'duration': round(duration, 2),
        'fps': round(fps, 2),
        'num_shots': len(scene_list),
    }

    # Frames per second for shot detection
    sbd_fps = total_frames / np.mean(timings['shot_detection'])
    results['sbd_fps'] = round(sbd_fps, 1)
    print(f"\n  Shot Detection Speed: {sbd_fps:.1f} FPS")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    print("Run evaluate.py functions from notebook cells.")
