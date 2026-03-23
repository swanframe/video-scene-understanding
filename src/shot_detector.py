"""
Shot Boundary Detection & Keyframe Extraction
==============================================
Uses PySceneDetect ContentDetector to find shot transitions
in a video, then extracts a representative keyframe from each shot.
"""

import os
import sys
import cv2
import json
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

sys.path.insert(0, '/content/video-scene-understanding')
from src.config import (
    SBD_THRESHOLD, SBD_MIN_SCENE_LEN,
    KEYFRAMES_DIR, RESULTS_DIR
)


def detect_shots(video_path, threshold=None, min_scene_len=None):
    """
    Detect shot boundaries in a video using ContentDetector.

    Args:
        video_path: Path to input video file.
        threshold: ContentDetector threshold (default from config).
        min_scene_len: Minimum scene length in frames.

    Returns:
        List of tuples: [(start_timecode, end_timecode), ...]
    """
    threshold = threshold or SBD_THRESHOLD
    min_scene_len = min_scene_len or SBD_MIN_SCENE_LEN

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )

    print(f"🎬 Analyzing video: {os.path.basename(video_path)}")
    print(f"   Threshold: {threshold}, Min scene length: {min_scene_len} frames")

    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    print(f"✅ Detected {len(scene_list)} shots")
    return scene_list


def extract_keyframes(video_path, scene_list, output_dir=None, method='middle'):
    """
    Extract a representative keyframe from each detected shot.

    Args:
        video_path: Path to input video file.
        scene_list: List of (start, end) timecode tuples from detect_shots().
        output_dir: Directory to save keyframe images.
        method: Keyframe selection method:
                'middle' — frame at the midpoint of each shot
                'first'  — first frame of each shot
                'best'   — frame with highest Laplacian variance (sharpest)

    Returns:
        List of dicts with keyframe metadata.
    """
    output_dir = output_dir or KEYFRAMES_DIR
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    keyframes_meta = []

    print(f"\n📸 Extracting keyframes (method: {method})...")

    for idx, (start, end) in enumerate(scene_list):
        start_frame = start.get_frames()
        end_frame = end.get_frames()
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        duration = end_sec - start_sec

        # Select target frame based on method
        if method == 'first':
            target_frame = start_frame
        elif method == 'middle':
            target_frame = (start_frame + end_frame) // 2
        elif method == 'best':
            target_frame = _find_sharpest_frame(cap, start_frame, end_frame)
        else:
            target_frame = (start_frame + end_frame) // 2

        # Read and save the keyframe
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if ret:
            filename = f"{video_name}_shot{idx:03d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)

            meta = {
                'shot_index': idx,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': round(start_sec, 2),
                'end_time': round(end_sec, 2),
                'duration': round(duration, 2),
                'keyframe_idx': target_frame,
                'keyframe_file': filename,
                'keyframe_path': filepath,
            }
            keyframes_meta.append(meta)

            print(f"  Shot {idx:3d} │ {start_sec:7.2f}s → {end_sec:7.2f}s │ "
                  f"duration: {duration:5.2f}s │ keyframe: frame {target_frame}")
        else:
            print(f"  Shot {idx:3d} │ ⚠️  Could not read frame {target_frame}")

    cap.release()

    # Save metadata as JSON
    meta_path = os.path.join(output_dir, f"{video_name}_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(keyframes_meta, f, indent=2)

    print(f"\n✅ Extracted {len(keyframes_meta)} keyframes → {output_dir}")
    print(f"📄 Metadata saved → {meta_path}")

    return keyframes_meta


def _find_sharpest_frame(cap, start_frame, end_frame, sample_count=10):
    """
    Find the sharpest frame in a range using Laplacian variance.
    Samples evenly across the shot to avoid reading every frame.
    """
    step = max(1, (end_frame - start_frame) // sample_count)
    best_frame = start_frame
    best_score = -1

    for f in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if score > best_score:
                best_score = score
                best_frame = f

    return best_frame


def get_shot_statistics(scene_list):
    """
    Compute summary statistics for detected shots.

    Returns:
        Dict with min/max/mean/median duration and shot count.
    """
    durations = [
        end.get_seconds() - start.get_seconds()
        for start, end in scene_list
    ]

    stats = {
        'num_shots': len(scene_list),
        'total_duration': round(sum(durations), 2),
        'min_duration': round(min(durations), 2) if durations else 0,
        'max_duration': round(max(durations), 2) if durations else 0,
        'mean_duration': round(np.mean(durations), 2) if durations else 0,
        'median_duration': round(np.median(durations), 2) if durations else 0,
    }
    return stats


def print_shot_statistics(stats):
    """Pretty-print shot statistics."""
    print("\n" + "=" * 55)
    print("SHOT DETECTION STATISTICS")
    print("=" * 55)
    print(f"  Total shots      : {stats['num_shots']}")
    print(f"  Total duration   : {stats['total_duration']:.2f}s")
    print(f"  Shortest shot    : {stats['min_duration']:.2f}s")
    print(f"  Longest shot     : {stats['max_duration']:.2f}s")
    print(f"  Mean duration    : {stats['mean_duration']:.2f}s")
    print(f"  Median duration  : {stats['median_duration']:.2f}s")
    print("=" * 55)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "/content/video-scene-understanding/videos/synthetic_test.mp4"

    scene_list = detect_shots(video_path)
    stats = get_shot_statistics(scene_list)
    print_shot_statistics(stats)
    keyframes = extract_keyframes(video_path, scene_list, method='middle')
