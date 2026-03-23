"""
Video Scene Understanding Pipeline
====================================
Master pipeline that chains all modules:
  Video → Shot Detection → Keyframe Extraction →
  Scene Classification → Captioning → Structured Summary
"""

import os
import sys
import json
import time
from datetime import timedelta

sys.path.insert(0, '/content/video-scene-understanding')
from src.config import KEYFRAMES_DIR, RESULTS_DIR, OUTPUTS_DIR


class VideoScenePipeline:
    """
    End-to-end video scene understanding pipeline.
    """

    def __init__(self, sbd_threshold=None, keyframe_method='middle'):
        """
        Initialize all pipeline modules.

        Args:
            sbd_threshold: ContentDetector threshold (None = use config default).
            keyframe_method: 'middle', 'first', or 'best'.
        """
        self.sbd_threshold = sbd_threshold
        self.keyframe_method = keyframe_method

        print("=" * 65)
        print("  VIDEO SCENE UNDERSTANDING PIPELINE — INITIALIZATION")
        print("=" * 65)

        # Lazy-load modules to show progress
        print("\n[1/3] Loading Shot Detector...")
        from src.shot_detector import detect_shots, extract_keyframes, get_shot_statistics
        self._detect_shots = detect_shots
        self._extract_keyframes = extract_keyframes
        self._get_shot_statistics = get_shot_statistics
        print("  ✅ Shot Detector ready")

        print("[2/3] Loading Scene Classifier (ViT)...")
        from src.classifier import SceneClassifier
        self.classifier = SceneClassifier()

        print("[3/3] Loading Scene Captioner (BLIP)...")
        from src.captioner import SceneCaptioner
        self.captioner = SceneCaptioner()

        print("\n" + "=" * 65)
        print("  ✅ PIPELINE READY")
        print("=" * 65)

    def process_video(self, video_path, output_dir=None):
        """
        Run the full pipeline on a video.

        Args:
            video_path: Path to input video file.
            output_dir: Output directory (auto-created per video).

        Returns:
            Dict with complete pipeline results.
        """
        start_time = time.time()

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = output_dir or os.path.join(RESULTS_DIR, video_name)
        keyframes_dir = os.path.join(output_dir, 'keyframes')
        os.makedirs(keyframes_dir, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"  PROCESSING: {video_name}")
        print(f"{'='*65}")

        # ── Step 1: Shot Boundary Detection ───────────────────────
        print(f"\n📌 STEP 1/4 — Shot Boundary Detection")
        scene_list = self._detect_shots(
            video_path, threshold=self.sbd_threshold
        )
        stats = self._get_shot_statistics(scene_list)

        # ── Step 2: Keyframe Extraction ───────────────────────────
        print(f"\n📌 STEP 2/4 — Keyframe Extraction")
        metadata = self._extract_keyframes(
            video_path, scene_list,
            output_dir=keyframes_dir,
            method=self.keyframe_method,
        )

        # ── Step 3: Scene Classification ──────────────────────────
        print(f"\n📌 STEP 3/4 — Scene Classification (ViT)")
        metadata = self.classifier.classify_keyframes(metadata)

        # ── Step 4: Scene Captioning ──────────────────────────────
        print(f"\n📌 STEP 4/4 — Scene Captioning (BLIP)")
        metadata = self.captioner.caption_keyframes(metadata)

        # ── Build Final Results ───────────────────────────────────
        elapsed = time.time() - start_time

        results = {
            'video_name': video_name,
            'video_path': video_path,
            'processing_time_seconds': round(elapsed, 2),
            'processing_time_human': str(timedelta(seconds=int(elapsed))),
            'shot_statistics': stats,
            'num_shots': len(metadata),
            'scenes': metadata,
        }

        # Save JSON results
        json_path = os.path.join(output_dir, f'{video_name}_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate HTML report
        html_path = os.path.join(output_dir, f'{video_name}_report.html')
        self._generate_html_report(results, html_path)

        # Print summary
        self._print_summary(results)

        print(f"\n💾 JSON results → {json_path}")
        print(f"📄 HTML report  → {html_path}")
        print(f"⏱️  Total time   → {results['processing_time_human']}")

        return results

    def _generate_html_report(self, results, output_path):
        """Generate a visual HTML report with timeline and scene cards."""

        video_name = results['video_name']
        stats = results['shot_statistics']
        scenes = results['scenes']

        # Build scene cards HTML
        scene_cards = ""
        for scene in scenes:
            keyframe_file = scene.get('keyframe_file', '')
            scene_label = scene.get('scene_label', 'N/A')
            confidence = scene.get('scene_confidence', 0)
            caption = scene.get('caption', 'No caption')
            start_t = scene.get('start_time', 0)
            end_t = scene.get('end_time', 0)
            duration = scene.get('duration', 0)
            shot_idx = scene.get('shot_index', 0)

            # Top-k predictions
            top_k_html = ""
            for pred in scene.get('scene_top_k', []):
                bar_width = pred['confidence'] * 100
                top_k_html += f"""
                <div class="pred-row">
                    <span class="pred-label">{pred['label']}</span>
                    <div class="pred-bar-bg">
                        <div class="pred-bar" style="width: {bar_width}%"></div>
                    </div>
                    <span class="pred-conf">{pred['confidence']:.1%}</span>
                </div>"""

            scene_cards += f"""
            <div class="scene-card">
                <div class="scene-header">
                    <span class="shot-badge">Shot {shot_idx}</span>
                    <span class="time-badge">{start_t:.1f}s → {end_t:.1f}s ({duration:.1f}s)</span>
                </div>
                <div class="scene-body">
                    <div class="keyframe-container">
                        <img src="keyframes/{keyframe_file}" alt="Shot {shot_idx}" class="keyframe-img">
                    </div>
                    <div class="scene-info">
                        <div class="scene-label">
                            <span class="label-tag">{scene_label}</span>
                            <span class="conf-text">{confidence:.1%}</span>
                        </div>
                        <div class="caption-text">"{caption}"</div>
                        <div class="top-k-section">
                            <div class="top-k-title">Top Predictions</div>
                            {top_k_html}
                        </div>
                    </div>
                </div>
            </div>"""

        # Timeline segments
        total_duration = stats.get('total_duration', 1)
        timeline_segments = ""
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c',
                  '#e67e22', '#34495e', '#16a085', '#c0392b']
        for i, scene in enumerate(scenes):
            width_pct = (scene.get('duration', 0) / total_duration) * 100
            color = colors[i % len(colors)]
            timeline_segments += f"""
            <div class="timeline-segment" style="width: {width_pct}%; background: {color};"
                 title="Shot {scene.get('shot_index', i)}: {scene.get('scene_label', 'N/A')}">
                <span class="segment-label">S{scene.get('shot_index', i)}</span>
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Scene Report — {video_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f5f7fa; color: #2c3e50; padding: 24px; }}
        .container {{ max-width: 960px; margin: 0 auto; }}
        h1 {{ font-size: 1.8em; margin-bottom: 8px; color: #2c3e50; }}
        .subtitle {{ color: #7f8c8d; margin-bottom: 24px; font-size: 0.95em; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }}
        .stat-card {{ background: white; border-radius: 10px; padding: 16px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .stat-value {{ font-size: 1.6em; font-weight: 700; color: #3498db; }}
        .stat-label {{ font-size: 0.8em; color: #95a5a6; margin-top: 4px; }}
        .timeline-container {{ background: white; border-radius: 10px; padding: 16px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .timeline-title {{ font-weight: 600; margin-bottom: 10px; }}
        .timeline {{ display: flex; height: 40px; border-radius: 6px; overflow: hidden; }}
        .timeline-segment {{ display: flex; align-items: center; justify-content: center; min-width: 30px; transition: opacity 0.2s; cursor: pointer; }}
        .timeline-segment:hover {{ opacity: 0.8; }}
        .segment-label {{ color: white; font-size: 0.75em; font-weight: 700; }}
        .scene-card {{ background: white; border-radius: 10px; margin-bottom: 16px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .scene-header {{ display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: #f8f9fa; border-bottom: 1px solid #eee; }}
        .shot-badge {{ background: #3498db; color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }}
        .time-badge {{ color: #7f8c8d; font-size: 0.85em; }}
        .scene-body {{ display: flex; padding: 16px; gap: 16px; }}
        .keyframe-container {{ flex-shrink: 0; }}
        .keyframe-img {{ width: 260px; height: 180px; object-fit: cover; border-radius: 8px; }}
        .scene-info {{ flex: 1; }}
        .scene-label {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }}
        .label-tag {{ background: #2ecc71; color: white; padding: 4px 12px; border-radius: 6px; font-weight: 600; font-size: 0.95em; }}
        .conf-text {{ color: #95a5a6; font-size: 0.85em; }}
        .caption-text {{ font-style: italic; color: #555; margin-bottom: 14px; line-height: 1.5; }}
        .top-k-section {{ border-top: 1px solid #eee; padding-top: 10px; }}
        .top-k-title {{ font-size: 0.8em; color: #95a5a6; margin-bottom: 6px; }}
        .pred-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
        .pred-label {{ width: 80px; font-size: 0.8em; text-align: right; }}
        .pred-bar-bg {{ flex: 1; height: 8px; background: #ecf0f1; border-radius: 4px; overflow: hidden; }}
        .pred-bar {{ height: 100%; background: #3498db; border-radius: 4px; }}
        .pred-conf {{ font-size: 0.8em; color: #7f8c8d; width: 45px; }}
        .footer {{ text-align: center; color: #bdc3c7; font-size: 0.8em; margin-top: 32px; padding-top: 16px; border-top: 1px solid #eee; }}
        @media (max-width: 640px) {{
            .scene-body {{ flex-direction: column; }}
            .keyframe-img {{ width: 100%; height: auto; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎬 Video Scene Analysis Report</h1>
        <div class="subtitle">{video_name} — {results['processing_time_human']} processing time</div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['num_shots']}</div>
                <div class="stat-label">Total Shots</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['total_duration']:.1f}s</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['mean_duration']:.1f}s</div>
                <div class="stat-label">Avg Shot Length</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['min_duration']:.1f}s</div>
                <div class="stat-label">Shortest Shot</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['max_duration']:.1f}s</div>
                <div class="stat-label">Longest Shot</div>
            </div>
        </div>

        <div class="timeline-container">
            <div class="timeline-title">📐 Video Timeline</div>
            <div class="timeline">{timeline_segments}</div>
        </div>

        {scene_cards}

        <div class="footer">
            Generated by Video Scene Understanding Pipeline<br>
            ViT (Scene Classification) + BLIP (Captioning) + PySceneDetect (Shot Detection)
        </div>
    </div>
</body>
</html>"""

        with open(output_path, 'w') as f:
            f.write(html)
        print(f"📄 HTML report generated → {output_path}")

    def _print_summary(self, results):
        """Print a formatted pipeline summary."""
        print(f"\n{'='*65}")
        print(f"  PIPELINE RESULTS SUMMARY")
        print(f"{'='*65}")
        print(f"  Video        : {results['video_name']}")
        print(f"  Total shots  : {results['num_shots']}")
        print(f"  Duration     : {results['shot_statistics']['total_duration']:.1f}s")

        print(f"\n  {'Shot':>5s} │ {'Time':>15s} │ {'Scene':>10s} │ {'Conf':>6s} │ Caption")
        print(f"  {'─'*5} │ {'─'*15} │ {'─'*10} │ {'─'*6} │ {'─'*30}")

        for s in results['scenes']:
            time_str = f"{s['start_time']:.1f}s→{s['end_time']:.1f}s"
            caption = s.get('caption', 'N/A')
            if len(caption) > 35:
                caption = caption[:32] + '...'
            print(f"  {s['shot_index']:5d} │ {time_str:>15s} │ "
                  f"{s.get('scene_label', 'N/A'):>10s} │ "
                  f"{s.get('scene_confidence', 0):>5.1%} │ {caption}")

        print(f"{'='*65}")


if __name__ == "__main__":
    pipeline = VideoScenePipeline()
    video_path = "/content/video-scene-understanding/videos/synthetic_test.mp4"
    results = pipeline.process_video(video_path)
