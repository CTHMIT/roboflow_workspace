"""
Video to Image Prediction Script
Extracts frames from videos and runs predictions
"""

import cv2
import os
from pathlib import Path
import sys

_SRC_ROOT = Path(__file__).resolve().parents[1]  # <repo>/src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from utils.logger import LOGGER

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from video at specified intervals
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 30 = 1 frame per second at 30fps)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        LOGGER.error(f"Error: Could not open video {video_path}")
        return []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_name = Path(video_path).stem
    LOGGER.info(f"\nProcessing: {video_name}")
    LOGGER.info(f"FPS: {fps}, Total Frames: {total_frames}")
    
    frame_count = 0
    saved_count = 0
    saved_frames = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
            saved_count += 1
            LOGGER.info(f"Saved: {frame_filename}")
        
        frame_count += 1
    
    cap.release()
    LOGGER.info(f"Extracted {saved_count} frames from {video_name}\n")
    
    return saved_frames


def run_predictions(video_paths, output_dir="output/frames", frame_interval=30):
    """
    Process multiple videos and extract frames
    
    Args:
        video_paths: List of video file paths
        output_dir: Base directory for output
        frame_interval: Extract every Nth frame
    """
    all_frames = []
    
    for video_path in video_paths:
        if not os.path.exists(video_path):
            LOGGER.info(f"Warning: Video not found: {video_path}")
            continue
        
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(output_dir, video_name)
        
        frames = extract_frames(video_path, video_output_dir, frame_interval)
        all_frames.extend(frames)
    
    LOGGER.info(f"\n{'='*50}")
    LOGGER.info(f"Total frames extracted: {len(all_frames)}")
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"{'='*50}\n")
    
    return all_frames


if __name__ == "__main__":

    video_paths = [
        "datasets/video/episode_000002.mp4",
        "datasets/video/episode_000003.mp4"
    ]


    # episode_000002/episode_000002_frame_001110.jpg
    # episode_000002/episode_000002_frame_001350.jpg
    # episode_000002/episode_000002_frame_001770.jpg
    # episode_000002/episode_000002_frame_003030.jpg

    
    OUTPUT_DIR = "output/frames"
    FRAME_INTERVAL = 30 
    
    LOGGER.info("Starting video frame extraction...")
    LOGGER.info(f"Frame interval: {FRAME_INTERVAL} (every {FRAME_INTERVAL} frames)")
    
    extracted_frames = run_predictions(
        video_paths=video_paths,
        output_dir=OUTPUT_DIR,
        frame_interval=FRAME_INTERVAL
    )
        