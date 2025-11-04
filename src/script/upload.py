"""
Roboflow Upload Script
Uploads extracted frames to Roboflow project
"""

import os
from pathlib import Path
from roboflow import Roboflow


def upload_to_roboflow(image_paths, api_key, workspace, project, batch_name=None, split="train"):
    """
    Upload images to Roboflow project
    
    Args:
        image_paths: List of image file paths to upload
        api_key: Your Roboflow API key
        workspace: Your Roboflow workspace ID
        project: Your Roboflow project ID
        batch_name: Optional batch name for organizing uploads
        split: Dataset split - 'train', 'valid', or 'test' (default: 'train')
    """
    # Initialize Roboflow
    rf = Roboflow(api_key=api_key)
    
    # Get project
    project_obj = rf.workspace(workspace).project(project)
    
    print(f"\nConnected to Roboflow project: {project}")
    print(f"Uploading {len(image_paths)} images...")
    print(f"Split: {split}")
    if batch_name:
        print(f"Batch: {batch_name}")
    print(f"\n{'='*50}\n")
    
    uploaded_count = 0
    failed_count = 0
    
    for i, image_path in enumerate(image_paths, 1):
        try:
            if not os.path.exists(image_path):
                print(f"[{i}/{len(image_paths)}] ❌ File not found: {image_path}")
                failed_count += 1
                continue
            
            # Upload image
            project_obj.upload(
                image_path=image_path,
                split=split,
                batch_name=batch_name,
                tag_names=["video-extracted"] if not batch_name else ["video-extracted", batch_name]
            )
            
            uploaded_count += 1
            print(f"[{i}/{len(image_paths)}] ✓ Uploaded: {Path(image_path).name}")
            
        except Exception as e:
            failed_count += 1
            print(f"[{i}/{len(image_paths)}] ❌ Failed: {Path(image_path).name}")
            print(f"    Error: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Upload complete!")
    print(f"Successfully uploaded: {uploaded_count}")
    print(f"Failed: {failed_count}")
    print(f"{'='*50}\n")
    
    return uploaded_count, failed_count


def load_frame_list(frame_list_file="extracted_frames.txt"):
    """
    Load list of extracted frames from file
    
    Args:
        frame_list_file: Path to file containing list of frame paths
    """
    if not os.path.exists(frame_list_file):
        print(f"Error: Frame list file not found: {frame_list_file}")
        return []
    
    with open(frame_list_file, "r") as f:
        frames = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(frames)} frames from {frame_list_file}")
    return frames


if __name__ == "__main__":
    # ====== CONFIGURATION ======
    # Get your API key from: https://app.roboflow.com/settings/api
    ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"
    
    # Your workspace and project IDs (found in your Roboflow project URL)
    # URL format: https://app.roboflow.com/WORKSPACE/PROJECT
    WORKSPACE = "your-workspace"
    PROJECT = "your-project"
    
    # Optional: Batch name for organizing uploads
    BATCH_NAME = "video-frames-batch-1"
    
    # Dataset split: 'train', 'valid', or 'test'
    SPLIT = "train"
    
    # ===========================
    
    # Validate configuration
    if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️  ERROR: Please set your Roboflow API key!")
        print("Get your API key from: https://app.roboflow.com/settings/api")
        exit(1)
    
    if WORKSPACE == "your-workspace" or PROJECT == "your-project":
        print("⚠️  ERROR: Please set your workspace and project IDs!")
        print("Find them in your Roboflow project URL")
        exit(1)
    
    print("Starting Roboflow upload...")
    print(f"Workspace: {WORKSPACE}")
    print(f"Project: {PROJECT}")
    
    # Option 1: Load frames from extracted_frames.txt (created by predict.py)
    frame_paths = load_frame_list("extracted_frames.txt")
    
    # Option 2: Or manually specify image paths
    # frame_paths = [
    #     "output/frames/episode_000000/episode_000000_frame_000000.jpg",
    #     "output/frames/episode_000000/episode_000000_frame_000030.jpg",
    #     # ... add more paths
    # ]
    
    # Option 3: Or load all images from a directory
    # frame_paths = []
    # for root, dirs, files in os.walk("output/frames"):
    #     for file in files:
    #         if file.endswith(('.jpg', '.jpeg', '.png')):
    #             frame_paths.append(os.path.join(root, file))
    
    if not frame_paths:
        print("No frames to upload!")
        exit(1)
    
    # Upload to Roboflow
    try:
        uploaded, failed = upload_to_roboflow(
            image_paths=frame_paths,
            api_key=ROBOFLOW_API_KEY,
            workspace=WORKSPACE,
            project=PROJECT,
            batch_name=BATCH_NAME,
            split=SPLIT
        )
        
        if uploaded > 0:
            print("✅ Upload successful!")
            print(f"View your images at: https://app.roboflow.com/{WORKSPACE}/{PROJECT}")
        
    except Exception as e:
        print(f"\n❌ Upload failed with error:")
        print(f"{str(e)}")
        print("\nPlease check:")
        print("1. Your API key is correct")
        print("2. Workspace and project IDs are correct")
        print("3. You have permission to upload to this project")
        print("4. The roboflow package is installed: pip install roboflow")