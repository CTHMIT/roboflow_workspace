"""
Roboflow Upload Script
Uploads extracted frames to Roboflow project
"""

import os
from pathlib import Path
from roboflow import Roboflow

from utils.logger import LOGGER


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
    
    LOGGER.info(f"\nConnected to Roboflow project: {project}")
    LOGGER.info(f"Uploading {len(image_paths)} images...")
    LOGGER.info(f"Split: {split}")
    if batch_name:
        LOGGER.info(f"Batch: {batch_name}")
    LOGGER.info(f"\n{'='*50}\n")
    
    uploaded_count = 0
    failed_count = 0
    
    for i, image_path in enumerate(image_paths, 1):
        try:
            if not os.path.exists(image_path):
                LOGGER.info(f"[{i}/{len(image_paths)}] File not found: {image_path}")
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
            LOGGER.info(f"[{i}/{len(image_paths)}] Uploaded: {Path(image_path).name}")
            
        except Exception as e:
            failed_count += 1
            LOGGER.error(f"[{i}/{len(image_paths)}] Failed: {Path(image_path).name}")
            LOGGER.error(f"    Error: {str(e)}")
    
    LOGGER.info(f"{'='*50}")
    LOGGER.info(f"Upload complete!")
    LOGGER.info(f"Successfully uploaded: {uploaded_count}")
    LOGGER.info(f"Failed: {failed_count}")
    LOGGER.info(f"{'='*50}\n")
    
    return uploaded_count, failed_count


def load_frame_list(frame_list_file="extracted_frames.txt"):
    """
    Load list of extracted frames from file
    
    Args:
        frame_list_file: Path to file containing list of frame paths
    """
    if not os.path.exists(frame_list_file):
        LOGGER.info(f"Error: Frame list file not found: {frame_list_file}")
        return []
    
    with open(frame_list_file, "r") as f:
        frames = [line.strip() for line in f if line.strip()]
    
    LOGGER.info(f"Loaded {len(frames)} frames from {frame_list_file}")
    return frames


if __name__ == "__main__":

    ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"
    WORKSPACE = "your-workspace"
    PROJECT = "your-project"
    BATCH_NAME = "video-frames-batch-1"
    SPLIT = "train"

    if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        LOGGER.info("ERROR: Please set your Roboflow API key!")
        LOGGER.info("Get your API key from: https://app.roboflow.com/settings/api")
        exit(1)
    
    if WORKSPACE == "your-workspace" or PROJECT == "your-project":
        LOGGER.info("ERROR: Please set your workspace and project IDs!")
        LOGGER.info("Find them in your Roboflow project URL")
        exit(1)
    
    LOGGER.info("Starting Roboflow upload...")
    LOGGER.info(f"Workspace: {WORKSPACE}")
    LOGGER.info(f"Project: {PROJECT}")
    
    frame_paths = load_frame_list("extracted_frames.txt")
    
    if not frame_paths:
        LOGGER.info("No frames to upload!")
        exit(1)
    
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
            LOGGER.info("Upload successful!")
            LOGGER.info(f"View your images at: https://app.roboflow.com/{WORKSPACE}/{PROJECT}")
        
    except Exception as e:
        LOGGER.info(f"\nUpload failed with error:")
        LOGGER.info(f"{str(e)}")
        LOGGER.info("\nPlease check:")
        LOGGER.info("1. Your API key is correct")
        LOGGER.info("2. Workspace and project IDs are correct")
        LOGGER.info("3. You have permission to upload to this project")
        LOGGER.info("4. The roboflow package is installed: pip install roboflow")