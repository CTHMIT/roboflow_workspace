"""
Roboflow Upload Script
Uploads extracted frames to Roboflow project
"""

import os
import argparse
from pathlib import Path
from roboflow import Roboflow

from setup.config import load_config
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
    rf = Roboflow(api_key=api_key)
    
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

def main():
    parser = argparse.ArgumentParser(description="Upload images to Roboflow")
    parser.add_argument("--frame-list", type=str, default=None, 
                        help="Path to file containing list of image paths. Default: uses upload.frame_list_file from config")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Roboflow API key. Default: reads from .env file (ROBOFLOW_API_KEY)")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Roboflow workspace ID. Default: uses roboflow.workspace from config")
    parser.add_argument("--project", type=str, default=None,
                        help="Roboflow project ID. Default: uses roboflow.project from config")
    parser.add_argument("--batch-name", type=str, default=None,
                        help="Batch name for organizing uploads. Default: uses upload.batch_name from config")
    parser.add_argument("--split", type=str, default=None, choices=["train", "valid", "test"],
                        help="Dataset split. Default: uses upload.split from config")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to config file")
    
    args = parser.parse_args()
    
    try:
        config, env_config = load_config(args.config)
        LOGGER.info(f"✓ Loaded configuration from: {args.config}")
    except FileNotFoundError:
        LOGGER.error(f"Configuration file not found: {args.config}")
        LOGGER.info("Please ensure config.yaml exists")
        exit(1)
    except Exception as e:
        LOGGER.error(f"Failed to load configuration: {str(e)}")
        exit(1)
    
    api_key = args.api_key or env_config.roboflow_api_key
    workspace = args.workspace or config.roboflow.workspace
    project = args.project or config.roboflow.project    
    batch_name = args.batch_name or config.upload.batch_name
    split = args.split or config.upload.split
    frame_list = args.frame_list or config.upload.frame_list_file
        
    validation_errors = []
    
    if not api_key or api_key == "your_api_key_here":
        validation_errors.append("Roboflow API key not set")
        LOGGER.error("ERROR: Roboflow API key is required!")
    
    if not workspace or workspace == "your-workspace":
        validation_errors.append("Workspace ID not set")
        
    if not project or project == "your-project":
        validation_errors.append("Project ID not set")
        LOGGER.error("ERROR: Project ID is required!")
    
    if validation_errors:
        LOGGER.error(f"\n{len(validation_errors)} validation error(s) found")
        exit(1)
    
    LOGGER.info(f"  Workspace: {workspace}")
    LOGGER.info(f"  Project: {project}")
    LOGGER.info(f"  Frame list: {frame_list}")
    LOGGER.info(f"  Batch name: {batch_name or 'None'}")
    LOGGER.info(f"  Split: {split}")
    LOGGER.info(f"  Tags: {config.upload.tags}")
    LOGGER.info("")
    
    frame_paths = load_frame_list(frame_list)
    
    if not frame_paths:
        LOGGER.error(f"No frames found in: {frame_list}")
        LOGGER.info(f"Please check file exists: {frame_list}")
        exit(1)
    
    try:
        tags = ["video-extracted"]
        if batch_name:
            tags.append(batch_name)
        tags.extend(config.upload.tags)
        
        # Upload
        uploaded, failed = upload_to_roboflow(
            image_paths=frame_paths,
            api_key=api_key,
            workspace=workspace,
            project=project,
            batch_name=batch_name,
            split=split
        )
        
        if uploaded > 0:
            LOGGER.info(f"Upload Statistics:")
            LOGGER.info(f"  ✓ Successfully uploaded: {uploaded}")
            LOGGER.info(f"  ✗ Failed: {failed}")
            LOGGER.info(f"View your images at:")
            LOGGER.info(f"  https://app.roboflow.com/{workspace}/{project}")
            LOGGER.info("")
        else:
            LOGGER.error("No images were successfully uploaded")
            exit(1)
        
    except Exception as e:
        LOGGER.error(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()