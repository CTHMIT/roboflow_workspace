"""Download dataset from Roboflow using configuration"""
import os
from pathlib import Path
from roboflow import Roboflow

from setup.config import load_config, RoboflowConfig, EnvironmentConfig


def download_roboflow_dataset(
    roboflow_config: RoboflowConfig,
    env_config: EnvironmentConfig
) -> Path:
    """
    Download YOLO segmentation dataset from Roboflow
    
    Args:
        roboflow_config: Roboflow configuration
        env_config: Environment configuration with API key
        
    Returns:
        Path to downloaded dataset
    """
    try:
        # Create save directory
        roboflow_config.dataset_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Save directory: {roboflow_config.dataset_path.absolute()}")
        
        # Connect to Roboflow
        print("🔗 Connecting to Roboflow...")
        rf = Roboflow(api_key=env_config.roboflow_api_key)
        
        # Access project
        print(f"📦 Accessing project: {roboflow_config.workspace}/{roboflow_config.project}")
        project = rf.workspace(roboflow_config.workspace).project(roboflow_config.project)
        
        # Get version
        print(f"📋 Fetching version {roboflow_config.version}...")
        version = project.version(roboflow_config.version)
        
        # Download dataset
        print(f"⬇️  Downloading dataset in {roboflow_config.format.value} format...")
        dataset = version.download(
            roboflow_config.format.value,
            location=str(roboflow_config.dataset_path),
            overwrite=roboflow_config.overwrite
        )
        
        print(f"✅ Download complete!")
        print(f"📂 Dataset location: {dataset.location}")
        
        # Verify data.yaml exists
        data_yaml = roboflow_config.data_yaml_path
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")
        
        print(f"✅ Verified data.yaml at: {data_yaml}")
        
        return roboflow_config.dataset_path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise

def main():
    app_config, env_config = load_config()
    
    dataset_path = download_roboflow_dataset(
        app_config.roboflow,
        env_config
    )
    
    print(f"\n🎉 Dataset ready at: {dataset_path}")



if __name__ == "__main__":
    main()
    
    