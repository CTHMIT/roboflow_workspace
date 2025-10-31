"""Main orchestrator for YOLO segmentation training pipeline"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from setup.config import load_config, AppConfig, EnvironmentConfig
from utils.download_roboflow import download_roboflow_dataset
from script.train import train_yolo_segmentation, resume_training


def download_only(app_config: AppConfig, env_config: EnvironmentConfig) -> None:
    """Download dataset only"""
    print("=" * 70)
    print("📥 DOWNLOADING DATASET")
    print("=" * 70)
    
    dataset_path = download_roboflow_dataset(
        app_config.roboflow,
        env_config
    )
    
    print(f"\n✅ Dataset downloaded successfully!")
    print(f"📂 Location: {dataset_path}")


def train_only(app_config: AppConfig) -> Path:
    """Train model only"""
    print("=" * 70)
    print("🚀 TRAINING MODEL")
    print("=" * 70)
    
    results, best_model_path = train_yolo_segmentation(
        app_config.training,
        app_config.roboflow
    )
    
    print(f"\n✅ Training completed successfully!")
    print(f"🏆 Best model: {best_model_path}")
    
    return best_model_path


def full_pipeline(app_config: AppConfig, env_config: EnvironmentConfig) -> Path:
    """Run full pipeline: download + train"""
    print("=" * 70)
    print("🔄 FULL PIPELINE: DOWNLOAD + TRAIN")
    print("=" * 70)
    
    # Step 1: Download dataset
    print("\n" + "=" * 70)
    print("STEP 1: Downloading Dataset")
    print("=" * 70)
    
    dataset_path = download_roboflow_dataset(
        app_config.roboflow,
        env_config
    )
    
    print(f"\n✅ Dataset ready at: {dataset_path}")
    
    # Step 2: Train model
    print("\n" + "=" * 70)
    print("STEP 2: Training Model")
    print("=" * 70)
    
    results, best_model_path = train_yolo_segmentation(
        app_config.training,
        app_config.roboflow
    )
    
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"📂 Dataset: {dataset_path}")
    print(f"🏆 Best model: {best_model_path}")
    
    return best_model_path


def resume_pipeline(app_config: AppConfig, checkpoint: Optional[str] = None) -> None:
    """Resume training from checkpoint"""
    print("=" * 70)
    print("🔄 RESUMING TRAINING")
    print("=" * 70)
    
    # Determine checkpoint path
    if checkpoint:
        checkpoint_path = Path(checkpoint)
    else:
        # Use last checkpoint from config
        run_name = app_config.training.get_run_name(app_config.roboflow.version)
        checkpoint_path = app_config.training.project_dir / run_name / "weights" / "last.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"📂 Checkpoint: {checkpoint_path}")
    
    results = resume_training(checkpoint_path)
    
    print("\n✅ Resumed training completed!")


def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description="YOLO Segmentation Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (download + train)
  python main.py --full
  
  # Download dataset only
  python main.py --download
  
  # Train only (dataset must exist)
  python main.py --train
  
  # Resume training from last checkpoint
  python main.py --resume
  
  # Resume from specific checkpoint
  python main.py --resume --checkpoint models/yolo11m-seg_v4/weights/last.pt
  
  # Use custom config file
  python main.py --full --config my_config.yaml
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline: download dataset and train model"
    )
    mode_group.add_argument(
        "--download",
        action="store_true",
        help="Download dataset only"
    )
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train model only (dataset must exist)"
    )
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for resuming (default: use last checkpoint)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"📋 Loading configuration from: {args.config}")
        app_config, env_config = load_config(args.config)
        print("✅ Configuration loaded successfully!\n")
        
        # Execute requested mode
        if args.full:
            full_pipeline(app_config, env_config)
        elif args.download:
            download_only(app_config, env_config)
        elif args.train:
            train_only(app_config)
        elif args.resume:
            resume_pipeline(app_config, args.checkpoint)
        
        print("\n" + "=" * 70)
        print("✨ ALL DONE! ✨")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()