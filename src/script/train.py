"""Train YOLO segmentation model using configuration"""
import os
from pathlib import Path
from ultralytics import YOLO

from setup.config import (
    load_config,
    TrainingConfig, 
    RoboflowConfig,
)

def train_yolo_segmentation(
    training_config: TrainingConfig,
    roboflow_config: RoboflowConfig
) -> tuple[object, Path]:
    """
    Train YOLO segmentation model
    
    Args:
        training_config: Training configuration
        roboflow_config: Roboflow configuration (for dataset path)
        
    Returns:
        Tuple of (training results, best model path)
    """
    try:
        # Create directories
        training_config.pretrain_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Pretrain directory: {training_config.pretrain_dir.absolute()}")
        
        # Verify dataset exists
        data_yaml_path = roboflow_config.data_yaml_path
        if not data_yaml_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {data_yaml_path}\n"
                "Please run download_roboflow.py first or use main.py"
            )
        print(f"✅ Dataset found: {data_yaml_path}")
        
        # Load pretrained model
        model_file = f"{training_config.model_name.value}.pt"
        print(f"🔧 Loading model: {model_file}")
        model = YOLO(model_file)  # Automatically downloads if not present
        
        # Display training configuration
        print(f"\n📊 Training Configuration:")
        print(f"   Model: {training_config.model_name.value}")
        print(f"   Epochs: {training_config.epochs}")
        print(f"   Batch size: {training_config.batch_size}")
        print(f"   Image size: {training_config.img_size}")
        print(f"   Device: {training_config.device.value}")
        print(f"   Optimizer: {training_config.optimizer.value}")
        print(f"   Workers: {training_config.workers}")
        
        # Get run name
        run_name = training_config.get_run_name(roboflow_config.version)
        print(f"   Run name: {run_name}")
        
        # Train the model
        print(f"\n🚀 Starting training for {training_config.epochs} epochs...")
        results = model.train(
            data=str(data_yaml_path),
            epochs=training_config.epochs,
            imgsz=training_config.img_size,
            batch=training_config.batch_size,
            workers=training_config.workers,
            device=training_config.device.value,
            project=str(training_config.project_dir),
            name=run_name,
            exist_ok=True,
            pretrained=True,
            
            # Optimizer settings
            optimizer=training_config.optimizer.value,
            lr0=training_config.lr0,
            lrf=training_config.lrf,
            momentum=training_config.momentum,
            weight_decay=training_config.weight_decay,
            
            # Warmup settings
            warmup_epochs=training_config.warmup_epochs,
            warmup_momentum=training_config.warmup_momentum,
            
            # Loss gains
            box=training_config.box_gain,
            cls=training_config.cls_gain,
            dfl=training_config.dfl_gain,
            
            # Training behavior
            patience=training_config.patience,
            save=True,
            save_period=training_config.save_period,
            cache=training_config.cache,
            
            # Features
            plots=training_config.plots,
            verbose=training_config.verbose,
            seed=training_config.seed,
            deterministic=training_config.deterministic,
            val=True,
            amp=training_config.amp,
        )
        
        print("\n✅ Training complete!")
        
        # Get best model path
        best_model_path = training_config.project_dir / run_name / "weights" / "best.pt"
        last_model_path = training_config.project_dir / run_name / "weights" / "last.pt"
        
        print(f"🏆 Best model saved at: {best_model_path}")
        print(f"💾 Last checkpoint at: {last_model_path}")
        
        # Validate the best model
        if best_model_path.exists():
            print("\n📊 Validating best model...")
            best_model = YOLO(best_model_path)
            metrics = best_model.val()
            
            print("\n📈 Validation Metrics:")
            print(f"   mAP50: {metrics.seg.map50:.4f}")
            print(f"   mAP50-95: {metrics.seg.map:.4f}")
        else:
            print("⚠️  Best model not found, training may have been interrupted")
        
        return results, best_model_path
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        raise
    except Exception as e:
        print(f"❌ Training Error: {e}")
        raise


def resume_training(
    checkpoint_path: Path,
    additional_epochs: int = 100
) -> object:
    """
    Resume training from a checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        additional_epochs: Number of additional epochs to train
        
    Returns:
        Training results
    """
    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"🔄 Resuming training from: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        results = model.train(resume=True, epochs=additional_epochs)
        print("✅ Resumed training complete!")
        return results
    except Exception as e:
        print(f"❌ Resume Error: {e}")
        raise

def main():
    
    
    # Load configuration
    app_config, env_config = load_config()
    
    # Train model
    results, best_model_path = train_yolo_segmentation(
        app_config.training,
        app_config.roboflow
    )
    
    print(f"\n🎉 Training complete! Best model: {best_model_path}")
    
    # Uncomment to resume training if needed
    # last_checkpoint = app_config.training.project_dir / \
    #     app_config.training.get_run_name(app_config.roboflow.version) / \
    #     "weights" / "last.pt"
    # resume_training(last_checkpoint, additional_epochs=100)

if __name__ == "__main__":
    main()