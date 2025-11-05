"""Train YOLO segmentation model using configuration"""
from pathlib import Path
import yaml
import numpy as np
import shutil
from datetime import datetime

import torch
import sys
import traceback
import logging
from datetime import datetime
import psutil
import torch, gc

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset

from checker.data_format import (
    validate_segmentation_labels, 
    validate_dataset,
)
from checker.debug_fix import (
    debug_and_fix_labels,
)

from setup.config import (
    load_config,
    TrainingConfig, 
    RoboflowConfig,
)
from utils.utils import (
    log_system_info, 
    log_training_config,
    validate_model_file,
)
from utils.logger import LOGGER

def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def train_progress(model:YOLO, data_yaml_path, LOGGER:logging, **train_kwargs):
    """Train with detailed error catching to identify problematic files"""
    
    try:
        results = model.train(
            data=str(data_yaml_path),
            **train_kwargs
        )
        return results
        
    except ValueError as e:
        if "input operand has more dimensions" in str(e):
            LOGGER.error("=" * 80)
            LOGGER.error("DEBUGGING LABEL FORMAT ERROR")
            LOGGER.error("=" * 80)
            LOGGER.error("This error typically means label files have wrong format.")
            LOGGER.error("Running deep label inspection...")
            
            try:
                
                with open(data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                dataset_root = Path(data_yaml_path).parent
                train_path = (dataset_root / data['train']).resolve()
                
                LOGGER.info(f"Attempting to load each image individually...")
                LOGGER.info(f"Train path: {train_path}")
                
                dataset = YOLODataset(
                    img_path=str(train_path),
                    imgsz=train_kwargs.get('imgsz', 640),
                    augment=False,
                    cache=False
                )
                
                LOGGER.info(f"Dataset has {len(dataset)} images")
                LOGGER.info("Testing each image...")
                
                for idx in range(len(dataset)):
                    try:
                        _ = dataset[idx]
                        if idx % 100 == 0:
                            LOGGER.info(f"Checked {idx}/{len(dataset)} images")
                    except Exception as item_error:
                        img_file = dataset.im_files[idx]
                        LOGGER.error(f"FOUND PROBLEMATIC FILE: {img_file}")
                        LOGGER.error(f"Error: {item_error}")
                        
                        label_file = Path(str(img_file).replace('images', 'labels').replace(Path(img_file).suffix, '.txt'))
                        if label_file.exists():
                            LOGGER.error(f"   Label file: {label_file}")
                            with open(label_file, 'r') as f:
                                content = f.read()
                            LOGGER.error(f"   Label content:\n{content}")
                        break
                        
            except Exception as debug_error:
                LOGGER.error(f"Could not debug further: {debug_error}")
                LOGGER.debug(traceback.format_exc())
            
            LOGGER.error("=" * 80)
        
        raise

def train_yolo_segmentation(
    training_config: TrainingConfig,
    roboflow_config: RoboflowConfig,
) -> tuple:
    """
    Train YOLO segmentation model with comprehensive error tracking
    
    Args:
        training_config: Training configuration
        roboflow_config: Roboflow configuration (for dataset path)
        
    Returns:
        Tuple of (training results, best model path)
    """
    

    cleanup_cuda()
        
    run_name = training_config.get_run_name(roboflow_config.version)
    
    training_stage = "initialization"
    
    try:
        # Log system information
        training_stage = "system_info"
        log_system_info()
        
        # Create directories
        training_stage = "directory_creation"
        LOGGER.info("Creating directories...")
        training_config.pretrain_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Pretrain directory: {training_config.pretrain_dir.absolute()}")
        
        # Log training configuration
        training_stage = "config_logging"
        log_training_config(training_config, roboflow_config)
        
        # Validate dataset
        training_stage = "dataset_validation"
        data_yaml_path = roboflow_config.data_yaml_path
        validation_results = validate_dataset(data_yaml_path)
        
        if not validation_results['valid']:
            error_msg = "Dataset validation failed:\n" + "\n".join(validation_results['errors'])
            LOGGER.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        training_stage = "label_validation"
        label_issues = validate_segmentation_labels(data_yaml_path)
        
        # Check for critical issues
        critical_issues = label_issues['corrupt_images'] + label_issues['bad_format']
        if critical_issues:
            LOGGER.error(f"Found {len(critical_issues)} critical label issues that must be fixed!")
            LOGGER.error("Training cannot proceed with corrupted labels.")
            raise ValueError(f"Critical label format issues detected in {len(critical_issues)} files")
        
        if label_issues['empty_labels']:
            LOGGER.warning(f"Found {len(label_issues['empty_labels'])} polygons with < 3 points")
            LOGGER.warning("These may cause issues during training")
        
        if not validation_results['valid']:
            error_msg = "Dataset validation failed:\n" + "\n".join(validation_results['errors'])
            LOGGER.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        if validation_results['warnings']:
            LOGGER.warning("Dataset validation warnings:")
            for warning in validation_results['warnings']:
                LOGGER.warning(f"  - {warning}")
        
        # Validate/download model
        training_stage = "model_loading"
        model_file = training_config.pretrain_dir / f"{training_config.model_name.value}.pt"
        validate_model_file(model_file)
        
        LOGGER.info(f"Loading model: {model_file}")
        model = YOLO(model_file)
        LOGGER.info("Model loaded successfully")
        LOGGER.debug(f"Model info: {model.info()}")


        # Debug and fix labels
        training_stage = "label_debugging"
        LOGGER.info("Running label format debugging...")
        issues = debug_and_fix_labels(data_yaml_path, fix=True)  # Set fix=True to auto-fix

        if issues['malformed'] or issues['empty']:
            LOGGER.error("Critical label issues found that couldn't be auto-fixed!")
            LOGGER.error("Please review the log and fix these files manually.")
            raise ValueError(f"Critical label format errors in {len(issues['malformed']) + len(issues['empty'])} files")

                
        # Train the model
        training_stage = "training"
        LOGGER.info("=" * 80)
        LOGGER.info(f"STARTING TRAINING - {training_config.epochs} EPOCHS")
        LOGGER.info("=" * 80)
        
        train_start_time = datetime.now()
        LOGGER.info(f"Training start time: {train_start_time}")
        
        results = train_progress(
            model=model,
            data_yaml_path=data_yaml_path,
            LOGGER=LOGGER,
            epochs=training_config.epochs,
            imgsz=training_config.img_size,
            batch=training_config.batch_size,
            workers=training_config.workers,
            device=training_config.device.value,
            project=str(training_config.project_dir),
            name=run_name,
            exist_ok=True,
            pretrained=False,
            
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
            cache=training_config.cache,
            
            # Features
            plots=training_config.plots,
            verbose=training_config.verbose,
            seed=training_config.seed,
            deterministic=training_config.deterministic,
            val=True,
            amp=training_config.amp,
        )
        
        train_end_time = datetime.now()
        train_duration = train_end_time - train_start_time
        
        LOGGER.info("=" * 80)
        LOGGER.info("TRAINING COMPLETE")
        LOGGER.info("=" * 80)
        LOGGER.info(f"Training end time: {train_end_time}")
        LOGGER.info(f"Total training duration: {train_duration}")
        LOGGER.info(f"Average time per epoch: {train_duration / training_config.epochs}")
        
        # Get model paths
        training_stage = "model_validation"
        best_model_path = training_config.project_dir / run_name / "weights" / "best.pt"
        last_model_path = training_config.project_dir / run_name / "weights" / "last.pt"
        
        LOGGER.info(f"Best model path: {best_model_path}")
        LOGGER.info(f"Best model exists: {best_model_path.exists()}")
        LOGGER.info(f"Last model path: {last_model_path}")
        LOGGER.info(f"Last model exists: {last_model_path.exists()}")
        
        if best_model_path.exists():
            LOGGER.info(f"Best model size: {best_model_path.stat().st_size / (1024**2):.2f} MB")
        
        if best_model_path.exists():
            LOGGER.info("=" * 80)
            LOGGER.info("VALIDATING BEST MODEL")
            LOGGER.info("=" * 80)
            
            best_model = YOLO(best_model_path)
            metrics = best_model.val()
            
            LOGGER.info("Validation Metrics:")
            LOGGER.info(f"  mAP50: {metrics.seg.map50:.4f}")
            LOGGER.info(f"  mAP50-95: {metrics.seg.map:.4f}")
            LOGGER.info(f"  Precision: {metrics.seg.mp:.4f}")
            LOGGER.info(f"  Recall: {metrics.seg.mr:.4f}")
            
            if hasattr(metrics.seg, 'ap_class_index'):
                LOGGER.info("Per-class mAP50:")
                for idx, ap in zip(metrics.seg.ap_class_index, metrics.seg.ap50):
                    LOGGER.info(f"  Class {idx}: {ap:.4f}")
        else:
            LOGGER.warning("Best model not found - training may have been interrupted")
        
        LOGGER.info("=" * 80)
        LOGGER.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        LOGGER.info("=" * 80)
        
        return results, best_model_path
        
    except FileNotFoundError as e:
        LOGGER.error("=" * 80)
        LOGGER.error(f"FILE NOT FOUND ERROR at stage: {training_stage}")
        LOGGER.error("=" * 80)
        LOGGER.error(f"Error message: {e}")
        LOGGER.debug(f"Full traceback:\n{traceback.format_exc()}")
        raise
        
    except MemoryError as e:
        LOGGER.error("=" * 80)
        LOGGER.error(f"MEMORY ERROR at stage: {training_stage}")
        LOGGER.error("=" * 80)
        LOGGER.error(f"Error message: {e}")
        
        memory = psutil.virtual_memory()
        LOGGER.error(f"Current RAM usage: {memory.percent}%")
        LOGGER.error(f"Available RAM: {memory.available / (1024**3):.2f} GB")
        
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    LOGGER.error(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
                    LOGGER.error(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
        except:
            pass
        
        LOGGER.error("Suggestions:")
        LOGGER.error("  - Reduce batch size")
        LOGGER.error("  - Reduce image size")
        LOGGER.error("  - Reduce number of workers")
        LOGGER.error("  - Use smaller model")
        LOGGER.debug(f"Full traceback:\n{traceback.format_exc()}")
        raise
        
    except KeyboardInterrupt:
        LOGGER.warning("=" * 80)
        LOGGER.warning(f"TRAINING INTERRUPTED BY USER at stage: {training_stage}")
        LOGGER.warning("=" * 80)
        raise
        
    except Exception as e:
        LOGGER.error("=" * 80)
        LOGGER.error(f"UNEXPECTED ERROR at stage: {training_stage}")
        LOGGER.error("=" * 80)
        LOGGER.error(f"Error type: {type(e).__name__}")
        LOGGER.error(f"Error message: {e}")
        LOGGER.error(f"\nFull traceback:")
        LOGGER.error(traceback.format_exc())
        LOGGER.error("\nError Context:")
        LOGGER.error(f"  Stage: {training_stage}")
        LOGGER.error(f"  Working directory: {Path.cwd()}")
        
        try:
            LOGGER.error(f"  Python executable: {sys.executable}")
            LOGGER.error(f"  Python path: {sys.path}")
        except:
            pass        
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
        
        LOGGER.info(f"Resuming training from: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        results = model.train(resume=True, epochs=additional_epochs)
        LOGGER.info("Resumed training complete!")
        return results
    except Exception as e:
        LOGGER.error(f"Resume Error: {e}")
        raise

def main(resume_train=False):
    
    app_config, env_config = load_config()
    results, best_model_path = train_yolo_segmentation(
        app_config.training,
        app_config.roboflow
    )
    
    LOGGER.info(f"\nTraining complete! Best model: {best_model_path}")
    
    if resume_train:
        last_checkpoint = app_config.training.project_dir / \
            app_config.training.get_run_name(app_config.roboflow.version) / \
            "weights" / "last.pt"
        resume_training(last_checkpoint, additional_epochs=100)

if __name__ == "__main__":
    main()