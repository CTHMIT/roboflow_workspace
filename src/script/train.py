"""Train YOLO segmentation model using configuration"""
import os
from pathlib import Path
from ultralytics import YOLO

from setup.config import (
    load_config,
    TrainingConfig, 
    RoboflowConfig,
)
import torch
import sys
import traceback
import logging
from datetime import datetime
import platform
import psutil
from typing import Optional

import torch, gc

from checker.data_format import validate_segmentation_labels
from utils.logger import LOGGER

def cleanup_cuda():
    gc.collect()
    torch.cuda.empty_cache()

def log_system_info():
    """Log detailed system information for debugging"""
    LOGGER.info("=" * 80)
    LOGGER.info("SYSTEM INFORMATION")
    LOGGER.info("=" * 80)
    
    # Python environment
    LOGGER.info(f"Python Version: {sys.version}")
    LOGGER.info(f"Platform: {platform.platform()}")
    LOGGER.info(f"Processor: {platform.processor()}")
    
    # CPU info
    LOGGER.info(f"CPU Cores (Physical): {psutil.cpu_count(logical=False)}")
    LOGGER.info(f"CPU Cores (Logical): {psutil.cpu_count(logical=True)}")
    
    # Memory info
    memory = psutil.virtual_memory()
    LOGGER.info(f"Total RAM: {memory.total / (1024**3):.2f} GB")
    LOGGER.info(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    LOGGER.info(f"RAM Usage: {memory.percent}%")
    
    # Disk info
    disk = psutil.disk_usage('/')
    LOGGER.info(f"Disk Total: {disk.total / (1024**3):.2f} GB")
    LOGGER.info(f"Disk Free: {disk.free / (1024**3):.2f} GB")
    LOGGER.info(f"Disk Usage: {disk.percent}%")
    
    # GPU info
    try:
        import torch
        LOGGER.info(f"PyTorch Version: {torch.__version__}")
        LOGGER.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            LOGGER.info(f"CUDA Version: {torch.version.cuda}")
            LOGGER.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                LOGGER.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                LOGGER.info(f"  Memory Total: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
                LOGGER.info(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
                LOGGER.info(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
    except ImportError:
        LOGGER.warning("PyTorch not available - cannot get GPU info")
    except Exception as e:
        LOGGER.error(f"Error getting GPU info: {e}")
    
    # Ultralytics version
    try:
        import ultralytics
        LOGGER.info(f"Ultralytics Version: {ultralytics.__version__}")
    except ImportError:
        LOGGER.warning("Ultralytics not available")
    
    LOGGER.info("=" * 80)


def validate_dataset(data_yaml_path: Path) -> dict:
    """Validate dataset and return detailed information"""
    import yaml
    
    LOGGER.info("=" * 80)
    LOGGER.info("DATASET VALIDATION")
    LOGGER.info("=" * 80)
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    # Check if data.yaml exists
    if not data_yaml_path.exists():
        validation_results['valid'] = False
        validation_results['errors'].append(f"data.yaml not found at: {data_yaml_path}")
        LOGGER.error(f"❌ data.yaml not found at: {data_yaml_path}")
        return validation_results
    
    LOGGER.info(f"✅ data.yaml found at: {data_yaml_path}")
    
    # Parse data.yaml
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        LOGGER.info(f"✅ data.yaml parsed successfully")
        LOGGER.debug(f"data.yaml contents: {data_config}")
    except Exception as e:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Failed to parse data.yaml: {e}")
        LOGGER.error(f"❌ Failed to parse data.yaml: {e}")
        LOGGER.debug(traceback.format_exc())
        return validation_results
    
    # Validate required fields
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in data_config:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Missing required field in data.yaml: {field}")
            LOGGER.error(f"❌ Missing required field: {field}")
        else:
            LOGGER.info(f"✅ Field '{field}': {data_config[field]}")
    
    if not validation_results['valid']:
        return validation_results
    
    # Validate dataset paths
    dataset_root = data_yaml_path
    
    for split in ['train', 'val', 'test']:
        if split in data_config:
            split_path = (dataset_root / data_config[split]).resolve()
            if split_path.exists():
                # Count images
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                images = []
                for ext in image_extensions:
                    images.extend(list(split_path.rglob(f'*{ext}')))
                
                LOGGER.info(f"✅ {split} split found: {split_path}")
                LOGGER.info(f"   Images found: {len(images)}")
                validation_results['info'][f'{split}_images'] = len(images)
                
                if len(images) == 0:
                    validation_results['warnings'].append(f"No images found in {split} split")
                    LOGGER.warning(f"⚠️  No images found in {split} split")
                
                # Check for labels directory
                labels_path = split_path.parent / 'labels'
                if labels_path.exists():
                    label_files = list(labels_path.glob('*.txt'))
                    LOGGER.info(f"   Labels found: {len(label_files)}")
                    validation_results['info'][f'{split}_labels'] = len(label_files)
                    
                    if len(label_files) < len(images):
                        validation_results['warnings'].append(
                            f"{split} split: {len(images)} images but only {len(label_files)} labels"
                        )
                        LOGGER.warning(f"⚠️  {split}: {len(images)} images but only {len(label_files)} labels")
                else:
                    validation_results['warnings'].append(f"Labels directory not found for {split} split")
                    LOGGER.warning(f"⚠️  Labels directory not found for {split} split")
            else:
                validation_results['errors'].append(f"{split} split path does not exist: {split_path}")
                LOGGER.error(f"❌ {split} split path does not exist: {split_path}")
                validation_results['valid'] = False
    
    # Log class information
    LOGGER.info(f"Number of classes: {data_config['nc']}")
    LOGGER.info(f"Class names: {data_config['names']}")
    
    LOGGER.info("=" * 80)
    return validation_results


def validate_model_file(model_file: Path) -> bool:
    """Validate model file exists and is loadable"""
    LOGGER.info("=" * 80)
    LOGGER.info("MODEL VALIDATION")
    LOGGER.info("=" * 80)
    
    LOGGER.info(f"Model file path: {model_file}")
    LOGGER.info(f"Model file exists: {model_file.exists()}")
    
    if model_file.exists():
        LOGGER.info(f"Model file size: {model_file.stat().st_size / (1024**2):.2f} MB")
        LOGGER.info("✅ Model file found")
        return True
    else:
        LOGGER.info("⚠️  Model file not found - will be downloaded")
        return False


def log_training_config(training_config, roboflow_config):
    """Log complete training configuration"""
    LOGGER.info("=" * 80)
    LOGGER.info("TRAINING CONFIGURATION")
    LOGGER.info("=" * 80)
    
    # Training config
    LOGGER.info(f"Model Name: {training_config.model_name.value}")
    LOGGER.info(f"Epochs: {training_config.epochs}")
    LOGGER.info(f"Batch Size: {training_config.batch_size}")
    LOGGER.info(f"Image Size: {training_config.img_size}")
    LOGGER.info(f"Device: {training_config.device.value}")
    LOGGER.info(f"Workers: {training_config.workers}")
    
    # Optimizer settings
    LOGGER.info(f"Optimizer: {training_config.optimizer.value}")
    LOGGER.info(f"Learning Rate (lr0): {training_config.lr0}")
    LOGGER.info(f"Final LR Factor (lrf): {training_config.lrf}")
    LOGGER.info(f"Momentum: {training_config.momentum}")
    LOGGER.info(f"Weight Decay: {training_config.weight_decay}")
    
    # Warmup settings
    LOGGER.info(f"Warmup Epochs: {training_config.warmup_epochs}")
    LOGGER.info(f"Warmup Momentum: {training_config.warmup_momentum}")
    
    # Loss gains
    LOGGER.info(f"Box Gain: {training_config.box_gain}")
    LOGGER.info(f"Class Gain: {training_config.cls_gain}")
    LOGGER.info(f"DFL Gain: {training_config.dfl_gain}")
    
    # Training behavior
    LOGGER.info(f"Patience: {training_config.patience}")
    LOGGER.info(f"Cache: {training_config.cache}")
    LOGGER.info(f"Plots: {training_config.plots}")
    LOGGER.info(f"Verbose: {training_config.verbose}")
    LOGGER.info(f"Seed: {training_config.seed}")
    LOGGER.info(f"Deterministic: {training_config.deterministic}")
    LOGGER.info(f"AMP: {training_config.amp}")
    
    # Paths
    LOGGER.info(f"Pretrain Dir: {training_config.pretrain_dir.absolute()}")
    LOGGER.info(f"Project Dir: {training_config.project_dir.absolute()}")
    
    # Roboflow config
    LOGGER.info(f"Dataset Version: {roboflow_config.version}")
    LOGGER.info(f"Data YAML Path: {roboflow_config.data_yaml_path}")
    
    LOGGER.info("=" * 80)

def safe_train_with_debugging(model, data_yaml_path, LOGGER, **train_kwargs):
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
            LOGGER.error("🔍 DEBUGGING LABEL FORMAT ERROR")
            LOGGER.error("=" * 80)
            LOGGER.error("This error typically means label files have wrong format.")
            LOGGER.error("Running deep label inspection...")
            
            # Try to load dataset manually to find the problematic file
            try:
                from ultralytics.data.dataset import YOLODataset
                from ultralytics.data.augment import Compose, Format
                
                import yaml
                with open(data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                dataset_root = Path(data_yaml_path).parent
                train_path = (dataset_root / data['train']).resolve()
                
                LOGGER.info(f"Attempting to load each image individually...")
                LOGGER.info(f"Train path: {train_path}")
                
                # Try to identify which file causes the issue
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
                            LOGGER.info(f"  ✅ Checked {idx}/{len(dataset)} images")
                    except Exception as item_error:
                        img_file = dataset.im_files[idx]
                        LOGGER.error(f"❌ FOUND PROBLEMATIC FILE: {img_file}")
                        LOGGER.error(f"   Error: {item_error}")
                        
                        # Check corresponding label
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

def debug_and_fix_labels(data_yaml_path: Path, fix: bool = False) -> dict:
    """
    Debug segmentation labels and optionally fix common issues
    
    Args:
        data_yaml_path: Path to data.yaml
        LOGGER: LOGGER instance
        fix: If True, automatically fix issues in-place
    """
    import yaml
    import numpy as np
    import shutil
    from datetime import datetime
    
    LOGGER.info("=" * 80)
    LOGGER.info("LABEL FORMAT DEBUG & FIX")
    LOGGER.info("=" * 80)
    
    issues_found = {
        'malformed': [],
        'empty': [],
        'wrong_dimensions': [],
        'fixed': []
    }
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_root = data_yaml_path
    
    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
        
        LOGGER.info(f"\n🔍 Checking {split} labels...")
        split_path = (dataset_root / data_config[split]).resolve()
        
        # Find labels directory
        if 'images' in str(split_path):
            labels_path = Path(str(split_path).replace('images', 'labels'))
        else:
            labels_path = split_path.parent / 'labels' / split_path.name
            
        if not labels_path.exists():
            LOGGER.warning(f"⚠️  Labels directory not found: {labels_path}")
            continue
        
        label_files = sorted(labels_path.glob('*.txt'))
        LOGGER.info(f"Found {len(label_files)} label files")
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                fixed_lines = []
                file_has_issues = False
                
                for line_num, line in enumerate(lines, 1):
                    original_line = line
                    line = line.strip()
                    
                    if not line:
                        continue
                    
                    parts = line.split()
                    
                    # Basic validation
                    if len(parts) < 7:  # class_id + at least 3 points (6 coords)
                        LOGGER.error(f"❌ {label_file.name}:{line_num} - Too few values: {len(parts)}")
                        issues_found['malformed'].append(f"{label_file.name}:{line_num}")
                        file_has_issues = True
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                        
                        # Check for common issues
                        
                        # Issue 1: Odd number of coordinates
                        if len(coords) % 2 != 0:
                            LOGGER.error(f"❌ {label_file.name}:{line_num} - Odd coordinates: {len(coords)}")
                            issues_found['wrong_dimensions'].append(f"{label_file.name}:{line_num}")
                            file_has_issues = True
                            continue
                        
                        # Issue 2: Check if coordinates are normalized
                        if np.any(coords < 0) or np.any(coords > 1):
                            LOGGER.warning(f"⚠️  {label_file.name}:{line_num} - Coordinates not normalized")
                            if fix:
                                # Clip to valid range
                                coords = np.clip(coords, 0, 1)
                                LOGGER.info(f"   → Fixed by clipping to [0, 1]")
                                issues_found['fixed'].append(f"{label_file.name}:{line_num} - clipped coords")
                                file_has_issues = True
                        
                        # Issue 3: Check for duplicate consecutive points
                        points = coords.reshape(-1, 2)
                        if len(points) > 1:
                            diffs = np.diff(points, axis=0)
                            duplicate_mask = np.all(np.abs(diffs) < 1e-6, axis=1)
                            if np.any(duplicate_mask):
                                LOGGER.warning(f"⚠️  {label_file.name}:{line_num} - Has duplicate consecutive points")
                                if fix:
                                    # Remove duplicates
                                    mask = np.concatenate([[True], ~duplicate_mask])
                                    points = points[mask]
                                    coords = points.flatten()
                                    LOGGER.info(f"   → Removed duplicates: {len(points)} points remain")
                                    issues_found['fixed'].append(f"{label_file.name}:{line_num} - removed duplicates")
                                    file_has_issues = True
                        
                        # Issue 4: Check minimum points
                        if len(coords) < 6:  # Less than 3 points
                            LOGGER.error(f"❌ {label_file.name}:{line_num} - < 3 points: {len(coords)//2}")
                            issues_found['empty'].append(f"{label_file.name}:{line_num}")
                            file_has_issues = True
                            continue
                        
                        # Reconstruct line if fixed
                        if fix and file_has_issues:
                            fixed_line = f"{class_id} " + " ".join(f"{c:.6f}" for c in coords) + "\n"
                            fixed_lines.append(fixed_line)
                        else:
                            fixed_lines.append(original_line)
                            
                    except ValueError as e:
                        LOGGER.error(f"❌ {label_file.name}:{line_num} - Parse error: {e}")
                        issues_found['malformed'].append(f"{label_file.name}:{line_num}")
                        file_has_issues = True
                        continue
                
                # Write fixed file if needed
                if fix and file_has_issues and fixed_lines:
                    # Backup original
                    backup_path = label_file.with_suffix('.txt.backup')
                    if not backup_path.exists():
                        shutil.copy(label_file, backup_path)
                        LOGGER.info(f"   💾 Backed up to {backup_path.name}")
                    
                    # Write fixed version
                    with open(label_file, 'w') as f:
                        f.writelines(fixed_lines)
                    LOGGER.info(f"   ✅ Fixed {label_file.name}")
                    
            except Exception as e:
                LOGGER.error(f"❌ {label_file.name} - Error: {e}")
                LOGGER.debug(traceback.format_exc())
                issues_found['malformed'].append(str(label_file.name))
    
    # Summary
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("DEBUG SUMMARY")
    LOGGER.info("=" * 80)
    
    total_issues = sum(len(v) for v in issues_found.values()) - len(issues_found['fixed'])
    
    if total_issues == 0:
        LOGGER.info("✅ All labels appear valid!")
    else:
        LOGGER.warning(f"⚠️  Found issues in {total_issues} label lines")
        for issue_type, issue_list in issues_found.items():
            if issue_list and issue_type != 'fixed':
                LOGGER.warning(f"  {issue_type}: {len(issue_list)}")
                for issue in issue_list[:5]:
                    LOGGER.warning(f"    - {issue}")
                if len(issue_list) > 5:
                    LOGGER.warning(f"    ... and {len(issue_list) - 5} more")
    
    if issues_found['fixed']:
        LOGGER.info(f"✅ Fixed {len(issues_found['fixed'])} issues")
    
    LOGGER.info("=" * 80)
    
    return issues_found

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
        LOGGER.info(f"✅ Pretrain directory: {training_config.pretrain_dir.absolute()}")
        
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
        critical_issues = label_issues['corrupted_files'] + label_issues['invalid_format']
        if critical_issues:
            LOGGER.error(f"❌ Found {len(critical_issues)} critical label issues that must be fixed!")
            LOGGER.error("Training cannot proceed with corrupted labels.")
            raise ValueError(f"Critical label format issues detected in {len(critical_issues)} files")
        
        if label_issues['empty_polygons']:
            LOGGER.warning(f"⚠️  Found {len(label_issues['empty_polygons'])} polygons with < 3 points")
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
        LOGGER.info("✅ Model loaded successfully")
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
        LOGGER.info(f"🚀 STARTING TRAINING - {training_config.epochs} EPOCHS")
        LOGGER.info("=" * 80)
        
        train_start_time = datetime.now()
        LOGGER.info(f"Training start time: {train_start_time}")
        
        results = safe_train_with_debugging(
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
        LOGGER.info("✅ TRAINING COMPLETE")
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
        
        # Validate the best model
        if best_model_path.exists():
            LOGGER.info("=" * 80)
            LOGGER.info("📊 VALIDATING BEST MODEL")
            LOGGER.info("=" * 80)
            
            best_model = YOLO(best_model_path)
            metrics = best_model.val()
            
            LOGGER.info("Validation Metrics:")
            LOGGER.info(f"  mAP50: {metrics.seg.map50:.4f}")
            LOGGER.info(f"  mAP50-95: {metrics.seg.map:.4f}")
            LOGGER.info(f"  Precision: {metrics.seg.mp:.4f}")
            LOGGER.info(f"  Recall: {metrics.seg.mr:.4f}")
            
            # Log per-class metrics if available
            if hasattr(metrics.seg, 'ap_class_index'):
                LOGGER.info("Per-class mAP50:")
                for idx, ap in zip(metrics.seg.ap_class_index, metrics.seg.ap50):
                    LOGGER.info(f"  Class {idx}: {ap:.4f}")
        else:
            LOGGER.warning("⚠️  Best model not found - training may have been interrupted")
        
        LOGGER.info("=" * 80)
        LOGGER.info("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        LOGGER.info("=" * 80)
        
        return results, best_model_path
        
    except FileNotFoundError as e:
        LOGGER.error("=" * 80)
        LOGGER.error(f"❌ FILE NOT FOUND ERROR at stage: {training_stage}")
        LOGGER.error("=" * 80)
        LOGGER.error(f"Error message: {e}")
        LOGGER.debug(f"Full traceback:\n{traceback.format_exc()}")
        raise
        
    except MemoryError as e:
        LOGGER.error("=" * 80)
        LOGGER.error(f"❌ MEMORY ERROR at stage: {training_stage}")
        LOGGER.error("=" * 80)
        LOGGER.error(f"Error message: {e}")
        
        # Log memory state
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
        LOGGER.warning(f"⚠️  TRAINING INTERRUPTED BY USER at stage: {training_stage}")
        LOGGER.warning("=" * 80)
        raise
        
    except Exception as e:
        LOGGER.error("=" * 80)
        LOGGER.error(f"❌ UNEXPECTED ERROR at stage: {training_stage}")
        LOGGER.error("=" * 80)
        LOGGER.error(f"Error type: {type(e).__name__}")
        LOGGER.error(f"Error message: {e}")
        LOGGER.error(f"\nFull traceback:")
        LOGGER.error(traceback.format_exc())
        
        # Log additional context
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