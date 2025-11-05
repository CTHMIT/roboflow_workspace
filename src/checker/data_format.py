"""Dataset validation and format checking module

This module provides comprehensive validation for YOLO segmentation datasets,
including dataset structure validation, label format checking, and integrity verification.
"""

import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import yaml
import traceback
from typing import Optional, Dict, List, Tuple
from setup.config import RoboflowConfig
from utils.logger import LOGGER

def is_float(x: str) -> bool:
    """Check if string can be converted to float"""
    try:
        float(x)
        return True
    except Exception:
        return False


def is_int(x: str) -> bool:
    """Check if string can be converted to int"""
    try:
        int(x)
        return True
    except Exception:
        return False


def check_image_valid(img_path: Path) -> Tuple[bool, str]:
    """Check if image can be opened and is valid with proper dimensions
    
    Args:
        img_path: Path to image file
        
    Returns:
        Tuple of (is_valid, message_or_dimensions)
    """
    try:
        img = Image.open(img_path)
        img.verify()
        
        img = Image.open(img_path)
        width, height = img.size
        
        if width <= 0 or height <= 0:
            return False, f"Invalid dimensions: {width}x{height}"
        if width > 50000 or height > 50000:
            return False, f"Suspiciously large dimensions: {width}x{height}"
        
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 100:
            return False, f"Extreme aspect ratio: {width}x{height} (ratio: {aspect_ratio:.1f})"
            
        return True, (width, height)
    except Exception as e:
        return False, str(e)


def check_coordinates_reshapeable(coords: List[float]) -> Tuple[bool, str]:
    """Check if coordinates can be safely reshaped without dimension errors
    
    Args:
        coords: List of coordinate values
        
    Returns:
        Tuple of (can_reshape, result_or_error_message)
    """
    try:
        if len(coords) % 2 != 0:
            return False, "Odd number of coordinates"
        
        coords_array = np.array(coords, dtype=np.float32)
        
        if np.any(np.isnan(coords_array)):
            return False, "Contains NaN values"
        if np.any(np.isinf(coords_array)):
            return False, "Contains Inf values"
        
        try:
            points = coords_array.reshape(-1, 2)
        except ValueError as e:
            return False, f"Cannot reshape to (N,2): {str(e)}"
        
        if points.shape[1] != 2:
            return False, f"Reshape produced wrong shape: {points.shape}"
        
        if len(points) < 3:
            return False, f"Less than 3 points after reshape: {len(points)}"
        
        return True, points
        
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def check_polygon_area(coords: List[float]) -> Tuple[bool, str]:
    """Check if polygon has non-zero area with proper dimension handling
    
    Args:
        coords: List of coordinate values
        
    Returns:
        Tuple of (valid_area, area_or_error_message)
    """
    try:
        can_reshape, result = check_coordinates_reshapeable(coords)
        if not can_reshape:
            return False, result
        
        points = result
        
        x = points[:, 0]
        y = points[:, 1]
        
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        if area < 1e-6:
            return False, f"Zero/near-zero area: {area}"
        
        return True, area
        
    except Exception as e:
        return False, f"Area calculation error: {str(e)}"


def validate_coordinates_with_image(coords: List[float], img_width: int, img_height: int) -> Tuple[bool, str]:
    """Validate that normalized coordinates work correctly with image dimensions
    
    Args:
        coords: List of normalized coordinate values
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        coords_array = np.array(coords, dtype=np.float32).reshape(-1, 2)
        
        pixel_coords = coords_array.copy()
        pixel_coords[:, 0] *= img_width
        pixel_coords[:, 1] *= img_height
        
        if np.any(pixel_coords[:, 0] < -1) or np.any(pixel_coords[:, 0] > img_width + 1):
            return False, "X coordinates map outside image bounds"
        if np.any(pixel_coords[:, 1] < -1) or np.any(pixel_coords[:, 1] > img_height + 1):
            return False, "Y coordinates map outside image bounds"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Coordinate validation error: {str(e)}"

def validate_dataset(data_yaml_path: Path, roboflow_config: Optional[RoboflowConfig] = None) -> Dict:
    """Validate dataset structure and return detailed information
    
    This function validates the dataset structure including:
    - data.yaml existence and parsing
    - Required fields (train, val, nc, names)
    - Dataset paths existence
    - Image and label counts
    - Mismatches between images and labels
    
    Args:
        data_yaml_path: Path to data.yaml file
        roboflow_config: Optional Roboflow configuration for additional context
        
    Returns:
        Dictionary with validation results including:
        - valid: bool indicating if dataset is valid
        - errors: List of error messages
        - warnings: List of warning messages
        - info: Dictionary with dataset statistics
    """
    LOGGER.info("=" * 80)
    LOGGER.info("DATASET STRUCTURE VALIDATION")
    LOGGER.info("=" * 80)
    
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    if not data_yaml_path.exists():
        validation_results['valid'] = False
        validation_results['errors'].append(f"data.yaml not found at: {data_yaml_path}")
        LOGGER.error(f"data.yaml not found at: {data_yaml_path}")
        return validation_results
    
    LOGGER.info(f"data.yaml found at: {data_yaml_path}")
    
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        LOGGER.info(f"data.yaml parsed successfully")
        LOGGER.debug(f"data.yaml contents: {data_config}")
    except Exception as e:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Failed to parse data.yaml: {e}")
        LOGGER.error(f"Failed to parse data.yaml: {e}")
        LOGGER.debug(traceback.format_exc())
        return validation_results
    
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in data_config:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Missing required field in data.yaml: {field}")
            LOGGER.error(f"Missing required field: {field}")
        else:
            LOGGER.info(f"Field '{field}': {data_config[field]}")
    
    if not validation_results['valid']:
        return validation_results
    
    dataset_root = data_yaml_path
    
    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
            
        split_path = (dataset_root / data_config[split]).resolve()
        if split_path.exists():

            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            images = []
            for ext in image_extensions:
                images.extend(list(split_path.rglob(f'*{ext}')))
            
            LOGGER.info(f"{split} split found: {split_path}")
            LOGGER.info(f"   Images found: {len(images)}")
            validation_results['info'][f'{split}_images'] = len(images)
            
            if len(images) == 0:
                validation_results['warnings'].append(f"No images found in {split} split")
                LOGGER.warning(f"No images found in {split} split")
            
            labels_path = split_path.parent / 'labels' / split_path.name
            if not labels_path.exists():
                if 'images' in str(split_path):
                    labels_path = Path(str(split_path).replace('images', 'labels'))
            
            if labels_path.exists():
                label_files = list(labels_path.glob('*.txt'))
                LOGGER.info(f"   Labels found: {len(label_files)}")
                validation_results['info'][f'{split}_labels'] = len(label_files)
                
                if len(label_files) < len(images):
                    validation_results['warnings'].append(
                        f"{split} split: {len(images)} images but only {len(label_files)} labels"
                    )
                    LOGGER.warning(f"{split}: {len(images)} images but only {len(label_files)} labels")
            else:
                validation_results['warnings'].append(f"Labels directory not found for {split} split")
                LOGGER.warning(f"Labels directory not found for {split} split")
        else:
            validation_results['errors'].append(f"{split} split path does not exist: {split_path}")
            LOGGER.error(f"{split} split path does not exist: {split_path}")
            validation_results['valid'] = False
    
    LOGGER.info(f"Number of classes: {data_config['nc']}")
    LOGGER.info(f"Class names: {data_config['names']}")
    validation_results['info']['num_classes'] = data_config['nc']
    validation_results['info']['class_names'] = data_config['names']
    
    LOGGER.info("=" * 80)
    return validation_results

def validate_folder_structure(
    folder_path: Path,
    folder_name: str = "dataset"
) -> Dict:
    """Validate image-label pairs and file integrity for a single folder
    
    Args:
        folder_path: Path to dataset folder (e.g., train, val, test)
        folder_name: Name of the folder for logging purposes
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'missing_labels': [],
        'missing_images': [],
        'corrupt_images': [],
        'empty_labels': [],
        'bad_format': [],
        'reshape_errors': [],
        'dimension_issues': [],
        'class_distribution': {},
        'total_annotations': 0,
        'valid_pairs': 0
    }
    
    # Define paths
    IMG_DIR = folder_path / 'images'
    LBL_DIR = folder_path / 'labels'
    
    if not IMG_DIR.exists() or not LBL_DIR.exists():
        LOGGER.warning(f"Skipping {folder_name}: missing images or labels directory")
        return stats
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"VALIDATING: {folder_name.upper()}")
    LOGGER.info(f"{'='*60}")
    
    # Get all image and label files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
    # Check 1: Match images with labels
    LOGGER.info(f"\n🔍 Checking file pairs...")
    
    img_files = set()
    for ext in image_extensions:
        for img_path in IMG_DIR.glob(f"*{ext}"):
            img_files.add(img_path.stem)
    
    lbl_files = {lbl.stem for lbl in LBL_DIR.glob("*.txt")}
    
    LOGGER.info(f"   Images: {len(img_files)}")
    LOGGER.info(f"   Labels: {len(lbl_files)}")
    
    missing_labels = img_files - lbl_files
    missing_images = lbl_files - img_files
    
    if missing_labels:
        LOGGER.warning(f"\nImages without labels ({len(missing_labels)}):")
        for name in sorted(list(missing_labels)[:10]):  # Show first 10
            LOGGER.warning(f"   - {name}")
            stats['missing_labels'].append(name)
        if len(missing_labels) > 10:
            LOGGER.warning(f"   ... and {len(missing_labels) - 10} more")
    
    if missing_images:
        LOGGER.warning(f"\nLabels without images ({len(missing_images)}):")
        for name in sorted(list(missing_images)[:10]):
            LOGGER.warning(f"   - {name}")
            stats['missing_images'].append(name)
        if len(missing_images) > 10:
            LOGGER.warning(f"   ... and {len(missing_images) - 10} more")
    
    # Check 2: Image validity and dimensions
    LOGGER.info(f"\n🖼️  Checking image integrity and dimensions...")
    corrupt_count = 0
    image_dimensions = {}  # Store for cross-validation with labels
    
    for img_stem in sorted(img_files):
        img_path = None
        for ext in image_extensions:
            candidate = IMG_DIR / f"{img_stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path:
            valid, info = check_image_valid(img_path)
            if not valid:
                LOGGER.info(f"   Corrupt/Invalid image: {img_path.name} - {info}")
                stats['corrupt_images'].append((img_path.name, info))
                corrupt_count += 1
            else:
                # Store dimensions for later validation
                image_dimensions[img_stem] = info
    
    if corrupt_count == 0:
        LOGGER.info(f"   All {len(img_files)} images are valid with proper dimensions")
    
    # Check 3: Label format validation with dimension checks
    LOGGER.info(f"\n📝 Checking label format and coordinate dimensions...")
    bad_files = []
    reshape_issues = []
    
    for lbl_path in sorted(LBL_DIR.glob("*.txt")):
        img_stem = lbl_path.stem
        img_dims = image_dimensions.get(img_stem)
        
        with lbl_path.open() as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            # Empty label (background image) - valid
            stats['empty_labels'].append(lbl_path.name)
            continue
        
        stats['total_annotations'] += len(lines)
        
        for li, line in enumerate(lines):
            parts = line.split()
            
            # Check minimum length (class_id + at least 3 points = 7 values)
            if len(parts) < 7:
                bad_files.append((lbl_path.name, f"line {li+1}: too short ({len(parts)} values, need ≥7)"))
                continue
            
            cls_id = parts[0]
            coords = parts[1:]
            
            # Check class ID is valid integer
            if not is_int(cls_id):
                bad_files.append((lbl_path.name, f"line {li+1}: invalid class_id '{cls_id}' (must be integer)"))
                continue
            
            cls_id_int = int(cls_id)
            if cls_id_int < 0:
                bad_files.append((lbl_path.name, f"line {li+1}: negative class_id {cls_id_int}"))
                continue
            
            # Track class distribution
            stats['class_distribution'][cls_id_int] = stats['class_distribution'].get(cls_id_int, 0) + 1
            
            # Check all coordinates are floats
            if not all(is_float(v) for v in coords):
                bad_files.append((lbl_path.name, f"line {li+1}: non-numeric coordinate(s)"))
                continue
            
            # Check even number of coordinates (x,y pairs)
            if len(coords) % 2 != 0:
                bad_files.append((lbl_path.name, f"line {li+1}: odd number of coords ({len(coords)})"))
                continue
            
            # Check minimum 3 points (6 coordinates)
            if len(coords) < 6:
                bad_files.append((lbl_path.name, f"line {li+1}: <3 points ({len(coords)//2} points)"))
                continue
            
            # Check coordinates in range [0,1]
            floats = list(map(float, coords))
            out_of_range = [(i, v) for i, v in enumerate(floats) if v < 0 or v > 1]
            if out_of_range:
                bad_files.append((lbl_path.name, f"line {li+1}: coords out of [0,1] range: {out_of_range[:3]}"))
                continue
            
            # CRITICAL: Check if coordinates can be reshaped (prevents axis remapping errors)
            can_reshape, result = check_coordinates_reshapeable(floats)
            if not can_reshape:
                reshape_issues.append((lbl_path.name, f"line {li+1}: RESHAPE ERROR - {result}"))
                bad_files.append((lbl_path.name, f"line {li+1}: dimension/reshape error - {result}"))
                continue
            
            # Check polygon area with safe dimension handling
            valid_area, area_result = check_polygon_area(floats)
            if not valid_area:
                bad_files.append((lbl_path.name, f"line {li+1}: {area_result}"))
                continue
            
            # Cross-validate with image dimensions if available
            if img_dims:
                img_width, img_height = img_dims
                coords_valid, msg = validate_coordinates_with_image(floats, img_width, img_height)
                if not coords_valid:
                    bad_files.append((lbl_path.name, f"line {li+1}: {msg}"))
                    stats['dimension_issues'].append((lbl_path.name, f"line {li+1}: {msg}"))
    
    # Report results
    if reshape_issues:
        LOGGER.info(f"\n🚨 CRITICAL: Found {len(reshape_issues)} RESHAPE/DIMENSION errors (will cause training to crash):")
        for name, reason in reshape_issues[:10]:
            LOGGER.info(f"   - {name}: {reason}")
        if len(reshape_issues) > 10:
            LOGGER.info(f"   ... and {len(reshape_issues) - 10} more reshape errors")
        stats['reshape_errors'] = reshape_issues
    
    if bad_files:
        LOGGER.info(f"\nFound {len(bad_files)} total format/validation issues:")
        for name, reason in bad_files[:20]:  # Show first 20
            LOGGER.info(f"   - {name}: {reason}")
        if len(bad_files) > 20:
            LOGGER.info(f"   ... and {len(bad_files) - 20} more issues")
        stats['bad_format'] = bad_files
    else:
        LOGGER.info(f"   All label files have correct format and dimensions")
    
    # Summary
    stats['valid_pairs'] = len(img_files & lbl_files)
    
    LOGGER.info(f"\n📈 Summary for {folder_name}:")
    LOGGER.info(f"   Valid image-label pairs: {stats['valid_pairs']}")
    LOGGER.info(f"   Total annotations: {stats['total_annotations']}")
    LOGGER.info(f"   Background images (empty labels): {len(stats['empty_labels'])}")
    LOGGER.info(f"   Images missing labels: {len(stats['missing_labels'])}")
    LOGGER.info(f"   Labels missing images: {len(stats['missing_images'])}")
    LOGGER.info(f"   Corrupt images: {len(stats['corrupt_images'])}")
    LOGGER.info(f"   🚨 RESHAPE/DIMENSION errors: {len(stats['reshape_errors'])}")
    LOGGER.info(f"   Coordinate-dimension mismatches: {len(stats['dimension_issues'])}")
    LOGGER.info(f"   Other format errors: {len(stats['bad_format'])}")
    
    if stats['class_distribution']:
        LOGGER.info(f"\n   Class distribution:")
        for cls_id in sorted(stats['class_distribution'].keys()):
            count = stats['class_distribution'][cls_id]
            LOGGER.info(f"      Class {cls_id}: {count} instances")
    
    # Final verdict
    critical_issues = (stats['reshape_errors'] or stats['corrupt_images'] or 
                    stats['missing_labels'] or stats['missing_images'])
    
    if not critical_issues and not stats['bad_format']:
        LOGGER.info(f"\n{folder_name.upper()} dataset is READY for training!")
    elif critical_issues:
        LOGGER.info(f"\n🚨 {folder_name.upper()} dataset has CRITICAL issues that WILL cause training to crash!")
        LOGGER.info(f"   Fix reshape/dimension errors before training!")
    else:
        LOGGER.info(f"\n{folder_name.upper()} dataset has issues that should be fixed")
    
    return stats


def validate_segmentation_labels(
    data_yaml_path: Path,
    roboflow_config: Optional[RoboflowConfig] = None
) -> Dict:
    """Comprehensive segmentation label validation
    
    This function performs detailed validation of segmentation labels including:
    - Label file format
    - Coordinate validity
    - Polygon area
    - Image-label matching
    - Dimension compatibility
    
    Args:
        data_yaml_path: Path to data.yaml file
        roboflow_config: Optional Roboflow configuration
        
    Returns:
        Dictionary with validation results for all splits
    """
    LOGGER.info("=" * 80)
    LOGGER.info("SEGMENTATION LABEL VALIDATION")
    LOGGER.info("=" * 80)
    
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        LOGGER.error(f"Failed to load data.yaml: {e}")
        return {}
    
    dataset_root = data_yaml_path
    all_stats = {}
    
    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
            
        split_path = (dataset_root / data_config[split]).resolve()
        
        if not split_path.exists():
            LOGGER.warning(f"Skipping {split}: path does not exist ({split_path})")
            continue
        
        # Get the parent folder that contains images and labels
        folder_path = split_path.parent
        stats = validate_folder_structure(folder_path, folder_name=split)
        all_stats.update(stats)
    
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("SEGMENTATION LABEL VALIDATION COMPLETE!")
    LOGGER.info('='*80)
    
    return all_stats


def validate_dataset_complete(
    roboflow_config: RoboflowConfig,
    perform_detailed_validation: bool = True
) -> Dict:
    """Perform complete dataset validation using configuration
    
    This is the main entry point for dataset validation. It performs:
    1. Dataset structure validation (from train.py)
    2. Detailed label validation (from data_format.py) - optional
    
    Args:
        roboflow_config: Roboflow configuration containing dataset paths
        perform_detailed_validation: Whether to perform detailed label validation
        
    Returns:
        Dictionary containing all validation results
    """
    data_yaml_path = roboflow_config.data_yaml_path
    
    LOGGER.info("=" * 80)
    LOGGER.info("STARTING COMPLETE DATASET VALIDATION")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Dataset version: {roboflow_config.version}")
    LOGGER.info(f"Dataset path: {roboflow_config.dataset_path}")
    LOGGER.info(f"Data YAML: {data_yaml_path}")
    LOGGER.info("=" * 80)
    
    results = {}
    
    # Step 1: Validate dataset structure
    structure_validation = validate_dataset(data_yaml_path, roboflow_config)
    results['structure'] = structure_validation
    
    if not structure_validation['valid']:
        LOGGER.error("\nDataset structure validation failed!")
        LOGGER.error("Cannot proceed with detailed validation.")
        return results
    
    # Step 2: Detailed label validation (optional)
    if perform_detailed_validation:
        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("PROCEEDING WITH DETAILED LABEL VALIDATION")
        LOGGER.info("=" * 80)
        
        label_validation = validate_segmentation_labels(data_yaml_path, roboflow_config)
        results['labels'] = label_validation
        
        # Check for critical issues
        has_critical_issues = False
        for split, stats in label_validation.items():
            if stats.get('reshape_errors') or stats.get('corrupt_images'):
                has_critical_issues = True
                break
        
        if has_critical_issues:
            LOGGER.error("\n" + "=" * 80)
            LOGGER.error("🚨 CRITICAL ISSUES DETECTED!")
            LOGGER.error("=" * 80)
            LOGGER.error("The dataset has issues that will cause training to crash.")
            LOGGER.error("Please fix the following before training:")
            LOGGER.error("  - Reshape/dimension errors in labels")
            LOGGER.error("  - Corrupt or invalid images")
            results['ready_for_training'] = False
        else:
            LOGGER.info("\n" + "=" * 80)
            LOGGER.info("DATASET VALIDATION PASSED!")
            LOGGER.info("=" * 80)
            results['ready_for_training'] = True
    else:
        results['ready_for_training'] = True
    
    return results


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_from_config(
    config_path: str = "src/config/config.yaml",
    perform_detailed_validation: bool = True
) -> Dict:
    """Validate dataset using configuration file
    
    Args:
        config_path: Path to configuration YAML file
        perform_detailed_validation: Whether to perform detailed label validation
        
    Returns:
        Dictionary containing validation results
    """
    from setup.config import load_config
    
    app_config, env_config = load_config(config_path)
    return validate_dataset_complete(
        app_config.roboflow,
        perform_detailed_validation=perform_detailed_validation
    )


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for dataset validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate YOLO segmentation dataset')
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Skip detailed label validation'
    )
    parser.add_argument(
        '--data-yaml',
        type=str,
        help='Direct path to data.yaml (bypasses config)'
    )
    
    args = parser.parse_args()
    
    if args.data_yaml:
        # Direct validation from data.yaml path
        data_yaml_path = Path(args.data_yaml)
        LOGGER.info(f"Validating dataset from: {data_yaml_path}")
        
        # Structure validation
        structure_results = validate_dataset(data_yaml_path)
        
        # Detailed validation
        if not args.quick and structure_results['valid']:
            label_results = validate_segmentation_labels(data_yaml_path)
    else:
        # Validation from config file
        results = validate_from_config(
            config_path=args.config,
            perform_detailed_validation=not args.quick
        )
        
        if results.get('ready_for_training'):
            LOGGER.info("\nDataset is ready for training!")
            return 0
        else:
            LOGGER.error("\nDataset has critical issues!")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())