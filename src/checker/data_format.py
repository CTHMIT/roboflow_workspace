import os
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from setup.config import TrainingConfig, RoboflowConfig
from utils.logger import LOGGER

version = 5
data_path = f'datasets/v{version}'
folders = ['test', 'train', 'valid']
sub_folders = ['images', 'labels']

def is_float(x: str):
    try:
        float(x)
        return True
    except Exception:
        return False

def is_int(x: str):
    try:
        int(x)
        return True
    except Exception:
        return False

def check_image_valid(img_path):
    """Check if image can be opened and is valid with proper dimensions"""
    try:
        img = Image.open(img_path)
        img.verify()  # Verify it's actually an image
        
        # Reopen to get size (verify() closes the file)
        img = Image.open(img_path)
        width, height = img.size
        
        # Check dimensions are positive and reasonable
        if width <= 0 or height <= 0:
            return False, f"Invalid dimensions: {width}x{height}"
        if width > 50000 or height > 50000:
            return False, f"Suspiciously large dimensions: {width}x{height}"
        
        # Check for unusual aspect ratios that might cause issues
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 100:
            return False, f"Extreme aspect ratio: {width}x{height} (ratio: {aspect_ratio:.1f})"
            
        return True, (width, height)
    except Exception as e:
        return False, str(e)

def check_coordinates_reshapeable(coords):
    """Check if coordinates can be safely reshaped without dimension errors"""
    try:
        # Verify we have even number of coordinates
        if len(coords) % 2 != 0:
            return False, "Odd number of coordinates"
        
        # Convert to float array
        coords_array = np.array(coords, dtype=np.float32)
        
        # Check for NaN or Inf values
        if np.any(np.isnan(coords_array)):
            return False, "Contains NaN values"
        if np.any(np.isinf(coords_array)):
            return False, "Contains Inf values"
        
        # Try to reshape to (N, 2) format - this is where axis remapping errors occur
        try:
            points = coords_array.reshape(-1, 2)
        except ValueError as e:
            return False, f"Cannot reshape to (N,2): {str(e)}"
        
        # Verify reshape worked correctly
        if points.shape[1] != 2:
            return False, f"Reshape produced wrong shape: {points.shape}"
        
        if len(points) < 3:
            return False, f"Less than 3 points after reshape: {len(points)}"
        
        return True, points
        
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def check_polygon_area(coords):
    """Check if polygon has non-zero area with proper dimension handling"""
    try:
        # First check if coordinates are reshapeable
        can_reshape, result = check_coordinates_reshapeable(coords)
        if not can_reshape:
            return False, result
        
        points = result
        
        # Shoelace formula for polygon area
        # Using proper indexing to avoid axis remapping issues
        x = points[:, 0]
        y = points[:, 1]
        
        # Calculate area using vectorized operations
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        if area < 1e-6:
            return False, f"Zero/near-zero area: {area}"
        
        return True, area
        
    except Exception as e:
        return False, f"Area calculation error: {str(e)}"

def validate_coordinates_with_image(coords, img_width, img_height):
    """Validate that normalized coordinates work correctly with image dimensions"""
    try:
        coords_array = np.array(coords, dtype=np.float32).reshape(-1, 2)
        
        # Convert normalized coords to pixel coords
        pixel_coords = coords_array.copy()
        pixel_coords[:, 0] *= img_width
        pixel_coords[:, 1] *= img_height
        
        # Check if pixel coordinates are within reasonable bounds
        if np.any(pixel_coords[:, 0] < -1) or np.any(pixel_coords[:, 0] > img_width + 1):
            return False, "X coordinates map outside image bounds"
        if np.any(pixel_coords[:, 1] < -1) or np.any(pixel_coords[:, 1] > img_height + 1):
            return False, "Y coordinates map outside image bounds"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Coordinate validation error: {str(e)}"


def validate_segmentation_labels(data_yaml_path: Path) -> dict:
    """Validate segmentation label files and detect issues"""
    import yaml
    import numpy as np
    
    LOGGER.info("=" * 80)
    LOGGER.info("SEGMENTATION LABEL VALIDATION")
    LOGGER.info("=" * 80)
    
    issues = {
        'corrupted_files': [],
        'empty_polygons': [],
        'invalid_format': [],
        'out_of_bounds': []
    }
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_root = data_yaml_path
    
    for split in ['train', 'val', 'test']:
        if split not in data_config:
            continue
            
        LOGGER.info(f"\nValidating {split} labels...")
        split_path = (dataset_root / data_config[split]).resolve()
        
        # Find labels directory
        if 'images' in str(split_path):
            labels_path = Path(str(split_path).replace('images', 'labels'))
        else:
            labels_path = split_path.parent / 'labels' / split_path.name
        
        
        if not labels_path.exists():
            LOGGER.warning(f"⚠️  Labels directory not found: {labels_path}")
            continue
        
        label_files = list(labels_path.glob('*.txt'))
        LOGGER.info(f"Checking {len(label_files)} label files...")
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    continue
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    
                    # Check format: class_id x1 y1 x2 y2 ... (min 7 values for segmentation)
                    if len(parts) < 7:
                        issues['invalid_format'].append({
                            'file': label_file.name,
                            'line': line_num,
                            'reason': f'Too few coordinates: {len(parts)} values'
                        })
                        LOGGER.warning(f"⚠️  {label_file.name}:{line_num} - Only {len(parts)} values")
                        continue
                    
                    # Check if coordinates are valid numbers
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                    except ValueError as e:
                        issues['corrupted_files'].append({
                            'file': label_file.name,
                            'line': line_num,
                            'reason': f'Invalid number format: {e}'
                        })
                        LOGGER.error(f"❌ {label_file.name}:{line_num} - Invalid numbers")
                        continue
                    
                    # Check if coordinates are normalized (0-1 range)
                    if any(c < 0 or c > 1 for c in coords):
                        issues['out_of_bounds'].append({
                            'file': label_file.name,
                            'line': line_num,
                            'reason': 'Coordinates outside 0-1 range'
                        })
                        LOGGER.warning(f"⚠️  {label_file.name}:{line_num} - Coords out of bounds")
                    
                    # Check if polygon has at least 3 points (6 coordinates)
                    if len(coords) < 6:
                        issues['empty_polygons'].append({
                            'file': label_file.name,
                            'line': line_num,
                            'reason': f'Polygon has < 3 points ({len(coords)//2} points)'
                        })
                        LOGGER.warning(f"⚠️  {label_file.name}:{line_num} - < 3 polygon points")
                    
                    # Check if coordinates come in pairs
                    if len(coords) % 2 != 0:
                        issues['invalid_format'].append({
                            'file': label_file.name,
                            'line': line_num,
                            'reason': 'Odd number of coordinates (need x,y pairs)'
                        })
                        LOGGER.error(f"❌ {label_file.name}:{line_num} - Odd coordinates: {len(coords)}")
                
            except Exception as e:
                issues['corrupted_files'].append({
                    'file': label_file.name,
                    'line': 0,
                    'reason': f'File read error: {e}'
                })
                LOGGER.error(f"❌ {label_file.name} - Cannot read: {e}")
    
    # Summary
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("LABEL VALIDATION SUMMARY")
    LOGGER.info("=" * 80)
    
    total_issues = sum(len(v) for v in issues.values())
    
    if total_issues == 0:
        LOGGER.info("✅ All labels valid!")
    else:
        LOGGER.warning(f"⚠️  Found {total_issues} issues:")
        for issue_type, issue_list in issues.items():
            if issue_list:
                LOGGER.warning(f"  {issue_type}: {len(issue_list)}")
                # Show first few examples
                for issue in issue_list[:3]:
                    LOGGER.warning(f"    - {issue['file']}:{issue['line']} - {issue['reason']}")
                if len(issue_list) > 3:
                    LOGGER.warning(f"    ... and {len(issue_list) - 3} more")
    
    LOGGER.info("=" * 80)
    
    return issues
def main():
    # Check each dataset split
    for folder_name in folders:
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"Checking: {folder_name}")
        LOGGER.info('='*60)
        
        IMG_DIR = Path(f"{data_path}/{folder_name}/{sub_folders[0]}")
        LBL_DIR = Path(f"{data_path}/{folder_name}/{sub_folders[1]}")
        
        if not IMG_DIR.exists():
            LOGGER.warning(f"⚠️  Image directory not found: {IMG_DIR}")
            continue
        if not LBL_DIR.exists():
            LOGGER.warning(f"⚠️  Label directory not found: {LBL_DIR}")
            continue
        
        # Collect all files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        img_files = set()
        for ext in image_extensions:
            img_files.update([f.stem for f in IMG_DIR.glob(f"*{ext}")])
            img_files.update([f.stem for f in IMG_DIR.glob(f"*{ext.upper()}")])
        
        lbl_files = set([f.stem for f in LBL_DIR.glob("*.txt")])
        
        # Statistics
        stats = {
            'total_images': len(img_files),
            'total_labels': len(lbl_files),
            'valid_pairs': 0,
            'missing_labels': [],
            'missing_images': [],
            'empty_labels': [],
            'corrupt_images': [],
            'bad_format': [],
            'dimension_issues': [],
            'reshape_errors': [],
            'total_annotations': 0,
            'class_distribution': {}
        }
        
        # Check 1: Parallel check (image-label pairing)
        LOGGER.info(f"\n📊 Dataset Statistics:")
        LOGGER.info(f"   Images: {len(img_files)}")
        LOGGER.info(f"   Labels: {len(lbl_files)}")
        
        missing_labels = img_files - lbl_files
        missing_images = lbl_files - img_files
        
        if missing_labels:
            LOGGER.warning(f"\n⚠️  Images without labels ({len(missing_labels)}):")
            for name in sorted(list(missing_labels)[:10]):  # Show first 10
                LOGGER.warning(f"   - {name}")
                stats['missing_labels'].append(name)
            if len(missing_labels) > 10:
                LOGGER.warning(f"   ... and {len(missing_labels) - 10} more")
        
        if missing_images:
            LOGGER.warning(f"\n⚠️  Labels without images ({len(missing_images)}):")
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
                candidate = IMG_DIR / f"{img_stem}{ext.upper()}"
                if candidate.exists():
                    img_path = candidate
                    break
            
            if img_path:
                valid, info = check_image_valid(img_path)
                if not valid:
                    LOGGER.info(f"   ❌ Corrupt/Invalid image: {img_path.name} - {info}")
                    stats['corrupt_images'].append((img_path.name, info))
                    corrupt_count += 1
                else:
                    # Store dimensions for later validation
                    image_dimensions[img_stem] = info
        
        if corrupt_count == 0:
            LOGGER.info(f"   ✅ All {len(img_files)} images are valid with proper dimensions")
        
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
            LOGGER.info(f"\n❌ Found {len(bad_files)} total format/validation issues:")
            for name, reason in bad_files[:20]:  # Show first 20
                LOGGER.info(f"   - {name}: {reason}")
            if len(bad_files) > 20:
                LOGGER.info(f"   ... and {len(bad_files) - 20} more issues")
            stats['bad_format'] = bad_files
        else:
            LOGGER.info(f"   ✅ All label files have correct format and dimensions")
        
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
            LOGGER.info(f"\n✅ {folder_name.upper()} dataset is READY for training!")
        elif critical_issues:
            LOGGER.info(f"\n🚨 {folder_name.upper()} dataset has CRITICAL issues that WILL cause training to crash!")
            LOGGER.info(f"   Fix reshape/dimension errors before training!")
        else:
            LOGGER.info(f"\n⚠️  {folder_name.upper()} dataset has issues that should be fixed")

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("✅ Data validation complete!")
    LOGGER.info('='*60)

if __name__ == "__main__":
    main()