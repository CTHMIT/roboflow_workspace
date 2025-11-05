from pathlib import Path
import yaml
import numpy as np
import traceback
import shutil

from utils.logger import LOGGER


def debug_and_fix_labels(data_yaml_path: Path, fix: bool = False) -> dict:
    """
    Debug segmentation labels and optionally fix common issues
    
    Args:
        data_yaml_path: Path to data.yaml
        LOGGER: LOGGER instance
        fix: If True, automatically fix issues in-place
    """
    
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
        
        LOGGER.info(f"\nChecking {split} labels...")
        split_path = (dataset_root / data_config[split]).resolve()
        
        if 'images' in str(split_path):
            labels_path = Path(str(split_path).replace('images', 'labels'))
        else:
            labels_path = split_path.parent / 'labels' / split_path.name
            
        if not labels_path.exists():
            LOGGER.warning(f"Labels directory not found: {labels_path}")
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
                    
                    if len(parts) < 7:  
                        LOGGER.error(f"{label_file.name}:{line_num} - Too few values: {len(parts)}")
                        issues_found['malformed'].append(f"{label_file.name}:{line_num}")
                        file_has_issues = True
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        coords = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                        
                        if len(coords) % 2 != 0:
                            LOGGER.error(f"{label_file.name}:{line_num} - Odd coordinates: {len(coords)}")
                            issues_found['wrong_dimensions'].append(f"{label_file.name}:{line_num}")
                            file_has_issues = True
                            continue
                        
                        # Issue 2: Check if coordinates are normalized
                        if np.any(coords < 0) or np.any(coords > 1):
                            LOGGER.warning(f"{label_file.name}:{line_num} - Coordinates not normalized")
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
                                LOGGER.warning(f"{label_file.name}:{line_num} - Has duplicate consecutive points")
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
                            LOGGER.error(f"{label_file.name}:{line_num} - < 3 points: {len(coords)//2}")
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
                        LOGGER.error(f"{label_file.name}:{line_num} - Parse error: {e}")
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
                    LOGGER.info(f"   Fixed {label_file.name}")
                    
            except Exception as e:
                LOGGER.error(f"{label_file.name} - Error: {e}")
                LOGGER.debug(traceback.format_exc())
                issues_found['malformed'].append(str(label_file.name))
    
    # Summary
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("DEBUG SUMMARY")
    LOGGER.info("=" * 80)
    
    total_issues = sum(len(v) for v in issues_found.values()) - len(issues_found['fixed'])
    
    if total_issues == 0:
        LOGGER.info("All labels appear valid!")
    else:
        LOGGER.warning(f"Found issues in {total_issues} label lines")
        for issue_type, issue_list in issues_found.items():
            if issue_list and issue_type != 'fixed':
                LOGGER.warning(f"  {issue_type}: {len(issue_list)}")
                for issue in issue_list[:5]:
                    LOGGER.warning(f"    - {issue}")
                if len(issue_list) > 5:
                    LOGGER.warning(f"    ... and {len(issue_list) - 5} more")
    
    if issues_found['fixed']:
        LOGGER.info(f"Fixed {len(issues_found['fixed'])} issues")
    
    LOGGER.info("=" * 80)
    
    return issues_found