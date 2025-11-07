"""
YOLO Prediction Script
Performs inference on images and saves results in Roboflow-compatible format
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Optional
import cv2
from ultralytics import YOLO

from setup.config import AppConfig, load_config
from utils.logger import LOGGER


class YOLOPredictor:
    """YOLO model predictor with Roboflow output format"""
    
    def __init__(self, config: AppConfig):
        """
        Initialize predictor with configuration
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.pred_config = config.prediction
        self.roboflow_config = config.roboflow
        self.model = None
        
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load YOLO model for inference
        
        Args:
            model_path: Path to model weights. If None, uses production model from config
        """
        if model_path is None:
            model_path = Path(self.pred_config.predict_dir) / "best.pt"
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        LOGGER.info(f"Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        LOGGER.info("Model loaded successfully!")
    
    def predict_images(
        self,
        image_dir: str,
        output_version: Optional[int] = None,
        save_visualizations: bool = True,
        save_txt: bool = True,
        file_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    ) -> tuple[Path, List[str]]:
        """
        Run predictions on images in a directory
        
        Args:
            image_dir: Directory containing images to predict
            output_version: Version number for output (default: uses roboflow version from config)
            save_visualizations: Save annotated images
            save_txt: Save YOLO format labels
            file_extensions: Tuple of valid image extensions
            
        Returns:
            Tuple of (output_directory, list_of_predicted_image_paths)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Set output version
        if output_version is None:
            output_version = self.roboflow_config.version
        
        # Create output directory structure
        output_base = Path("datasets") / "predict" / f"v{output_version}"
        output_images = output_base / "images"
        output_labels = output_base / "labels"
        output_visualizations = output_base / "visualizations"
        
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        if save_visualizations:
            output_visualizations.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(list(image_dir.glob(f"*{ext}")))
            image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            LOGGER.warning(f"No images found in {image_dir}")
            return output_base, []
        
        LOGGER.info(f"\nFound {len(image_files)} images to predict")
        LOGGER.info(f"Output directory: {output_base}")
        LOGGER.info(f"{'='*50}\n")
        
        predicted_image_paths = []
        successful_predictions = 0
        failed_predictions = 0
        
        for i, image_file in enumerate(image_files, 1):
            try:
                results = self.model.predict(
                    source=str(image_file),
                    conf=self.pred_config.confidence,
                    iou=self.pred_config.iou,
                    half=self.pred_config.half,
                    device=self.pred_config.device,
                    verbose=self.pred_config.verbose
                )
                
                result = results[0]
                
                output_image_path = output_images / image_file.name
                shutil.copy2(image_file, output_image_path)
                predicted_image_paths.append(str(output_image_path))
                
                if save_txt and result.masks is not None:
                    label_file = output_labels / f"{image_file.stem}.txt"
                    self._save_yolo_labels(result, label_file)
                
                if save_visualizations:
                    vis_path = output_visualizations / image_file.name
                    annotated = result.plot()
                    cv2.imwrite(str(vis_path), annotated)
                
                successful_predictions += 1
                LOGGER.info(f"[{i}/{len(image_files)}] Predicted: {image_file.name} "
                          f"- {len(result.boxes) if result.boxes is not None else 0} objects detected")
                
            except Exception as e:
                failed_predictions += 1
                LOGGER.error(f"[{i}/{len(image_files)}] Failed: {image_file.name}")
                LOGGER.error(f"    Error: {str(e)}")
        
        # Save predicted images list for easy upload
        self._save_image_list(predicted_image_paths, output_base / "predicted_images.txt")
        
        LOGGER.info(f"\n{'='*50}")
        LOGGER.info(f"Prediction complete!")
        LOGGER.info(f"Successful: {successful_predictions}")
        LOGGER.info(f"Failed: {failed_predictions}")
        LOGGER.info(f"Output saved to: {output_base}")
        LOGGER.info(f"{'='*50}\n")
        
        return output_base, predicted_image_paths
    
    def _save_yolo_labels(self, result, label_file: Path) -> None:
        """
        Save predictions in YOLO segmentation format
        
        Format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
        All coordinates are normalized (0-1)
        """
        if result.masks is None or len(result.masks) == 0:
            label_file.write_text("")
            return
        
        img_h, img_w = result.orig_shape
        
        with open(label_file, 'w') as f:
            for mask, box, cls in zip(result.masks.xy, result.boxes, result.boxes.cls):
                class_id = int(cls.item())
                
                normalized_coords = []
                for x, y in mask:
                    normalized_coords.append(f"{x/img_w:.6f}")
                    normalized_coords.append(f"{y/img_h:.6f}")
                
                # Write: class_id x1 y1 x2 y2 ... xn yn
                line = f"{class_id} " + " ".join(normalized_coords)
                f.write(line + "\n")
    
    def _save_image_list(self, image_paths: List[str], output_file: Path) -> None:
        """Save list of predicted image paths to file"""
        with open(output_file, 'w') as f:
            for path in image_paths:
                f.write(f"{path}\n")
        LOGGER.info(f"Image list saved to: {output_file}")
    
    def predict_single_image(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_visualization: bool = True
    ) -> dict:
        """
        Run prediction on a single image
        
        Args:
            image_path: Path to input image
            output_dir: Optional output directory
            save_visualization: Save annotated image
            
        Returns:
            Dictionary containing prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        results = self.model.predict(
            source=str(image_path),
            conf=self.pred_config.confidence,
            iou=self.pred_config.iou,
            half=self.pred_config.half,
            device=self.pred_config.device,
            verbose=self.pred_config.verbose
        )
        
        result = results[0]
        
        if save_visualization and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            vis_path = output_dir / f"{image_path.stem}_predicted.jpg"
            annotated = result.plot()
            cv2.imwrite(str(vis_path), annotated)
            LOGGER.info(f"Visualization saved to: {vis_path}")
        
        return {
            'boxes': result.boxes,
            'masks': result.masks,
            'classes': result.boxes.cls if result.boxes is not None else None,
            'confidences': result.boxes.conf if result.boxes is not None else None,
            'image_path': str(image_path)
        }


def main():    
    parser = argparse.ArgumentParser(description="Run YOLO predictions on images")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images to predict")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model weights (.pt file). Default: uses predict_dir from config")
    parser.add_argument("--output-version", type=int, default=None, help="Output version number. Default: uses roboflow version from config")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to config file")
    parser.add_argument("--no-visualizations", action="store_true", help="Don't save visualization images")
    parser.add_argument("--no-labels", action="store_true", help="Don't save YOLO format labels")
    
    args = parser.parse_args()
    
    LOGGER.info("=" * 60)
    LOGGER.info("Loading configuration from: {}".format(args.config))
    LOGGER.info("=" * 60)
    
    try:
        config, env_config = load_config(args.config)
    except FileNotFoundError:
        LOGGER.error(f"Configuration file not found: {args.config}")
        LOGGER.info("Please ensure config.yaml exists in the correct location")
        exit(1)
    except Exception as e:
        LOGGER.error(f"Failed to load configuration: {str(e)}")
        exit(1)
    
    if not Path(args.image_dir).exists():
        LOGGER.error(f"Image directory not found: {args.image_dir}")
        exit(1)
    
    LOGGER.info("\nConfiguration:")
    LOGGER.info(f"  Workspace: {config.roboflow.workspace}")
    LOGGER.info(f"  Project: {config.roboflow.project}")
    LOGGER.info(f"  Confidence: {config.prediction.confidence}")
    LOGGER.info(f"  IOU: {config.prediction.iou}")
    LOGGER.info(f"  Output version: {args.output_version or config.roboflow.version}")
    LOGGER.info("")
    
    predictor = YOLOPredictor(config)
    
    try:
        predictor.load_model(model_path=args.model_path)
    except FileNotFoundError as e:
        LOGGER.error(str(e))
        exit(1)
    except Exception as e:
        LOGGER.error(f"Failed to load model: {str(e)}")
        exit(1)
    
    try:
        output_dir, predicted_paths = predictor.predict_images(
            image_dir=args.image_dir,
            output_version=args.output_version,
            save_visualizations=not args.no_visualizations,
            save_txt=not args.no_labels
        )
        
        if predicted_paths:
            LOGGER.info("\n" + "="*60)
            LOGGER.info("SUCCESS - Prediction Complete")
            LOGGER.info("="*60)
            LOGGER.info(f"\nOutput location: {output_dir}")
            LOGGER.info(f"  - Images: {output_dir / 'images'}")
            LOGGER.info(f"  - Labels: {output_dir / 'labels'}")
            if not args.no_visualizations:
                LOGGER.info(f"  - Visualizations: {output_dir / 'visualizations'}")
            
    except Exception as e:
        LOGGER.error(f"\nPrediction failed: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())
        exit(1)

if __name__ == "__main__":
    main()