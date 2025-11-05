#!/usr/bin/env python3
"""
RealSense 435i + YOLO11 Segmentation
Real-time object detection and segmentation using Intel RealSense 435i camera
"""
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
import pyrealsense2 as rs
from ultralytics import YOLO
from typing import Union, Optional
from setup.config import AppConfig
from utils.logger import LOGGER

class RealSenseYOLO:
    def __init__(
            self, 
            width: int = 640,
            height: int = 480,
            fps: int = 30,
            model_path: Union[str, Path] = 'yolo11-seg.pt', 
            confidence: float = 0.5, 
            iou: float = 0.8, 
            verbose: bool = False,
        ):
        """
        Initialize RealSense camera and YOLO model
        
        Args:
            model_path: Path to YOLO11 segmentation model
            confidence: Confidence threshold for detections
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.confidence = confidence
        self.iou = iou
        self.verbose = verbose
        
        LOGGER.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        self.pipeline: rs.pipeline = rs.pipeline()
        self.config: rs.config = rs.config()
        self.profile: Optional[rs.pipeline_profile] = None
        
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        
        LOGGER.info(f"Using device: {device.get_info(rs.camera_info.name)}")
        
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                
        LOGGER.info("Starting RealSense pipeline...")
        self.pipeline.start(self.config)
        
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        LOGGER.info("Camera ready!")
    
    def get_frames(self):
        """
        Get color and depth frames from RealSense camera
        
        Returns:
            color_image: BGR color image
            depth_image: Depth image
        """
        frames: rs.composite_frame = self.pipeline.wait_for_frames()
        
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        
        color_frame: rs.video_frame = aligned_frames.get_color_frame()
        depth_frame: rs.depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame:
            return None, None
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        
        return color_image, depth_image
    
    def process_frame(self, frame):
        """
        Process frame with YOLO segmentation
        
        Args:
            frame: Input BGR image
            
        Returns:
            annotated_frame: Frame with detections drawn
            results: YOLO results object
        """
        results = self.model(
            frame, 
            conf=self.confidence, 
            iou=self.iou,
            verbose=self.verbose)
        
        annotated_frame = results[0].plot()
        
        return annotated_frame, results[0]
    
    def run(self, verbose:bool = False):
        """
        Main loop for real-time detection
        """
        LOGGER.info("\nStarting real-time detection...")
        LOGGER.info("Press 'q' to quit")
        LOGGER.info("Press 's' to save screenshot")
        
        frame_count = 0
        
        try:
            while True:
                color_image, depth_image = self.get_frames()
                
                if color_image is None:
                    continue
                
                annotated_frame, results = self.process_frame(color_image)
                
                frame_count += 1
                cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                cv2.imshow('RealSense 435i + YOLO11 Segmentation', annotated_frame)
                
                if depth_image is not None:
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow('Depth', depth_colormap)
                
                if verbose == True:
                    if len(results.boxes) > 0:
                        LOGGER.info(f"Frame {frame_count}: Detected {len(results.boxes)} objects")
                        for box in results.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            cls_name = results.names[cls_id]
                            LOGGER.info(f"  - {cls_name}: {conf:.2f}")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    LOGGER.info("\nQuitting...")
                    break
                elif key == ord('s'):
                    filename = f"detection_{frame_count}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    LOGGER.info(f"Saved screenshot: {filename}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Stop pipeline and close windows"""
        LOGGER.info("Cleaning up...")
        self.pipeline.stop()
        cv2.destroyAllWindows()


def rs_predict(app_config: AppConfig):
    model_path = Path(app_config.prediction.predict_dir.absolute(), f"{app_config.training.model_name}.pt")
    LOGGER.info(f"Load Model from : {model_path}")
    confidence = app_config.prediction.confidence 
    iou = app_config.prediction.iou
    width = app_config.video.width
    height = app_config.video.height
    fps = app_config.video.fps
    detector = RealSenseYOLO(
        width=width,
        height=height,
        fps=fps,
        model_path=model_path, 
        confidence=confidence, 
        iou=iou)
    detector.run(verbose=app_config.prediction.verbose)


if __name__ == "__main__":
    rs_predict()