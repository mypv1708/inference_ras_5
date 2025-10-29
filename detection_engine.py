"""
Main detection engine
Handles the core detection loop and processing pipeline
"""

import time
from ultralytics import YOLO
from typing import Dict, Any

from config import Config
from camera_utils import setup_camera, get_model_info
from silkworm_detection import process_detections
from display_utils import draw_performance_info, draw_scale_bar, display_frame
from performance_monitor import PerformanceMonitor
from object_tracker import SilkwormTracker
from freeze_detection import process_freeze
from overlap_detection import process_overlap


class DetectionEngine:
    """Main detection engine for silkworm detection."""
    
    def __init__(self, args, cfg: Config):
        """
        Initialize detection engine.
        
        Args:
            args: Command line arguments
            cfg: Configuration object
        """
        self.args = args
        self.cfg = cfg
        
        # Initialize components
        self.model = self._setup_model()
        self.cap = self._setup_camera()
        self.tracker = self._setup_tracker()
        self.performance_monitor = PerformanceMonitor()
        
        # Detection state
        self.overlap_counters = {}
        self.head_history = {}
        self.freeze_counters = {}
        self.device = cfg.device if cfg.device is not None else 'cpu'
        
        print(f"Camera: 640x480 @ {args.fps} FPS")
        print(f"Inference size: {args.imgsz}")
        print(f"Config: conf={args.conf}, skip={args.skip}, pose_conf={cfg.pose_conf}")
    
    def _setup_model(self) -> YOLO:
        """Setup YOLO model."""
        model_path = self.args.model if self.args.model else self.cfg.model_path
        print(get_model_info(model_path))
        return YOLO(model_path)
    
    def _setup_camera(self):
        """Setup camera."""
        return setup_camera(self.args.camera, self.args.fps)
    
    def _setup_tracker(self) -> SilkwormTracker:
        """Setup object tracker."""
        return SilkwormTracker(
            max_distance=self.cfg.max_distance,
            max_disappeared=self.cfg.max_disappeared
        )
    
    def run(self):
        """Run the main detection loop."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Cannot read frame")
                    break
                
                # Record frame read
                self.performance_monitor.record_frame_read()
                
                # Skip frames for performance
                if self.performance_monitor.frame_count % self.args.skip != 0:
                    self.performance_monitor.start_frame()
                    continue
                
                # Process frame
                self._process_frame(frame)
                
                # Update frame count
                self.performance_monitor.start_frame()
                
                # Periodic cleanup
                if self.performance_monitor.should_cleanup():
                    self.performance_monitor.cleanup()
                
                # FPS control
                self._control_fps()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self._cleanup()
            self._print_summary()
    
    def _process_frame(self, frame):
        """Process a single frame."""
        # Inference
        inference_start = time.time()
        results = self.model.predict(
            frame, 
            imgsz=self.args.imgsz, 
            conf=self.args.conf,
            iou=self.cfg.iou_thresh,
            verbose=False,
            device=self.device
        )
        inference_time = time.time() - inference_start
        
        # Record performance
        if self.args.benchmark:
            self.performance_monitor.record_inference_time(inference_time)
        
        # Process detections
        result = results[0]
        silkworms = process_detections(result, frame, self.cfg, self.tracker)
        
        # Update total detections
        if result.boxes is not None:
            self.performance_monitor.record_detections(len(result.boxes))
        
        # Freeze detection
        for silkworm in silkworms:
            obj_id, head, body, tail, bbox = silkworm
            head_c = 1.0  # treat as confident detection for freeze logic gating
            process_freeze(obj_id, head, head_c, bbox, self.cfg, self.head_history, self.freeze_counters, frame)
        
        # Overlap detection
        if len(silkworms) >= 2:
            process_overlap(silkworms, self.overlap_counters, frame, self.cfg)
        
        # Performance display
        if self.args.benchmark:
            draw_performance_info(frame, inference_time, silkworms, self.performance_monitor.frame_count, self.performance_monitor)
        
        # Draw scale bar
        draw_scale_bar(frame)
        
        # Display
        if not self.args.no_display:
            if display_frame(frame):
                raise KeyboardInterrupt("User requested quit")
    
    def _control_fps(self):
        """Control FPS to target rate."""
        current_fps = self.performance_monitor.get_current_fps()
        if current_fps > self.args.fps:
            time.sleep(0.1)
    
    def _cleanup(self):
        """Cleanup resources."""
        self.cap.release()
        if not self.args.no_display:
            import cv2
            cv2.destroyAllWindows()
    
    def _print_summary(self):
        """Print performance summary."""
        if self.args.benchmark:
            self.performance_monitor.print_summary()
