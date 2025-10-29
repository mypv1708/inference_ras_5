"""
Performance monitoring utilities
Tracks and manages performance metrics for silkworm detection
"""

import time
import gc
import numpy as np
from typing import List


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, max_inference_samples: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_inference_samples: Maximum number of inference time samples to keep
        """
        self.max_inference_samples = max_inference_samples
        self.inference_times: List[float] = []
        self.frame_count = 0
        self.total_frames_read = 0  # Count all frames read from camera
        self.start_time = time.time()
        self.total_detections = 0
    
    def start_frame(self):
        """Mark the start of a new frame."""
        self.frame_count += 1
    
    def record_frame_read(self):
        """Record that a frame was read from camera."""
        self.total_frames_read += 1
    
    def record_inference_time(self, inference_time: float):
        """Record inference time for this frame."""
        self.inference_times.append(inference_time)
        # Limit memory usage by keeping only recent samples
        if len(self.inference_times) > self.max_inference_samples:
            self.inference_times = self.inference_times[-self.max_inference_samples:]
    
    def record_detections(self, detection_count: int):
        """Record number of detections for this frame."""
        self.total_detections += detection_count
    
    def should_cleanup(self) -> bool:
        """Check if it's time for garbage collection."""
        return self.frame_count % 1000 == 0
    
    def cleanup(self):
        """Perform garbage collection."""
        gc.collect()
    
    def get_current_fps(self) -> float:
        """Get current FPS based on elapsed time."""
        elapsed = time.time() - self.start_time
        return self.total_frames_read / elapsed if elapsed > 0 else 0
    
    def get_processing_fps(self) -> float:
        """Get processing FPS (frames actually processed)."""
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0
    
    def get_average_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times) * 1000
    
    def get_performance_summary(self) -> dict:
        """Get comprehensive performance summary."""
        total_time = time.time() - self.start_time
        avg_fps = self.total_frames_read / total_time if total_time > 0 else 0
        processing_fps = self.frame_count / total_time if total_time > 0 else 0
        
        return {
            'frames_read': self.total_frames_read,
            'frames_processed': self.frame_count,
            'total_time': total_time,
            'average_fps': avg_fps,
            'processing_fps': processing_fps,
            'total_detections': self.total_detections,
            'average_inference_ms': self.get_average_inference_time()
        }
    
    def print_summary(self):
        """Print performance summary to console."""
        summary = self.get_performance_summary()
        
        print(f"Frames read: {summary['frames_read']}, Processed: {summary['frames_processed']} in {summary['total_time']:.1f}s")
        print(f"Camera FPS: {summary['average_fps']:.1f}")
        print(f"Processing FPS: {summary['processing_fps']:.1f}")
        print(f"Total detections: {summary['total_detections']}")
        
        if self.inference_times:
            print(f"Average inference: {summary['average_inference_ms']:.1f}ms")
