"""
Display utilities for silkworm detection
Handles drawing performance info, scale bars, and display management
"""

import cv2
import psutil
from typing import List, Tuple


def draw_performance_info(frame, inference_time: float, silkworms: List[Tuple], frame_count: int, performance_monitor=None):
    """Draw performance information on frame."""
    # Calculate FPS based on performance monitor if available
    if performance_monitor:
        camera_fps = performance_monitor.get_current_fps()
        processing_fps = performance_monitor.get_processing_fps()
        fps_text = f"Cam: {camera_fps:.1f} | Proc: {processing_fps:.1f}"
    else:
        # Fallback to inference time based FPS
        fps = (1.0 / inference_time) if inference_time > 0 else 0
        fps_text = f"FPS: {fps:.1f}"
    
    # Memory usage
    memory_percent = psutil.virtual_memory().percent
    
    cv2.putText(frame, fps_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Detections: {len(silkworms)}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Memory: {memory_percent:.1f}%", (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_scale_bar(frame):
    """Draw a 50 px scale bar at bottom-left of the frame."""
    h, w = frame.shape[:2]
    x0, y0 = 50, h - 10
    x1 = x0 + 50  # 50 pixels long
    # Bar
    cv2.line(frame, (x0, y0), (x1, y0), (255, 255, 255), 2)
    # End caps for visibility
    cv2.line(frame, (x0, y0 - 4), (x0, y0 + 4), (255, 255, 255), 2)
    cv2.line(frame, (x1, y0 - 4), (x1, y0 + 4), (255, 255, 255), 2)
    # Label
    cv2.putText(frame, "50 px", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def display_frame(frame, window_name: str = "Silkworm Detection (Optimized)"):
    """Display frame and handle key input."""
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF
    return key == ord('q') or key == 27  # Return True if should quit
